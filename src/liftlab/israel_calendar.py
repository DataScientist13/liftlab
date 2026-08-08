"""Israeli retail calendar features.

Israeli retail does not follow the Gregorian rhythm that most MMM tooling assumes. Three
effects dominate and none of them sit on a fixed Gregorian date:

- **Shabbat.** Most non-food retail closes from Friday evening to Saturday evening, so the
  weekly cycle troughs on Saturday rather than Sunday.
- **Pre-holiday surges.** The two weeks before Pesach and Rosh Hashana are the largest
  grocery and gifting periods of the year.
- **Yom Kippur.** The country effectively stops. Not a dip — a hard zero.

Because the Hebrew calendar is lunisolar, these dates move by weeks across Gregorian years.
Hard-coding them is how portfolio projects quietly ship wrong numbers, so they are derived
from the Hebrew calendar via :mod:`pyluach` instead.

Notes
-----
Hebrew months are numbered from Nisan (month 1) while Hebrew *years* increment at Tishrei
(month 7). Within a single Hebrew year, Tishrei therefore falls in an *earlier* Gregorian
year than Nisan: Rosh Hashana 5784 is September 2023, but Pesach 5784 is April 2024. The
year-range scan below is deliberately padded to avoid dropping holidays at the edges.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pyluach import dates as hdates

__all__ = [
    "HOLIDAY_FEATURES",
    "holiday_frame",
    "jewish_holiday_dates",
    "weekly_seasonality",
]

_NISAN = 1
_TISHREI = 7

#: Columns produced by :func:`holiday_frame`, in a stable order.
HOLIDAY_FEATURES: tuple[str, ...] = (
    "pesach_runup",
    "rosh_hashana_runup",
    "yom_kippur",
)


def _hebrew_years_covering(start: date, end: date) -> range:
    """Return the Hebrew years whose holidays can fall inside a Gregorian window.

    Padded by one year on each side because a Hebrew year straddles two Gregorian ones.
    """
    first = hdates.GregorianDate(start.year, start.month, start.day).to_heb().year
    last = hdates.GregorianDate(end.year, end.month, end.day).to_heb().year
    return range(first - 1, last + 2)


def jewish_holiday_dates(start: date, end: date) -> dict[str, list[date]]:
    """Compute Gregorian dates of the retail-relevant Jewish holidays in a window.

    Parameters
    ----------
    start, end
        Inclusive bounds of the Gregorian window.

    Returns
    -------
    dict
        Maps ``"pesach"``, ``"rosh_hashana"``, and ``"yom_kippur"`` to the first day of each
        occurrence falling within the window, in ascending order.
    """
    spec = {
        "pesach": (_NISAN, 15),
        "rosh_hashana": (_TISHREI, 1),
        "yom_kippur": (_TISHREI, 10),
    }
    out: dict[str, list[date]] = {name: [] for name in spec}

    for hebrew_year in _hebrew_years_covering(start, end):
        for name, (month, day) in spec.items():
            gregorian = hdates.HebrewDate(hebrew_year, month, day).to_greg().to_pydate()
            if start <= gregorian <= end:
                out[name].append(gregorian)

    for name in out:
        out[name].sort()
    return out


def _ramp_before(
    index: pd.DatetimeIndex,
    holidays: list[date],
    window_days: int,
) -> np.ndarray:
    """Build a linear ramp rising to 1.0 on the day before each holiday.

    The surge is a run-*up*: demand builds over the fortnight before the holiday and
    collapses on the day itself, when shops close. The ramp therefore covers
    ``[holiday - window_days, holiday - 1]`` and is zero on the holiday.
    """
    values = np.zeros(len(index), dtype=np.float64)
    if window_days < 1:
        return values

    day = index.normalize()
    for holiday in holidays:
        stamp = pd.Timestamp(holiday)
        # Subtract index-from-scalar (not the reverse) to get a TimedeltaIndex, then flip
        # the sign so `offset` counts days remaining *until* the holiday.
        offset = -((day - stamp).days.to_numpy())
        in_window = (offset >= 1) & (offset <= window_days)
        # offset == window_days -> start of ramp; offset == 1 -> peak, the eve.
        strength = (window_days - offset + 1) / window_days
        values = np.maximum(values, np.where(in_window, strength, 0.0))
    return values


def _on_day(index: pd.DatetimeIndex, holidays: list[date]) -> np.ndarray:
    """Return a 0/1 indicator for the holiday day itself."""
    day = index.normalize()
    stamps = {pd.Timestamp(h) for h in holidays}
    return np.fromiter((1.0 if d in stamps else 0.0 for d in day), dtype=np.float64, count=len(day))


def holiday_frame(index: pd.DatetimeIndex, *, runup_days: int = 14) -> pd.DataFrame:
    """Build Israeli holiday regressors for a daily date index.

    Parameters
    ----------
    index
        Daily ``DatetimeIndex``. Times of day are ignored.
    runup_days
        Length of the pre-holiday surge window, in days.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``index``, with the columns in :data:`HOLIDAY_FEATURES`.
        ``pesach_runup`` and ``rosh_hashana_runup`` ramp from just above 0 to 1.0 on the eve;
        ``yom_kippur`` is a 0/1 indicator for the day itself.

    Notes
    -----
    Ramadan is **not** modelled here. It matters for retail in mixed and Arab-majority
    localities and belongs in a geo-resolved model, but it needs the Hijri calendar and
    locality weighting. Leaving it out and saying so is preferable to approximating it.
    """
    if len(index) == 0:
        return pd.DataFrame(
            {name: np.zeros(0, dtype=np.float64) for name in HOLIDAY_FEATURES},
            index=index,
        )

    start = index.min().date()
    end = index.max().date()
    holidays = jewish_holiday_dates(start, end)

    return pd.DataFrame(
        {
            "pesach_runup": _ramp_before(index, holidays["pesach"], runup_days),
            "rosh_hashana_runup": _ramp_before(index, holidays["rosh_hashana"], runup_days),
            "yom_kippur": _on_day(index, holidays["yom_kippur"]),
        },
        index=index,
    )


def weekly_seasonality(
    index: pd.DatetimeIndex,
    *,
    saturday_trough: float = 0.45,
    thursday_peak: float = 1.25,
) -> np.ndarray:
    """Build a multiplicative Israeli weekly retail cycle.

    The Israeli working week runs Sunday to Thursday. Thursday and Friday morning carry the
    pre-Shabbat shop; Saturday is the trough.

    Parameters
    ----------
    index
        Daily ``DatetimeIndex``.
    saturday_trough
        Multiplier applied on Saturday. Below 1 by construction.
    thursday_peak
        Multiplier applied on Thursday, the pre-Shabbat peak.

    Returns
    -------
    numpy.ndarray
        Multipliers aligned to ``index``, averaging approximately 1.0 over a full week.
    """
    if not 0.0 < saturday_trough < 1.0:
        msg = "saturday_trough must lie in (0, 1)"
        raise ValueError(msg)
    if thursday_peak <= 1.0:
        msg = "thursday_peak must exceed 1"
        raise ValueError(msg)

    # pandas dayofweek: Monday=0 ... Saturday=5, Sunday=6.
    profile = {
        6: 1.00,  # Sunday, start of the working week
        0: 0.95,  # Monday
        1: 0.95,  # Tuesday
        2: 1.00,  # Wednesday
        3: thursday_peak,  # Thursday
        4: 1.05,  # Friday, short trading day
        5: saturday_trough,  # Shabbat
    }
    raw = np.array([profile[d] for d in index.dayofweek], dtype=np.float64)
    # Normalise so the cycle rescales the level rather than shifting it.
    mean_multiplier = float(np.mean(list(profile.values())))
    return raw / mean_multiplier
