"""Synthetic Israeli e-commerce data-generating process.

The recovery benchmark is the load-bearing claim of this project: simulate data from a
process whose parameters are known, fit the model, and report how close the estimates land
and whether the credible intervals cover at their nominal rate. That claim is only worth
anything if the DGP is fully specified and reproducible, which is what this module is for.

Design notes that matter for identifiability
--------------------------------------------
**Spend must vary.** Adstock decay and saturation are estimated from how response bends as
spend moves. A channel held at a constant budget carries almost no information about either,
and a model fit to flat spend will happily report a confident, arbitrary answer. Spend here
is therefore generated with both day-to-day lognormal variation and discrete campaign
flights, mimicking real always-on-plus-burst media plans.

**Saturation is on the spend scale.** ``half_saturation`` is expressed in the same currency
units as spend, so it can be reasoned about (and given a prior) by someone who buys media.

**Channel coefficients are a revenue ceiling.** :func:`liftlab.hill_saturation` returns a
value in ``[0, 1)``, so ``coefficient`` is the maximum daily incremental revenue a channel
could contribute at infinite spend. Realised contribution is always below it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from liftlab.israel_calendar import holiday_frame, weekly_seasonality
from liftlab.transforms import geometric_adstock, hill_saturation, weibull_adstock

__all__ = [
    "DEFAULT_CHANNELS",
    "ChannelSpec",
    "DGPConfig",
    "GeometricAdstock",
    "GroundTruth",
    "SyntheticPanel",
    "WeibullAdstock",
    "generate_panel",
]


@dataclass(frozen=True)
class GeometricAdstock:
    """Exponential carryover with a constant retention rate."""

    decay: float


@dataclass(frozen=True)
class WeibullAdstock:
    """Carryover with a Weibull kernel, allowing the peak to fall after lag zero."""

    shape: float
    scale: float
    max_lag: int = 21


AdstockSpec = GeometricAdstock | WeibullAdstock


@dataclass(frozen=True)
class ChannelSpec:
    """Truth for one media channel.

    Attributes
    ----------
    slug
        ASCII identifier used for column names.
    display_name_he
        Hebrew channel name, as it would appear in an Israeli media plan.
    base_spend
        Median daily spend in currency units when no campaign is running.
    adstock
        Carryover specification.
    half_saturation
        Spend level at which response reaches half its ceiling, on the spend scale.
    slope
        Hill coefficient. Above 1 gives an S-curve, at or below 1 a concave curve.
    coefficient
        Maximum daily incremental revenue attributable to this channel.
    burst_rate
        Probability per day that a campaign flight starts.
    burst_length
        Length of a campaign flight in days.
    burst_multiplier
        Spend multiplier while a flight is active.
    """

    slug: str
    display_name_he: str
    base_spend: float
    adstock: AdstockSpec
    half_saturation: float
    slope: float
    coefficient: float
    burst_rate: float = 0.02
    burst_length: int = 10
    burst_multiplier: float = 3.0


#: A five-channel Israeli e-commerce media plan, spanning fast and slow carryover.
DEFAULT_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        slug="search",
        display_name_he="חיפוש ממומן",
        base_spend=9_000.0,
        # Search intent is immediate: almost no carryover.
        adstock=GeometricAdstock(decay=0.15),
        half_saturation=11_000.0,
        slope=1.1,
        coefficient=52_000.0,
        burst_rate=0.01,
    ),
    ChannelSpec(
        slug="social",
        display_name_he="מדיה חברתית",
        base_spend=7_000.0,
        adstock=GeometricAdstock(decay=0.45),
        half_saturation=9_000.0,
        slope=1.4,
        coefficient=38_000.0,
    ),
    ChannelSpec(
        slug="video",
        display_name_he="וידאו",
        base_spend=4_000.0,
        adstock=GeometricAdstock(decay=0.65),
        half_saturation=7_500.0,
        slope=1.8,
        coefficient=26_000.0,
        burst_rate=0.03,
    ),
    ChannelSpec(
        slug="tv",
        display_name_he="טלוויזיה",
        base_spend=2_500.0,
        # TV response builds and peaks days after the flight starts.
        adstock=WeibullAdstock(shape=2.2, scale=6.0, max_lag=21),
        half_saturation=12_000.0,
        slope=2.0,
        coefficient=31_000.0,
        burst_rate=0.02,
        burst_length=21,
        burst_multiplier=6.0,
    ),
    ChannelSpec(
        slug="email",
        display_name_he="דיוור אלקטרוני",
        base_spend=900.0,
        adstock=GeometricAdstock(decay=0.30),
        half_saturation=1_100.0,
        slope=1.0,
        coefficient=9_500.0,
        burst_rate=0.04,
        burst_length=3,
        burst_multiplier=2.5,
    ),
)


@dataclass(frozen=True)
class DGPConfig:
    """Configuration for one synthetic panel.

    Attributes
    ----------
    n_periods
        Number of daily observations.
    start
        First calendar day of the panel.
    channels
        Channel truths.
    baseline
        Daily organic revenue before seasonality, trend, and media.
    annual_growth
        Fractional year-on-year growth of the baseline, e.g. ``0.08`` for 8%.
    pesach_lift, rosh_hashana_lift
        Peak multiplicative uplift at the top of each pre-holiday ramp.
    yom_kippur_multiplier
        Multiplier applied on Yom Kippur. Near zero: commerce stops.
    noise_cv
        Coefficient of variation of the multiplicative revenue noise.
    seed
        Seed for the random generator. Identical seeds reproduce identical panels.
    """

    n_periods: int = 1_095
    start: date = date(2023, 1, 1)
    channels: tuple[ChannelSpec, ...] = DEFAULT_CHANNELS
    baseline: float = 120_000.0
    annual_growth: float = 0.08
    pesach_lift: float = 0.55
    rosh_hashana_lift: float = 0.40
    yom_kippur_multiplier: float = 0.03
    noise_cv: float = 0.06
    seed: int = 20260809


@dataclass(frozen=True)
class GroundTruth:
    """Known parameters and realised decomposition behind a synthetic panel.

    Attributes
    ----------
    config
        The configuration that produced the panel.
    contributions
        Per-channel incremental revenue actually realised, indexed by date.
    baseline_series
        Realised organic baseline before media and noise.
    roas
        Realised true ROAS per channel: total contribution divided by total spend. This is
        the quantity an MMM is ultimately trying to recover, so it is the headline number in
        the recovery report.
    """

    config: DGPConfig
    contributions: pd.DataFrame
    baseline_series: pd.Series
    roas: dict[str, float] = field(default_factory=dict)

    def parameter_table(self) -> pd.DataFrame:
        """Return true parameters per channel as a tidy frame for reporting."""
        rows = []
        for channel in self.config.channels:
            adstock = channel.adstock
            rows.append(
                {
                    "channel": channel.slug,
                    "display_name_he": channel.display_name_he,
                    "adstock_kind": type(adstock).__name__,
                    "decay": adstock.decay if isinstance(adstock, GeometricAdstock) else np.nan,
                    "weibull_shape": (
                        adstock.shape if isinstance(adstock, WeibullAdstock) else np.nan
                    ),
                    "weibull_scale": (
                        adstock.scale if isinstance(adstock, WeibullAdstock) else np.nan
                    ),
                    "half_saturation": channel.half_saturation,
                    "slope": channel.slope,
                    "coefficient": channel.coefficient,
                    "true_roas": self.roas.get(channel.slug, np.nan),
                }
            )
        return pd.DataFrame(rows).set_index("channel")


@dataclass(frozen=True)
class SyntheticPanel:
    """A generated panel together with the truth that produced it."""

    data: pd.DataFrame
    truth: GroundTruth


def _apply_adstock(spend: np.ndarray, spec: AdstockSpec) -> np.ndarray:
    """Dispatch to the adstock transform named by ``spec``.

    Geometric adstock is normalised so decay does not silently rescale effect size, which
    would confound it with the channel coefficient.
    """
    if isinstance(spec, GeometricAdstock):
        return geometric_adstock(spend, spec.decay, normalize=True)
    return weibull_adstock(spend, spec.shape, spec.scale, max_lag=spec.max_lag)


def _simulate_spend(
    rng: np.random.Generator,
    channel: ChannelSpec,
    n_periods: int,
) -> np.ndarray:
    """Simulate a daily spend series with always-on noise plus campaign flights."""
    # Always-on component: lognormal around the base budget.
    log_noise = rng.normal(loc=0.0, scale=0.25, size=n_periods)
    spend = channel.base_spend * np.exp(log_noise - 0.5 * 0.25**2)

    # Campaign flights: discrete bursts that give the model the spend variation it needs
    # to separate carryover from saturation.
    active = np.zeros(n_periods, dtype=bool)
    starts = rng.random(n_periods) < channel.burst_rate
    for t in np.flatnonzero(starts):
        active[t : t + channel.burst_length] = True
    spend = np.where(active, spend * channel.burst_multiplier, spend)

    return np.asarray(spend, dtype=np.float64)


def generate_panel(config: DGPConfig | None = None) -> SyntheticPanel:
    """Generate a synthetic Israeli e-commerce panel with known parameters.

    Parameters
    ----------
    config
        DGP configuration. Defaults to :class:`DGPConfig`, which is three years of daily
        data across five channels.

    Returns
    -------
    SyntheticPanel
        ``data`` holds one row per day with a ``spend_<slug>`` column per channel and a
        ``revenue`` column. ``truth`` holds the parameters and the realised decomposition.

    Examples
    --------
    >>> panel = generate_panel()
    >>> panel.data.columns.tolist()[:2]
    ['spend_search', 'spend_social']
    """
    cfg = config if config is not None else DGPConfig()
    if cfg.n_periods < 1:
        msg = "n_periods must be at least 1"
        raise ValueError(msg)
    if not cfg.channels:
        msg = "at least one channel is required"
        raise ValueError(msg)

    rng = np.random.default_rng(cfg.seed)
    index = pd.date_range(start=cfg.start, periods=cfg.n_periods, freq="D", name="date")

    # --- Organic baseline: level, trend, weekly cycle, holidays ------------------------
    t = np.arange(cfg.n_periods, dtype=np.float64)
    trend = (1.0 + cfg.annual_growth) ** (t / 365.25)
    weekly = weekly_seasonality(index)

    holidays = holiday_frame(index)
    # Pre-holiday surges are demand-side: organic appetite to buy rises ahead of the chag.
    runup_multiplier = (
        1.0
        + cfg.pesach_lift * holidays["pesach_runup"].to_numpy()
        + cfg.rosh_hashana_lift * holidays["rosh_hashana_runup"].to_numpy()
    )
    baseline = cfg.baseline * trend * weekly * runup_multiplier

    # Yom Kippur is supply-side: the shops are shut, so it suppresses *total* revenue rather
    # than only the organic part. Advertising that ran beforehand does not sell anything on a
    # day when nobody can buy. Applied after media is added, below.
    is_yom_kippur = holidays["yom_kippur"].to_numpy() > 0.0
    closure_multiplier = np.where(is_yom_kippur, cfg.yom_kippur_multiplier, 1.0)

    # --- Media contributions -----------------------------------------------------------
    spend_columns: dict[str, np.ndarray] = {}
    contribution_columns: dict[str, np.ndarray] = {}
    roas: dict[str, float] = {}

    for channel in cfg.channels:
        spend = _simulate_spend(rng, channel, cfg.n_periods)
        carried = _apply_adstock(spend, channel.adstock)
        response = hill_saturation(carried, channel.half_saturation, channel.slope)
        contribution = channel.coefficient * response * closure_multiplier

        spend_columns[f"spend_{channel.slug}"] = spend
        contribution_columns[channel.slug] = contribution
        total_spend = float(spend.sum())
        roas[channel.slug] = float(contribution.sum()) / total_spend if total_spend > 0 else np.nan

    media_total = np.sum(list(contribution_columns.values()), axis=0)

    # --- Observation noise -------------------------------------------------------------
    # Multiplicative and median-preserving, so noise scales with the level and revenue
    # cannot go negative.
    sigma = np.sqrt(np.log1p(cfg.noise_cv**2))
    noise = np.exp(rng.normal(loc=-0.5 * sigma**2, scale=sigma, size=cfg.n_periods))
    baseline = baseline * closure_multiplier
    revenue = (baseline + media_total) * noise

    data = pd.DataFrame(spend_columns, index=index)
    data["revenue"] = revenue

    truth = GroundTruth(
        config=cfg,
        contributions=pd.DataFrame(contribution_columns, index=index),
        baseline_series=pd.Series(baseline, index=index, name="baseline"),
        roas=roas,
    )
    return SyntheticPanel(data=data, truth=truth)
