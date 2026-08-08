"""Tests for the Israeli retail calendar.

The holiday dates below were cross-checked against the Hebrew calendar rather than
transcribed from memory. They are the reason this module exists: getting them wrong would
silently corrupt every seasonality estimate downstream.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from liftlab.israel_calendar import (
    HOLIDAY_FEATURES,
    holiday_frame,
    jewish_holiday_dates,
    weekly_seasonality,
)


class TestJewishHolidayDates:
    def test_known_dates(self):
        found = jewish_holiday_dates(date(2023, 1, 1), date(2025, 12, 31))
        assert date(2023, 9, 16) in found["rosh_hashana"]
        assert date(2024, 10, 3) in found["rosh_hashana"]
        assert date(2023, 9, 25) in found["yom_kippur"]
        assert date(2024, 10, 12) in found["yom_kippur"]
        assert date(2024, 4, 23) in found["pesach"]
        assert date(2025, 4, 13) in found["pesach"]

    def test_yom_kippur_is_nine_days_after_rosh_hashana(self):
        found = jewish_holiday_dates(date(2023, 1, 1), date(2026, 12, 31))
        for rh, yk in zip(found["rosh_hashana"], found["yom_kippur"], strict=True):
            assert (yk - rh).days == 9

    def test_one_of_each_per_year(self):
        found = jewish_holiday_dates(date(2023, 1, 1), date(2025, 12, 31))
        # Three full Gregorian years contain three of each.
        assert len(found["rosh_hashana"]) == 3
        assert len(found["yom_kippur"]) == 3
        assert len(found["pesach"]) == 3

    def test_dates_are_sorted_and_within_window(self):
        start, end = date(2024, 1, 1), date(2024, 12, 31)
        found = jewish_holiday_dates(start, end)
        for occurrences in found.values():
            assert occurrences == sorted(occurrences)
            assert all(start <= d <= end for d in occurrences)

    def test_narrow_window_excludes_everything(self):
        found = jewish_holiday_dates(date(2024, 1, 1), date(2024, 1, 31))
        assert all(len(v) == 0 for v in found.values())


class TestHolidayFrame:
    @pytest.fixture
    def frame(self):
        index = pd.date_range("2023-01-01", periods=1095, freq="D")
        return holiday_frame(index)

    def test_columns_and_alignment(self, frame):
        assert list(frame.columns) == list(HOLIDAY_FEATURES)
        assert len(frame) == 1095

    def test_yom_kippur_is_one_day_per_year(self, frame):
        assert frame["yom_kippur"].sum() == 3
        assert frame.loc["2023-09-25", "yom_kippur"] == 1.0

    def test_runup_peaks_on_the_eve_and_is_zero_on_the_day(self, frame):
        # Pesach 2024 falls on 23 April; the eve is the peak shopping day.
        assert frame.loc["2024-04-22", "pesach_runup"] == pytest.approx(1.0)
        assert frame.loc["2024-04-23", "pesach_runup"] == 0.0

    def test_runup_is_monotonically_increasing_toward_the_eve(self, frame):
        window = frame.loc["2024-04-10":"2024-04-22", "pesach_runup"]
        assert np.all(np.diff(window.to_numpy()) > 0)

    def test_runup_window_length_is_configurable(self):
        index = pd.date_range("2024-04-01", periods=30, freq="D")
        short = holiday_frame(index, runup_days=3)
        assert (short["pesach_runup"] > 0).sum() == 3

    def test_zero_runup_window_disables_the_ramp(self):
        index = pd.date_range("2024-04-01", periods=30, freq="D")
        flat = holiday_frame(index, runup_days=0)
        assert flat["pesach_runup"].sum() == 0.0

    def test_values_stay_in_unit_interval(self, frame):
        assert frame.to_numpy().min() >= 0.0
        assert frame.to_numpy().max() <= 1.0

    def test_empty_index_returns_empty_frame(self):
        empty = holiday_frame(pd.DatetimeIndex([]))
        assert len(empty) == 0
        assert list(empty.columns) == list(HOLIDAY_FEATURES)


class TestWeeklySeasonality:
    @pytest.fixture
    def index(self):
        return pd.date_range("2024-01-01", periods=28, freq="D")

    def test_saturday_is_the_trough(self, index):
        weekly = pd.Series(weekly_seasonality(index), index=index)
        by_day = weekly.groupby(index.dayofweek).mean()
        assert by_day.idxmin() == 5  # Saturday

    def test_thursday_is_the_peak(self, index):
        weekly = pd.Series(weekly_seasonality(index), index=index)
        by_day = weekly.groupby(index.dayofweek).mean()
        assert by_day.idxmax() == 3  # Thursday

    def test_averages_to_one_over_whole_weeks(self, index):
        assert float(np.mean(weekly_seasonality(index))) == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize(
        ("trough", "peak", "match"),
        [
            (0.0, 1.25, "saturday_trough"),
            (1.5, 1.25, "saturday_trough"),
            (0.45, 0.9, "thursday_peak"),
        ],
    )
    def test_rejects_invalid_parameters(self, index, trough, peak, match):
        with pytest.raises(ValueError, match=match):
            weekly_seasonality(index, saturday_trough=trough, thursday_peak=peak)
