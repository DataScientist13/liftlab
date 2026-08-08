"""Tests for the synthetic data-generating process.

The recovery benchmark's credibility rests on this DGP being reproducible and on its truth
matching what it actually generated, so those are the properties tested hardest here.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from liftlab.dgp import (
    DEFAULT_CHANNELS,
    ChannelSpec,
    DGPConfig,
    GeometricAdstock,
    WeibullAdstock,
    generate_panel,
)


@pytest.fixture(scope="module")
def panel():
    return generate_panel(DGPConfig(n_periods=1095, start=date(2023, 1, 1)))


class TestReproducibility:
    def test_same_seed_gives_identical_panels(self):
        a = generate_panel(DGPConfig(n_periods=200, seed=7))
        b = generate_panel(DGPConfig(n_periods=200, seed=7))
        assert a.data.equals(b.data)

    def test_different_seed_gives_different_panels(self):
        a = generate_panel(DGPConfig(n_periods=200, seed=7))
        b = generate_panel(DGPConfig(n_periods=200, seed=8))
        assert not a.data["revenue"].equals(b.data["revenue"])


class TestPanelShape:
    def test_has_a_spend_column_per_channel_plus_revenue(self, panel):
        expected = {f"spend_{c.slug}" for c in DEFAULT_CHANNELS} | {"revenue"}
        assert set(panel.data.columns) == expected

    def test_daily_index_of_requested_length(self, panel):
        assert len(panel.data) == 1095
        assert panel.data.index.freqstr == "D"
        assert panel.data.index[0] == np.datetime64("2023-01-01")

    def test_revenue_and_spend_are_strictly_positive(self, panel):
        assert (panel.data > 0).all().all()

    def test_default_config_covers_three_years(self):
        cfg = DGPConfig()
        assert cfg.n_periods == 1095

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_periods": 0}, "n_periods"),
            ({"channels": ()}, "at least one channel"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            generate_panel(DGPConfig(**kwargs))


class TestIdentifiabilityProperties:
    """Spend must carry information, or nothing downstream is estimable."""

    def test_spend_varies_substantially(self, panel):
        for channel in DEFAULT_CHANNELS:
            spend = panel.data[f"spend_{channel.slug}"]
            assert spend.std() / spend.mean() > 0.2

    def test_campaign_bursts_actually_occur(self, panel):
        for channel in DEFAULT_CHANNELS:
            spend = panel.data[f"spend_{channel.slug}"]
            # A flight should push spend well above the always-on level.
            assert spend.max() > 2.0 * spend.median()


class TestCalendarEffects:
    def test_yom_kippur_collapses_revenue(self, panel):
        yom_kippur = panel.data.loc["2023-09-25", "revenue"]
        nearby = panel.data.loc["2023-09-18":"2023-09-24", "revenue"].mean()
        assert yom_kippur < 0.25 * nearby

    def test_pre_pesach_revenue_is_elevated(self, panel):
        eve_window = panel.data.loc["2024-04-16":"2024-04-22", "revenue"].mean()
        ordinary = panel.data.loc["2024-02-01":"2024-02-28", "revenue"].mean()
        assert eve_window > 1.2 * ordinary

    def test_saturday_is_the_weekly_trough(self, panel):
        by_day = panel.data["revenue"].groupby(panel.data.index.dayofweek).mean()
        assert by_day.idxmin() == 5


class TestGroundTruth:
    def test_contributions_align_with_the_panel(self, panel):
        assert panel.truth.contributions.index.equals(panel.data.index)
        assert set(panel.truth.contributions.columns) == {c.slug for c in DEFAULT_CHANNELS}

    def test_contributions_stay_below_the_channel_ceiling(self, panel):
        # hill_saturation is bounded below 1, so contribution < coefficient always.
        for channel in DEFAULT_CHANNELS:
            assert panel.truth.contributions[channel.slug].max() < channel.coefficient

    def test_decomposition_reconstructs_revenue_up_to_noise(self, panel):
        reconstructed = panel.truth.baseline_series + panel.truth.contributions.sum(axis=1)
        ratio = panel.data["revenue"] / reconstructed
        # Only multiplicative noise separates the two, and it is median-preserving.
        assert ratio.median() == pytest.approx(1.0, abs=0.01)
        assert ratio.std() == pytest.approx(panel.truth.config.noise_cv, abs=0.02)

    def test_true_roas_is_positive_and_finite(self, panel):
        for channel in DEFAULT_CHANNELS:
            roas = panel.truth.roas[channel.slug]
            assert np.isfinite(roas)
            assert roas > 0

    def test_parameter_table_reports_every_channel(self, panel):
        table = panel.truth.parameter_table()
        assert list(table.index) == [c.slug for c in DEFAULT_CHANNELS]
        assert table.loc["search", "adstock_kind"] == "GeometricAdstock"
        assert table.loc["tv", "adstock_kind"] == "WeibullAdstock"
        # Geometric channels have no Weibull parameters and vice versa.
        assert np.isnan(table.loc["search", "weibull_shape"])
        assert np.isnan(table.loc["tv", "decay"])
        assert table.loc["search", "decay"] == pytest.approx(0.15)


class TestCustomChannels:
    def test_single_weibull_channel_round_trips(self):
        channel = ChannelSpec(
            slug="ooh",
            display_name_he="שילוט חוצות",
            base_spend=3_000.0,
            adstock=WeibullAdstock(shape=1.8, scale=5.0, max_lag=14),
            half_saturation=6_000.0,
            slope=1.5,
            coefficient=15_000.0,
        )
        panel = generate_panel(DGPConfig(n_periods=120, channels=(channel,)))
        assert "spend_ooh" in panel.data.columns
        assert panel.truth.parameter_table().loc["ooh", "weibull_scale"] == pytest.approx(5.0)

    def test_zero_carryover_channel_is_supported(self):
        channel = ChannelSpec(
            slug="affiliate",
            display_name_he="שותפים",
            base_spend=1_000.0,
            adstock=GeometricAdstock(decay=0.0),
            half_saturation=1_200.0,
            slope=1.0,
            coefficient=5_000.0,
        )
        panel = generate_panel(DGPConfig(n_periods=90, channels=(channel,)))
        assert panel.truth.roas["affiliate"] > 0
