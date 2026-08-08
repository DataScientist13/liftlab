"""Tests for the MMM data preparation and NumPyro model.

Deliberately fast: these check that the model is wired correctly and samples, not that it
recovers parameters. Recovery is a separate, slower benchmark — conflating the two produces a
test suite too slow to run and a benchmark too shallow to trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liftlab.dgp import DEFAULT_CHANNELS, DGPConfig, generate_panel
from liftlab.model import ChannelModel, fit, prepare_data


@pytest.fixture(scope="module")
def panel():
    return generate_panel(DGPConfig(n_periods=120, seed=5))


@pytest.fixture(scope="module")
def channels():
    return tuple(
        ChannelModel(slug=c.slug, adstock="weibull" if c.slug == "tv" else "geometric")
        for c in DEFAULT_CHANNELS
    )


class TestPrepareData:
    def test_shapes(self, panel, channels):
        data = prepare_data(panel.data, channels)
        assert data.spend.shape == (120, 5)
        assert data.revenue is not None
        assert data.revenue.shape == (120,)
        assert data.n_periods == 120
        assert data.n_channels == 5

    def test_weekly_fourier_has_two_columns_per_harmonic(self, panel, channels):
        data = prepare_data(panel.data, channels, weekly_order=3)
        assert data.weekly_fourier.shape == (120, 6)

    def test_spend_is_median_scaled(self, panel, channels):
        data = prepare_data(panel.data, channels)
        # Median of each scaled channel is 1 by construction, which is what makes the
        # half-saturation prior interpretable.
        np.testing.assert_allclose(np.median(data.spend, axis=0), 1.0, rtol=1e-12)

    def test_revenue_is_mean_scaled(self, panel, channels):
        data = prepare_data(panel.data, channels)
        assert data.revenue is not None
        assert float(np.mean(data.revenue)) == pytest.approx(1.0)
        assert data.revenue_scale == pytest.approx(panel.data["revenue"].mean())

    def test_scaling_is_invertible(self, panel, channels):
        data = prepare_data(panel.data, channels)
        recovered = data.spend * data.spend_scale
        np.testing.assert_allclose(
            recovered[:, 0], panel.data["spend_search"].to_numpy(), rtol=1e-12
        )

    def test_time_is_centred_and_in_years(self, panel, channels):
        data = prepare_data(panel.data, channels)
        assert float(np.mean(data.time)) == pytest.approx(0.0, abs=1e-12)
        assert data.time.max() - data.time.min() == pytest.approx(119 / 365.25)

    def test_holiday_regressors_are_present(self, panel, channels):
        data = prepare_data(panel.data, channels)
        for arr in (data.pesach, data.rosh_hashana, data.yom_kippur):
            assert arr.shape == (120,)

    def test_missing_revenue_column_is_allowed(self, panel, channels):
        without = panel.data.drop(columns=["revenue"])
        data = prepare_data(without, channels)
        assert data.revenue is None
        assert data.revenue_scale == 1.0

    def test_rejects_empty_channels(self, panel):
        with pytest.raises(ValueError, match="at least one channel"):
            prepare_data(panel.data, ())

    def test_rejects_missing_spend_column(self, panel):
        with pytest.raises(ValueError, match="missing spend columns"):
            prepare_data(panel.data, (ChannelModel(slug="nonexistent"),))

    def test_rejects_non_datetime_index(self, channels):
        frame = pd.DataFrame({f"spend_{c.slug}": [1.0, 2.0] for c in channels})
        frame["revenue"] = [10.0, 11.0]
        with pytest.raises(TypeError, match="DatetimeIndex"):
            prepare_data(frame, channels)

    def test_rejects_channel_with_zero_median_spend(self, panel, channels):
        dark = panel.data.copy()
        dark["spend_email"] = 0.0
        with pytest.raises(ValueError, match="positive median spend"):
            prepare_data(dark, channels)


@pytest.fixture(scope="module")
def idata(panel, channels):
    """One very small fit, shared across the assertions that inspect it."""
    data = prepare_data(panel.data, channels)
    return fit(data, num_warmup=40, num_samples=40, num_chains=1, seed=0)


class TestFit:
    """Proves the model is traceable and samples end to end."""

    def test_returns_expected_posterior_variables(self, idata):
        for name in ("beta", "decay", "half_saturation", "slope", "sigma", "roas"):
            assert name in idata.posterior

    def test_posterior_shapes_match_channel_count(self, idata):
        assert idata.posterior["beta"].shape == (1, 40, 5)
        # Four geometric channels; TV uses Weibull and so has shape/scale instead.
        assert idata.posterior["decay"].shape == (1, 40, 4)
        assert idata.posterior["weibull_shape"].shape == (1, 40, 1)

    def test_all_posterior_draws_are_finite(self, idata):
        for name in ("beta", "half_saturation", "slope", "roas"):
            assert np.all(np.isfinite(idata.posterior[name].values))

    def test_roas_is_positive(self, idata):
        assert np.all(idata.posterior["roas"].values > 0)

    def test_decay_stays_in_the_unit_interval(self, idata):
        decay = idata.posterior["decay"].values
        assert np.all((decay >= 0.0) & (decay < 1.0))

    def test_fitting_without_revenue_is_refused(self, panel, channels):
        data = prepare_data(panel.data.drop(columns=["revenue"]), channels)
        with pytest.raises(ValueError, match="requires observed revenue"):
            fit(data, num_warmup=1, num_samples=1, num_chains=1)


class TestAllGeometricPanel:
    def test_model_runs_without_any_weibull_channel(self, panel):
        channels = tuple(ChannelModel(slug=c.slug) for c in DEFAULT_CHANNELS)
        data = prepare_data(panel.data, channels)
        idata = fit(data, num_warmup=20, num_samples=20, num_chains=1, seed=0)
        assert idata.posterior["decay"].shape == (1, 20, 5)
        assert "weibull_shape" not in idata.posterior


class TestAllWeibullPanel:
    def test_model_runs_without_any_geometric_channel(self, panel):
        channels = tuple(
            ChannelModel(slug=c.slug, adstock="weibull", max_lag=10) for c in DEFAULT_CHANNELS
        )
        data = prepare_data(panel.data, channels)
        idata = fit(data, num_warmup=20, num_samples=20, num_chains=1, seed=0)
        assert idata.posterior["weibull_shape"].shape == (1, 20, 5)
        assert "decay" not in idata.posterior
