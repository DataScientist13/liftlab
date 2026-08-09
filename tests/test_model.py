"""Tests for the MMM data preparation and NumPyro model.

Deliberately fast: these check that the model is wired correctly and samples, not that it
recovers parameters. Recovery is a separate, slower benchmark — conflating the two produces a
test suite too slow to run and a benchmark too shallow to trust.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from liftlab.dgp import DEFAULT_CHANNELS, DGPConfig, generate_panel
from liftlab.model import ChannelModel, ExperimentResult, fit, prepare_data


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


class TestExperimentResult:
    def test_rejects_non_positive_standard_error(self):
        with pytest.raises(ValueError, match="standard_error"):
            ExperimentResult("search", date(2023, 1, 1), date(2023, 2, 1), 1000.0, 0.0)

    def test_rejects_reversed_window(self):
        with pytest.raises(ValueError, match="end must not precede"):
            ExperimentResult("search", date(2023, 3, 1), date(2023, 1, 1), 1000.0, 10.0)


class TestCalibrationBridge:
    @pytest.fixture
    def experiment(self, panel):
        window = panel.data.index[10:40]
        true_incremental = float(panel.truth.contributions["search"].iloc[10:40].sum())
        return ExperimentResult(
            channel="search",
            start=window[0].date(),
            end=window[-1].date(),
            incremental_revenue=true_incremental,
            standard_error=0.2 * true_incremental,
        )

    def test_no_experiments_by_default(self, panel, channels):
        assert prepare_data(panel.data, channels).n_experiments == 0

    def test_mask_covers_exactly_the_window(self, panel, channels, experiment):
        data = prepare_data(panel.data, channels, experiments=(experiment,))
        assert data.n_experiments == 1
        assert data.experiment_mask.shape == (1, 120)
        assert data.experiment_mask.sum() == 30

    def test_channel_index_points_at_the_right_column(self, panel, channels, experiment):
        data = prepare_data(panel.data, channels, experiments=(experiment,))
        assert channels[int(data.experiment_channel[0])].slug == "search"

    def test_rejects_unknown_channel(self, panel, channels, experiment):
        bad = ExperimentResult("nosuchchannel", experiment.start, experiment.end, 1.0, 1.0)
        with pytest.raises(ValueError, match="unknown channel"):
            prepare_data(panel.data, channels, experiments=(bad,))

    def test_rejects_window_outside_the_panel(self, panel, channels):
        stale = ExperimentResult("search", date(1999, 1, 1), date(1999, 2, 1), 1.0, 1.0)
        with pytest.raises(ValueError, match="does not overlap"):
            prepare_data(panel.data, channels, experiments=(stale,))

    def test_fit_exposes_the_experiment_prediction(self, panel, channels, experiment):
        data = prepare_data(panel.data, channels, experiments=(experiment,))
        idata = fit(data, num_warmup=30, num_samples=30, num_chains=1, seed=0)
        predicted = idata.posterior["experiment_predicted"]
        assert predicted.shape == (1, 30, 1)
        assert np.all(predicted.values > 0)

    def test_a_precise_experiment_pulls_the_posterior_toward_it(self, panel, channels):
        """A tight experiment must move the channel's estimate more than a vague one.

        This is the property that makes the bridge worth having: the standard error, not
        the point estimate alone, controls how much the experiment is allowed to say.
        """
        true_incremental = float(panel.truth.contributions["search"].iloc[10:40].sum())
        window = panel.data.index[10:40]
        # Both experiments claim the same (deliberately inflated) effect; they differ only
        # in how confidently they claim it.
        claim = 3.0 * true_incremental

        def roas_with(relative_se: float) -> float:
            experiment = ExperimentResult(
                "search", window[0].date(), window[-1].date(), claim, relative_se * claim
            )
            data = prepare_data(panel.data, channels, experiments=(experiment,))
            idata = fit(data, num_warmup=150, num_samples=150, num_chains=2, seed=0)
            return float(idata.posterior["roas"].values[..., 0].mean())

        assert roas_with(0.05) > roas_with(2.0)


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
