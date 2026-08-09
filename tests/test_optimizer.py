"""Tests for the budget optimizer.

Built on hand-constructed response curves rather than a fitted model, so the right answer is
known analytically and a failure points at the optimizer rather than at the sampler.
"""

from __future__ import annotations

import numpy as np
import pytest

from liftlab.optimizer import Allocation, ResponseCurves, optimize_budget


def curves(
    *,
    beta: list[float],
    half_saturation: list[float],
    slope: list[float] | None = None,
    n_draws: int = 50,
    spend_scale: list[float] | None = None,
) -> ResponseCurves:
    """Build response curves with no posterior spread unless asked for."""
    n = len(beta)
    ones = np.ones((n_draws, n))
    return ResponseCurves(
        channels=tuple(f"ch{i}" for i in range(n)),
        beta=ones * np.array(beta),
        half_saturation=ones * np.array(half_saturation),
        slope=ones * np.array(slope if slope is not None else [1.0] * n),
        spend_scale=np.array(spend_scale if spend_scale is not None else [1.0] * n),
        revenue_scale=1.0,
    )


class TestResponseCurves:
    def test_revenue_increases_with_spend(self):
        c = curves(beta=[1.0], half_saturation=[1.0])
        assert (
            c.revenue_draws(np.array([2.0]), 1).mean() > c.revenue_draws(np.array([1.0]), 1).mean()
        )

    def test_revenue_scales_with_horizon(self):
        c = curves(beta=[1.0], half_saturation=[1.0])
        one = c.revenue_draws(np.array([1.0]), 1).mean()
        ten = c.revenue_draws(np.array([1.0]), 10).mean()
        assert ten == pytest.approx(10 * one)

    def test_half_saturation_gives_half_the_asymptote(self):
        c = curves(beta=[100.0], half_saturation=[5.0])
        assert c.revenue_draws(np.array([5.0]), 1).mean() == pytest.approx(50.0)

    def test_spend_scale_is_applied(self):
        # half_saturation is on scaled spend, so a scale of 1000 puts the half-way point
        # at 5000 currency units.
        c = curves(beta=[100.0], half_saturation=[5.0], spend_scale=[1000.0])
        assert c.revenue_draws(np.array([5000.0]), 1).mean() == pytest.approx(50.0)

    def test_draws_are_returned_per_draw(self):
        c = curves(beta=[1.0], half_saturation=[1.0], n_draws=37)
        assert c.revenue_draws(np.array([1.0]), 1).shape == (37,)


class TestOptimizeBudget:
    def test_sends_budget_to_the_more_productive_channel(self):
        # Identical curves except one returns twice as much.
        c = curves(beta=[100.0, 200.0], half_saturation=[10.0, 10.0])
        result = optimize_budget(c, total_daily_budget=10.0, n_days=1)
        allocation = result.to_frame()
        assert allocation["ch1"] > allocation["ch0"]

    def test_spends_the_whole_budget(self):
        c = curves(beta=[100.0, 50.0], half_saturation=[5.0, 8.0])
        result = optimize_budget(c, total_daily_budget=12.0, n_days=1)
        assert result.daily_spend.sum() == pytest.approx(12.0, abs=1e-6)

    def test_diminishing_returns_produce_a_split_not_a_corner(self):
        """Concave curves must spread budget; an all-in answer would signal a broken objective."""
        c = curves(beta=[100.0, 100.0], half_saturation=[4.0, 6.0])
        result = optimize_budget(c, total_daily_budget=20.0, n_days=1)
        assert np.all(result.daily_spend > 1.0)

    def test_respects_channel_bounds(self):
        c = curves(beta=[500.0, 10.0], half_saturation=[5.0, 5.0])
        result = optimize_budget(
            c,
            total_daily_budget=10.0,
            n_days=1,
            bounds={"ch0": (0.0, 3.0), "ch1": (2.0, 100.0)},
        )
        allocation = result.to_frame()
        assert allocation["ch0"] <= 3.0 + 1e-6
        assert allocation["ch1"] >= 2.0 - 1e-6
        assert result.daily_spend.sum() == pytest.approx(10.0, abs=1e-6)

    def test_beats_an_equal_split(self):
        c = curves(beta=[300.0, 60.0, 20.0], half_saturation=[6.0, 6.0, 6.0])
        result = optimize_budget(c, total_daily_budget=30.0, n_days=1)
        equal = c.revenue_draws(np.array([10.0, 10.0, 10.0]), 1).mean()
        assert result.expected_revenue > equal

    def test_finds_the_analytic_optimum_for_two_linear_ish_channels(self):
        """With slope=1 and equal half-saturation, budget should split in proportion to beta.

        For y = b*x/(k+x), the first-order condition equalises b*k/(k+x)^2 across channels,
        so with a shared k the optimal split satisfies x_i + k proportional to sqrt(b_i).
        """
        b0, b1, k, budget = 100.0, 400.0, 10.0, 30.0
        c = curves(beta=[b0, b1], half_saturation=[k, k])
        result = optimize_budget(c, total_daily_budget=budget, n_days=1)

        # Solve the first-order condition directly.
        ratio = np.sqrt(b1 / b0)
        x0 = (budget + k - ratio * k) / (1 + ratio)
        np.testing.assert_allclose(result.daily_spend[0], x0, rtol=1e-3)

    def test_s_curves_are_handled_by_restarts(self):
        """S-curves make the objective non-concave; restarts must still find a good point."""
        c = curves(beta=[100.0, 100.0], half_saturation=[8.0, 8.0], slope=[3.0, 3.0])
        many = optimize_budget(c, total_daily_budget=16.0, n_days=1, n_starts=8, seed=1)
        # An S-curve rewards concentrating spend past the threshold rather than splitting it.
        assert many.expected_revenue >= c.revenue_draws(np.array([8.0, 8.0]), 1).mean() - 1e-9

    @pytest.mark.parametrize("budget", [0.0, -5.0])
    def test_rejects_non_positive_budget(self, budget):
        c = curves(beta=[1.0], half_saturation=[1.0])
        with pytest.raises(ValueError, match="must be positive"):
            optimize_budget(c, total_daily_budget=budget)

    def test_rejects_minimums_that_exceed_the_budget(self):
        c = curves(beta=[1.0, 1.0], half_saturation=[1.0, 1.0])
        floors = {"ch0": (4.0, 10.0), "ch1": (4.0, 10.0)}
        with pytest.raises(ValueError, match="exceeds the daily budget"):
            optimize_budget(c, total_daily_budget=5.0, bounds=floors)

    def test_rejects_maximums_that_cannot_absorb_the_budget(self):
        c = curves(beta=[1.0, 1.0], half_saturation=[1.0, 1.0])
        ceilings = {"ch0": (0.0, 5.0), "ch1": (0.0, 5.0)}
        with pytest.raises(ValueError, match="cannot absorb"):
            optimize_budget(c, total_daily_budget=50.0, bounds=ceilings)

    def test_rejects_reversed_bounds(self):
        c = curves(beta=[1.0], half_saturation=[1.0])
        with pytest.raises(ValueError, match="invalid bounds"):
            optimize_budget(c, total_daily_budget=10.0, bounds={"ch0": (9.0, 2.0)})


class TestUncertaintyPropagation:
    def test_reports_an_interval_not_just_a_point(self):
        rng = np.random.default_rng(0)
        n_draws = 500
        spread = ResponseCurves(
            channels=("a", "b"),
            beta=np.column_stack(
                [rng.lognormal(np.log(100), 0.4, n_draws), rng.lognormal(np.log(80), 0.4, n_draws)]
            ),
            half_saturation=np.column_stack(
                [rng.lognormal(np.log(5), 0.3, n_draws), rng.lognormal(np.log(6), 0.3, n_draws)]
            ),
            slope=np.ones((n_draws, 2)),
            spend_scale=np.ones(2),
            revenue_scale=1.0,
        )
        result = optimize_budget(spread, total_daily_budget=12.0, n_days=7)
        low, high = result.revenue_interval(0.95)

        assert low < result.expected_revenue < high
        assert high > low
        assert result.revenue_interval(0.5)[0] > low

    def test_objective_is_the_mean_of_curves_not_the_curve_of_means(self):
        """Averaging parameters and averaging curves differ; the optimizer must do the latter.

        Hill response is concave in beta-weighted terms over this range, so Jensen's
        inequality makes the mean of the curves sit below the curve at the mean parameters.
        Using the latter would systematically overstate expected revenue.
        """
        rng = np.random.default_rng(1)
        n_draws = 2000
        half = rng.lognormal(np.log(5.0), 0.6, (n_draws, 1))
        spread = ResponseCurves(
            channels=("a",),
            beta=np.full((n_draws, 1), 100.0),
            half_saturation=half,
            slope=np.ones((n_draws, 1)),
            spend_scale=np.ones(1),
            revenue_scale=1.0,
        )
        mean_of_curves = spread.revenue_draws(np.array([5.0]), 1).mean()

        at_mean_parameters = (
            ResponseCurves(
                channels=("a",),
                beta=np.full((1, 1), 100.0),
                half_saturation=np.full((1, 1), float(half.mean())),
                slope=np.ones((1, 1)),
                spend_scale=np.ones(1),
                revenue_scale=1.0,
            )
            .revenue_draws(np.array([5.0]), 1)
            .mean()
        )

        assert mean_of_curves != pytest.approx(at_mean_parameters, rel=1e-3)


class _Variable:
    """Minimal stand-in for an xarray DataArray, exposing only `.values`."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _InferenceData:
    """Minimal stand-in for an ArviZ InferenceData."""

    def __init__(self, posterior: dict[str, _Variable]) -> None:
        self.posterior = posterior


def fake_idata(n_chains: int = 2, n_draws: int = 100, n_channels: int = 3) -> _InferenceData:
    rng = np.random.default_rng(0)
    shape = (n_chains, n_draws, n_channels)
    return _InferenceData(
        {
            "beta": _Variable(rng.lognormal(0.0, 0.2, shape)),
            "half_saturation": _Variable(rng.lognormal(0.0, 0.2, shape)),
            "slope": _Variable(rng.lognormal(0.0, 0.1, shape)),
        }
    )


class TestFromPosterior:
    """The seam between a fitted model and the optimizer — where a rename breaks silently."""

    def test_flattens_chains_and_draws(self):
        curves_ = ResponseCurves.from_posterior(
            fake_idata(n_chains=4, n_draws=50, n_channels=3),
            channels=("a", "b", "c"),
            spend_scale=np.ones(3),
            revenue_scale=1.0,
            max_draws=1_000,
        )
        assert curves_.beta.shape == (200, 3)
        assert curves_.n_channels == 3

    def test_thins_to_max_draws(self):
        curves_ = ResponseCurves.from_posterior(
            fake_idata(n_chains=2, n_draws=500, n_channels=2),
            channels=("a", "b"),
            spend_scale=np.ones(2),
            revenue_scale=1.0,
            max_draws=100,
        )
        assert curves_.beta.shape == (100, 2)
        assert curves_.half_saturation.shape == (100, 2)
        assert curves_.slope.shape == (100, 2)

    def test_thinning_is_reproducible(self):
        def thinned(seed: int) -> ResponseCurves:
            return ResponseCurves.from_posterior(
                fake_idata(n_draws=200, n_channels=2),
                channels=("a", "b"),
                spend_scale=np.ones(2),
                revenue_scale=1.0,
                max_draws=40,
                seed=seed,
            )

        np.testing.assert_array_equal(thinned(7).beta, thinned(7).beta)
        # A different seed selects different draws, so the match above is not vacuous.
        assert not np.array_equal(thinned(7).beta, thinned(8).beta)

    def test_carries_scale_factors_through(self):
        curves_ = ResponseCurves.from_posterior(
            fake_idata(n_channels=2),
            channels=("a", "b"),
            spend_scale=np.array([1_000.0, 2_000.0]),
            revenue_scale=50_000.0,
        )
        np.testing.assert_array_equal(curves_.spend_scale, [1_000.0, 2_000.0])
        assert curves_.revenue_scale == 50_000.0

    def test_optimises_directly_from_a_posterior(self):
        curves_ = ResponseCurves.from_posterior(
            fake_idata(n_channels=3),
            channels=("a", "b", "c"),
            spend_scale=np.full(3, 1_000.0),
            revenue_scale=100_000.0,
        )
        plan = optimize_budget(curves_, total_daily_budget=3_000.0, n_days=30)
        assert plan.daily_spend.sum() == pytest.approx(3_000.0, abs=1e-6)
        assert plan.expected_revenue > 0
        low, high = plan.revenue_interval()
        assert low < plan.expected_revenue < high


class TestConstraintRepair:
    def test_budget_is_met_exactly_even_with_tight_bounds(self):
        """SLSQP leaves small equality drift; the repair step must not violate bounds."""
        c = curves(beta=[100.0, 100.0, 100.0], half_saturation=[5.0, 5.0, 5.0])
        result = optimize_budget(
            c,
            total_daily_budget=9.0,
            n_days=1,
            bounds={"ch0": (2.9, 3.1), "ch1": (2.9, 3.1), "ch2": (2.9, 3.1)},
        )
        assert result.daily_spend.sum() == pytest.approx(9.0, abs=1e-9)
        assert np.all(result.daily_spend >= 2.9 - 1e-9)
        assert np.all(result.daily_spend <= 3.1 + 1e-9)


class TestAllocation:
    def test_totals_and_roas(self):
        allocation = Allocation(
            channels=("a", "b"),
            daily_spend=np.array([4.0, 6.0]),
            n_days=10,
            revenue_draws=np.array([200.0, 300.0, 400.0]),
            converged=True,
        )
        assert allocation.total_budget == pytest.approx(100.0)
        assert allocation.expected_revenue == pytest.approx(300.0)
        assert allocation.expected_roas == pytest.approx(3.0)
        assert allocation.to_frame() == {"a": 4.0, "b": 6.0}
