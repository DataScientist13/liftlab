"""Budget allocation over fitted response curves, carrying posterior uncertainty through.

The usual failure mode of an MMM-driven budget recommendation is to collapse the posterior to
its mean, optimise against that single response curve, and present the result as if it were
certain. That throws away the only thing the Bayesian machinery bought, and — as the recovery
benchmark shows — the point estimates are the least trustworthy part of the fit.

This module optimises the *posterior mean of revenue* rather than the revenue implied by the
posterior mean of the parameters. Those are not the same thing: the response curve is concave
in spend over most of its range, so by Jensen's inequality averaging curves and averaging
parameters give different answers, and the second one is systematically over-optimistic about
channels whose saturation is uncertain.

Every allocation is returned with the posterior distribution of its incremental revenue, so a
recommendation can be reported as a range rather than a number.

Method
------
Response for one posterior draw ``d`` and channel ``c`` at daily spend ``x``::

    revenue_dc(x) = beta_dc * hill(x / spend_scale_c ; k_dc, slope_dc) * n_days

The objective is the mean over draws of the summed channel revenue, maximised subject to a
total budget and per-channel bounds. It is smooth, and concave in each channel wherever
``slope <= 1``; with S-curves (``slope > 1``) it is not globally concave, so the optimiser is
started from several points and the best local solution is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from liftlab.transforms import hill_saturation

__all__ = [
    "Allocation",
    "ResponseCurves",
    "optimize_budget",
]


@dataclass(frozen=True)
class ResponseCurves:
    """Posterior draws of each channel's response curve, on the currency spend scale.

    Attributes
    ----------
    channels
        Channel slugs, in column order.
    beta, half_saturation, slope
        Posterior draws shaped ``(n_draws, n_channels)``. ``beta`` is on the model's scaled
        revenue units; ``half_saturation`` is on scaled spend.
    spend_scale
        Per-channel median spend used to scale the panel, shaped ``(n_channels,)``.
    revenue_scale
        Mean revenue used to scale the panel.
    """

    channels: tuple[str, ...]
    beta: np.ndarray
    half_saturation: np.ndarray
    slope: np.ndarray
    spend_scale: np.ndarray
    revenue_scale: float

    @classmethod
    def from_posterior(
        cls,
        idata: object,
        channels: tuple[str, ...],
        spend_scale: np.ndarray,
        revenue_scale: float,
        *,
        max_draws: int = 400,
        seed: int = 0,
    ) -> ResponseCurves:
        """Extract response-curve draws from an ArviZ ``InferenceData``.

        Parameters
        ----------
        idata
            Result of :func:`liftlab.model.fit`.
        channels
            Channel slugs in the order the model used.
        spend_scale, revenue_scale
            Scale factors from :class:`liftlab.model.MMMData`.
        max_draws
            Thin the posterior to at most this many draws. The optimizer evaluates every
            draw at every step, so this trades precision for speed; 400 draws puts Monte
            Carlo error on the objective well below the posterior width.
        seed
            Seed for thinning.
        """
        posterior = idata.posterior  # type: ignore[attr-defined]

        def draws(name: str) -> np.ndarray:
            values = np.asarray(posterior[name].values, dtype=np.float64)
            return values.reshape(-1, values.shape[-1])

        beta = draws("beta")
        half = draws("half_saturation")
        slope = draws("slope")
        if beta.shape[0] > max_draws:
            rng = np.random.default_rng(seed)
            keep = rng.choice(beta.shape[0], size=max_draws, replace=False)
            beta, half, slope = beta[keep], half[keep], slope[keep]

        return cls(
            channels=channels,
            beta=beta,
            half_saturation=half,
            slope=slope,
            spend_scale=np.asarray(spend_scale, dtype=np.float64),
            revenue_scale=float(revenue_scale),
        )

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self.channels)

    def revenue_draws(self, daily_spend: np.ndarray, n_days: int) -> np.ndarray:
        """Incremental revenue per posterior draw for a daily spend vector.

        Parameters
        ----------
        daily_spend
            Currency spend per day per channel, shaped ``(n_channels,)``.
        n_days
            Length of the planning horizon.

        Returns
        -------
        numpy.ndarray
            Total incremental revenue per draw, shaped ``(n_draws,)``.
        """
        scaled = np.asarray(daily_spend, dtype=np.float64) / self.spend_scale
        response = hill_saturation(
            np.broadcast_to(scaled, self.beta.shape),
            self.half_saturation,
            self.slope,
        )
        per_channel = self.beta * response * self.revenue_scale * n_days
        return np.asarray(per_channel.sum(axis=1), dtype=np.float64)


@dataclass(frozen=True)
class Allocation:
    """An optimised budget split and the posterior of what it is expected to return."""

    channels: tuple[str, ...]
    daily_spend: np.ndarray
    n_days: int
    revenue_draws: np.ndarray
    converged: bool

    @property
    def total_budget(self) -> float:
        """Total spend over the horizon."""
        return float(self.daily_spend.sum() * self.n_days)

    @property
    def expected_revenue(self) -> float:
        """Posterior mean incremental revenue."""
        return float(self.revenue_draws.mean())

    def revenue_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Central credible interval for incremental revenue."""
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(self.revenue_draws, [tail, 1.0 - tail])
        return float(low), float(high)

    @property
    def expected_roas(self) -> float:
        """Posterior mean incremental revenue divided by total spend."""
        budget = self.total_budget
        return self.expected_revenue / budget if budget > 0 else float("nan")

    def to_frame(self) -> dict[str, float]:
        """Return the allocation as a channel-to-daily-spend mapping."""
        return dict(zip(self.channels, (float(v) for v in self.daily_spend), strict=True))


def optimize_budget(
    curves: ResponseCurves,
    total_daily_budget: float,
    *,
    n_days: int = 30,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_starts: int = 5,
    seed: int = 0,
) -> Allocation:
    """Allocate a daily budget across channels to maximise expected incremental revenue.

    Parameters
    ----------
    curves
        Posterior response curves.
    total_daily_budget
        Daily spend to allocate. The constraint binds with equality: spending less is never
        better under a monotone response curve, so the solver is not asked to decide whether
        to spend.
    n_days
        Planning horizon, used only to scale reported revenue.
    bounds
        Optional per-channel ``(min, max)`` daily spend. Channels absent from the mapping are
        bounded below at zero and above at the total budget. Real plans always have these —
        contractual minimums, inventory ceilings — and an optimizer without them happily
        recommends putting the entire budget into one channel far outside its observed range.
    n_starts
        Number of random restarts. With S-curves the objective is not concave, so a single
        start can land on a local optimum.
    seed
        Seed for the restarts.

    Returns
    -------
    Allocation
        Optimised split with the posterior of its incremental revenue.

    Raises
    ------
    ValueError
        If the budget is not positive, or the bounds make it infeasible.
    """
    if total_daily_budget <= 0:
        msg = "total_daily_budget must be positive"
        raise ValueError(msg)

    n_channels = curves.n_channels
    lower = np.zeros(n_channels)
    upper = np.full(n_channels, float(total_daily_budget))
    for position, slug in enumerate(curves.channels):
        if bounds is not None and slug in bounds:
            low, high = bounds[slug]
            if low < 0 or high < low:
                msg = f"invalid bounds for {slug!r}: ({low}, {high})"
                raise ValueError(msg)
            lower[position], upper[position] = low, min(high, total_daily_budget)

    if lower.sum() > total_daily_budget + 1e-9:
        msg = (
            f"channel minimums total {lower.sum():,.0f}, which exceeds the daily budget "
            f"of {total_daily_budget:,.0f}"
        )
        raise ValueError(msg)
    if upper.sum() < total_daily_budget - 1e-9:
        msg = (
            f"channel maximums total {upper.sum():,.0f}, which cannot absorb the daily "
            f"budget of {total_daily_budget:,.0f}"
        )
        raise ValueError(msg)

    def negative_expected_revenue(x: np.ndarray) -> float:
        return -float(curves.revenue_draws(x, n_days).mean())

    constraint = {"type": "eq", "fun": lambda x: float(x.sum() - total_daily_budget)}
    rng = np.random.default_rng(seed)

    best: np.ndarray | None = None
    best_value = np.inf
    converged = False

    for start in range(n_starts):
        if start == 0:
            # Feasible, neutral starting point: minimums plus an equal share of the slack.
            guess = lower + (total_daily_budget - lower.sum()) / n_channels
        else:
            weights = rng.dirichlet(np.ones(n_channels))
            guess = lower + weights * (total_daily_budget - lower.sum())
        guess = np.clip(guess, lower, upper)

        result = minimize(
            negative_expected_revenue,
            guess,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints=[constraint],
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best = np.asarray(result.x)
            converged = bool(result.success)

    if best is None:  # pragma: no cover - n_starts >= 1 guarantees a solution
        msg = "optimisation produced no candidate; n_starts must be at least 1"
        raise ValueError(msg)

    # Repair any small constraint drift SLSQP leaves behind before reporting.
    best = np.clip(best, lower, upper)
    drift = total_daily_budget - best.sum()
    if abs(drift) > 1e-6:
        headroom = upper - best if drift > 0 else best - lower
        share = headroom / headroom.sum() if headroom.sum() > 0 else np.zeros(n_channels)
        best = best + np.sign(drift) * share * abs(drift)

    return Allocation(
        channels=curves.channels,
        daily_spend=best,
        n_days=n_days,
        revenue_draws=curves.revenue_draws(best, n_days),
        converged=converged,
    )
