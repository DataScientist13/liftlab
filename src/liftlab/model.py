"""Hierarchical Bayesian marketing mix model in NumPyro.

Structure mirrors the data-generating process in :mod:`liftlab.dgp`, because a recovery
benchmark only isolates *estimation* error if the model can in principle represent the truth.
Where the model deliberately differs from the DGP, it is noted below — those gaps are part of
what the benchmark measures.

The response curve
------------------
::

    baseline_t = intercept * exp(growth * t) * weekly_t * (1 + a * pesach_t + b * rosh_t)
    media_t    = sum_c  beta_c * hill(adstock(spend_ct))
    mu_t       = (baseline_t + media_t) * (1 - closure * yom_kippur_t)
    revenue_t ~ LogNormal(log(mu_t), sigma)

Parameterisation choices that matter
------------------------------------
**Everything is scaled.** Spend is divided by its per-channel median and revenue by its mean,
so priors sit on an interpretable, unit-free scale and NUTS sees a well-conditioned geometry.
Currency-scale quantities are recovered afterwards from the stored scale factors.

**Adstock is normalised.** Without it, decay silently rescales effect size and becomes
confounded with the channel coefficient — the single most common identifiability failure in
MMM implementations.

**The channel level is parameterised as ROAS, not as the Hill asymptote.** Realised ROAS is
what the data identifies and what an advertiser can hold a prior opinion about; the asymptote
``beta`` is derived from it. The earlier hierarchical prior on ``beta`` put the prior's weight
on the least identified quantity in the model, and the recovery benchmark measured the cost.

**Half-saturation is on the spend scale.** After scaling, a value of 1.0 means "half response
at median spend", which is a quantity a media buyer can actually hold an opinion about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_median

from liftlab.israel_calendar import holiday_frame
from liftlab.jax_transforms import geometric_adstock, hill_saturation, weibull_adstock

if TYPE_CHECKING:  # pragma: no cover - typing only
    import arviz as az

__all__ = [
    "ChannelModel",
    "ExperimentResult",
    "MMMData",
    "fit",
    "mmm_model",
    "prepare_data",
]

AdstockKind = Literal["geometric", "weibull"]


@dataclass(frozen=True)
class ChannelModel:
    """How one channel is represented in the model.

    Attributes
    ----------
    slug
        Must match a ``spend_<slug>`` column in the panel.
    adstock
        Carryover family. Chosen by the analyst rather than estimated: the data rarely
        distinguishes the families cleanly, and pretending otherwise invites a bimodal
        posterior.
    max_lag
        Kernel length for Weibull carryover. Ignored for geometric.
    """

    slug: str
    adstock: AdstockKind = "geometric"
    max_lag: int = 21


@dataclass(frozen=True)
class ExperimentResult:
    """A completed incrementality experiment for one channel over one window.

    This is the calibration bridge's input. ``liftlab`` does not run or analyse the
    experiment — the number comes from wherever the test was actually measured (a geo
    holdout, Meta Conversion Lift, a Google geo experiment, an in-house synthetic control).
    What matters here is that it is an estimate of *incremental revenue* with an honest
    standard error, on the same currency scale as the panel's revenue column.

    Attributes
    ----------
    channel
        Slug of the channel the experiment tested.
    start, end
        Inclusive window the experiment covered, in the panel's calendar.
    incremental_revenue
        Estimated incremental revenue attributable to the channel over the window.
    standard_error
        Standard error of that estimate. This is what controls how hard the experiment
        pulls the model: a noisy test moves the posterior very little, which is the correct
        behaviour and the reason the bridge takes an interval rather than a point.
    """

    channel: str
    start: date
    end: date
    incremental_revenue: float
    standard_error: float

    def __post_init__(self) -> None:
        """Reject an experiment that cannot carry information."""
        if self.standard_error <= 0:
            msg = "standard_error must be positive"
            raise ValueError(msg)
        if self.end < self.start:
            msg = "experiment end must not precede its start"
            raise ValueError(msg)


@dataclass(frozen=True)
class MMMData:
    """Model-ready arrays plus the scale factors needed to return to currency units."""

    spend: np.ndarray
    weekly_fourier: np.ndarray
    pesach: np.ndarray
    rosh_hashana: np.ndarray
    yom_kippur: np.ndarray
    time: np.ndarray
    channels: tuple[ChannelModel, ...]
    spend_scale: np.ndarray
    revenue_scale: float
    revenue: np.ndarray | None = None
    experiment_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    experiment_channel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    experiment_value: np.ndarray = field(default_factory=lambda: np.zeros(0))
    experiment_se: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_periods(self) -> int:
        """Number of observations."""
        return int(self.spend.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of media channels."""
        return int(self.spend.shape[1])

    @property
    def n_experiments(self) -> int:
        """Number of calibrating experiments."""
        return int(self.experiment_value.shape[0])


def _fourier_terms(index: pd.DatetimeIndex, period: float, order: int) -> np.ndarray:
    """Build sine/cosine pairs for a seasonal period."""
    day = np.arange(len(index), dtype=np.float64)
    terms: list[np.ndarray] = []
    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * day / period
        terms.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(terms)


def _experiment_arrays(
    index: pd.DatetimeIndex,
    channels: tuple[ChannelModel, ...],
    experiments: tuple[ExperimentResult, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn experiment windows into masks aligned to the panel index."""
    slugs = [c.slug for c in channels]
    n_periods = len(index)
    days = index.date

    masks = np.zeros((len(experiments), n_periods), dtype=np.float64)
    channel_index = np.zeros(len(experiments), dtype=np.int32)
    values = np.zeros(len(experiments), dtype=np.float64)
    errors = np.zeros(len(experiments), dtype=np.float64)

    for position, experiment in enumerate(experiments):
        if experiment.channel not in slugs:
            msg = f"experiment references unknown channel {experiment.channel!r}"
            raise ValueError(msg)
        mask = (days >= experiment.start) & (days <= experiment.end)
        if not mask.any():
            msg = (
                f"experiment window {experiment.start}..{experiment.end} for "
                f"{experiment.channel!r} does not overlap the panel"
            )
            raise ValueError(msg)
        masks[position] = mask.astype(np.float64)
        channel_index[position] = slugs.index(experiment.channel)
        values[position] = experiment.incremental_revenue
        errors[position] = experiment.standard_error

    return masks, channel_index, values, errors


def prepare_data(
    panel: pd.DataFrame,
    channels: tuple[ChannelModel, ...],
    *,
    experiments: tuple[ExperimentResult, ...] = (),
    revenue_column: str = "revenue",
    weekly_order: int = 3,
) -> MMMData:
    """Scale a panel and build the model's design arrays.

    Parameters
    ----------
    panel
        Daily frame indexed by date, with one ``spend_<slug>`` column per channel and a
        revenue column. Revenue may be absent when preparing data for prediction.
    channels
        Channel specifications, in the order the model should use.
    experiments
        Completed incrementality experiments used to calibrate the model. Each becomes an
        extra likelihood term tying the model's implied incremental revenue over the test
        window to what the experiment measured.
    revenue_column
        Name of the revenue column. Ignored if absent.
    weekly_order
        Number of Fourier harmonics for the weekly cycle. Three captures the Israeli
        Sunday-to-Thursday shape including the Saturday trough.

    Returns
    -------
    MMMData
        Scaled arrays plus the factors needed to invert the scaling.
    """
    if not channels:
        msg = "at least one channel is required"
        raise ValueError(msg)
    if not isinstance(panel.index, pd.DatetimeIndex):
        msg = "panel must be indexed by a DatetimeIndex"
        raise TypeError(msg)

    missing = [c.slug for c in channels if f"spend_{c.slug}" not in panel.columns]
    if missing:
        msg = f"panel is missing spend columns for: {', '.join(missing)}"
        raise ValueError(msg)

    spend_raw = panel[[f"spend_{c.slug}" for c in channels]].to_numpy(dtype=np.float64)
    spend_scale = np.median(spend_raw, axis=0)
    if np.any(spend_scale <= 0):
        msg = "every channel needs positive median spend to be identifiable"
        raise ValueError(msg)

    if revenue_column in panel.columns:
        revenue_raw = panel[revenue_column].to_numpy(dtype=np.float64)
        revenue_scale = float(np.mean(revenue_raw))
        revenue = revenue_raw / revenue_scale
    else:
        revenue_scale = 1.0
        revenue = None

    holidays = holiday_frame(panel.index)
    n = len(panel)
    exp_mask, exp_channel, exp_value, exp_se = _experiment_arrays(
        panel.index, channels, experiments
    )

    return MMMData(
        spend=spend_raw / spend_scale,
        weekly_fourier=_fourier_terms(panel.index, period=7.0, order=weekly_order),
        pesach=holidays["pesach_runup"].to_numpy(),
        rosh_hashana=holidays["rosh_hashana_runup"].to_numpy(),
        yom_kippur=holidays["yom_kippur"].to_numpy(),
        # Exactly centred and scaled to years, so `growth` is a yearly rate and the
        # intercept refers to the midpoint of the window rather than its start.
        time=(np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / 365.25,
        channels=channels,
        spend_scale=spend_scale,
        revenue_scale=revenue_scale,
        revenue=revenue,
        experiment_mask=exp_mask,
        experiment_channel=exp_channel,
        experiment_value=exp_value,
        experiment_se=exp_se,
    )


def mmm_model(data: MMMData) -> None:
    """NumPyro model for the hierarchical MMM.

    Parameters
    ----------
    data
        Output of :func:`prepare_data`. When ``data.revenue`` is ``None`` the model runs as a
        generative prior/posterior predictive.
    """
    n_channels = data.n_channels
    spend = jnp.asarray(data.spend)

    # --- Organic baseline ---------------------------------------------------------------
    # Baseline is expressed as a share of mean revenue, so the prior says "most revenue is
    # organic" without committing to a currency scale.
    intercept = numpyro.sample("intercept", dist.LogNormal(jnp.log(0.6), 0.4))
    growth = numpyro.sample("growth", dist.Normal(0.0, 0.2))
    weekly_beta = numpyro.sample(
        "weekly_beta",
        dist.Normal(0.0, 0.3).expand([data.weekly_fourier.shape[1]]).to_event(1),
    )
    pesach_lift = numpyro.sample("pesach_lift", dist.HalfNormal(0.6))
    rosh_hashana_lift = numpyro.sample("rosh_hashana_lift", dist.HalfNormal(0.6))
    # Yom Kippur removes most of a day's revenue; the prior is informative about that.
    closure = numpyro.sample("closure", dist.Beta(6.0, 2.0))

    seasonal = jnp.exp(jnp.asarray(data.weekly_fourier) @ weekly_beta)
    trend = jnp.exp(growth * jnp.asarray(data.time))
    runup = (
        1.0
        + pesach_lift * jnp.asarray(data.pesach)
        + rosh_hashana_lift * jnp.asarray(data.rosh_hashana)
    )
    baseline = intercept * trend * seasonal * runup

    # --- Media transforms ---------------------------------------------------------------
    geo_idx = jnp.array(
        [i for i, c in enumerate(data.channels) if c.adstock == "geometric"], dtype=jnp.int32
    )
    wei_idx = jnp.array(
        [i for i, c in enumerate(data.channels) if c.adstock == "weibull"], dtype=jnp.int32
    )

    carried = jnp.zeros_like(spend)
    if geo_idx.size > 0:
        decay = numpyro.sample("decay", dist.Beta(2.0, 3.0).expand([int(geo_idx.size)]).to_event(1))
        carried = carried.at[:, geo_idx].set(
            geometric_adstock(spend[:, geo_idx], decay, normalize=True)
        )
    if wei_idx.size > 0:
        wei_shape = numpyro.sample(
            "weibull_shape",
            dist.LogNormal(jnp.log(2.0), 0.3).expand([int(wei_idx.size)]).to_event(1),
        )
        wei_scale = numpyro.sample(
            "weibull_scale",
            dist.LogNormal(jnp.log(6.0), 0.4).expand([int(wei_idx.size)]).to_event(1),
        )
        max_lag = max(c.max_lag for c in data.channels if c.adstock == "weibull")
        carried = carried.at[:, wei_idx].set(
            weibull_adstock(spend[:, wei_idx], wei_shape, wei_scale, max_lag=max_lag)
        )

    # Spend is median-scaled, so a half-saturation of 1.0 means "half response at typical
    # spend" and the prior is genuinely elicitable.
    # Centred above median spend, because channels are normally operated below their
    # saturation point, and wide because the data often cannot pin it: a channel whose spend
    # never approaches saturation looks linear, and its half-saturation is then informed
    # mostly by this prior. LogNormal(0, 0.6) was too tight — it put a half-saturation of
    # four times median spend, which is unremarkable for upper-funnel media, at +2.4 sd.
    half_saturation = numpyro.sample(
        "half_saturation", dist.LogNormal(jnp.log(1.5), 0.8).expand([n_channels]).to_event(1)
    )
    slope = numpyro.sample(
        "slope", dist.LogNormal(jnp.log(1.3), 0.35).expand([n_channels]).to_event(1)
    )
    response = hill_saturation(carried, half_saturation, slope)

    # --- Yom Kippur closure applies to total demand, not just the baseline ---------------
    open_fraction = 1.0 - closure * jnp.asarray(data.yom_kippur)

    # --- Channel level, parameterised as ROAS --------------------------------------------
    # The sampled level parameter is each channel's realised ROAS — total incremental
    # revenue over total spend — and the Hill asymptote `beta` is *derived* from it. Two
    # reasons, both measured on this model rather than assumed:
    #
    # 1. Identification. `beta` is the least identified quantity in the system: below
    #    saturation the likelihood constrains only the product beta * k**-s, so the prior
    #    on `beta` does the work where the data cannot, and the recovery benchmark showed
    #    the result — beta biased +40%, half_saturation +53%, ROAS +26%, driven by the
    #    weakly identified low-spend channels. Realised ROAS, by contrast, is essentially
    #    the regression level of the contribution series, which the data pins well. With
    #    the level sampled directly, k and slope only carry curve *shape*.
    #
    # 2. Elicitability. ROAS is the one level quantity with a defensible prior that does
    #    not reference this DGP: incremental ROAS above ~6 is rare in published lift
    #    experiments, and below ~0.3 the channel is burning money. LogNormal(log 1.5, 0.7)
    #    spans [0.38, 5.9] at 95%. A prior on the abstract asymptote of a saturation curve
    #    is a number nobody can hold an opinion about.
    roas = numpyro.sample(
        "roas", dist.LogNormal(jnp.log(1.5), 0.7).expand([n_channels]).to_event(1)
    )
    total_spend = spend.sum(axis=0) * jnp.asarray(data.spend_scale)
    delivered_response = (response * open_fraction[:, None]).sum(axis=0)
    beta = numpyro.deterministic(
        "beta",
        roas * total_spend / (data.revenue_scale * jnp.maximum(delivered_response, 1e-9)),
    )
    contribution = beta[None, :] * response

    mu = (baseline + contribution.sum(axis=-1)) * open_fraction

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))
    obs = None if data.revenue is None else jnp.asarray(data.revenue)
    numpyro.sample("revenue", dist.LogNormal(jnp.log(mu), sigma), obs=obs)

    # --- Calibration bridge -------------------------------------------------------------
    # Each experiment enters as an additional observation: what the model implies the
    # channel contributed over the test window, against what the experiment measured.
    #
    # This is deliberately *not* implemented as a hand-derived prior on beta. The mapping
    # from a lift measurement to a channel coefficient runs through adstock and saturation,
    # so it depends on decay, half-saturation and slope as well — parameters that are
    # themselves uncertain. Adding a likelihood term lets NUTS propagate the experiment's
    # information through that whole mapping, which a prior placed directly on beta cannot.
    # The standard error does the work: a noisy experiment barely moves the posterior.
    if data.n_experiments > 0:
        contribution_currency = contribution * open_fraction[:, None] * data.revenue_scale
        windowed = jnp.asarray(data.experiment_mask) @ contribution_currency
        predicted = windowed[jnp.arange(data.n_experiments), jnp.asarray(data.experiment_channel)]
        numpyro.deterministic("experiment_predicted", predicted)
        numpyro.sample(
            "experiment",
            dist.Normal(predicted, jnp.asarray(data.experiment_se)),
            obs=jnp.asarray(data.experiment_value),
        )


def fit(
    data: MMMData,
    *,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 4,
    seed: int = 0,
    target_accept_prob: float = 0.9,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Fit the MMM with NUTS and return an ArviZ ``InferenceData``.

    Parameters
    ----------
    data
        Output of :func:`prepare_data`, with observed revenue.
    num_warmup, num_samples, num_chains
        Sampler budget. Four chains is the minimum that makes R-hat meaningful.
    seed
        PRNG seed.
    target_accept_prob
        Raised above the NumPyro default because adstock and saturation produce curved
        geometry; the default step size tends to yield divergences here.
    progress_bar
        Show the NumPyro progress bar.

    Returns
    -------
    arviz.InferenceData
        Posterior samples and sampler diagnostics.

    Notes
    -----
    Enables JAX 64-bit mode as a global side effect. The adstock recursion runs over a
    thousand time steps and accumulates visible error in float32.

    Chains run in parallel only if this is the first JAX work in the process.
    ``set_host_device_count`` sets an XLA flag that is read when the backend initialises, so
    calling it afterwards is silently a no-op and NumPyro falls back to sampling chains
    sequentially. Results are identical either way — only wall-clock time changes — but it
    is why a second fit in the same session takes roughly twice as long per chain.
    """
    import arviz as az

    if data.revenue is None:
        msg = "fitting requires observed revenue; prepare_data found no revenue column"
        raise ValueError(msg)

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    numpyro.set_host_device_count(num_chains)

    # init_to_median beats the default init_to_uniform here: a random start can land in the
    # flat tail of the Hill curve, where the gradient is ~0 and warmup never recovers.
    kernel = NUTS(
        mmm_model,
        target_accept_prob=target_accept_prob,
        init_strategy=init_to_median,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return az.from_numpyro(mcmc)
