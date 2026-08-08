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

**Channel coefficients are hierarchical and non-centered.** Partial pooling across channels
regularises the small, low-spend ones. The non-centered form avoids the funnel geometry that
otherwise produces divergences at small ``tau``.

**Half-saturation is on the spend scale.** After scaling, a value of 1.0 means "half response
at median spend", which is a quantity a media buyer can actually hold an opinion about.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def n_periods(self) -> int:
        """Number of observations."""
        return int(self.spend.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of media channels."""
        return int(self.spend.shape[1])


def _fourier_terms(index: pd.DatetimeIndex, period: float, order: int) -> np.ndarray:
    """Build sine/cosine pairs for a seasonal period."""
    day = np.arange(len(index), dtype=np.float64)
    terms: list[np.ndarray] = []
    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * day / period
        terms.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(terms)


def prepare_data(
    panel: pd.DataFrame,
    channels: tuple[ChannelModel, ...],
    *,
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

    # --- Hierarchical channel coefficients, non-centered ---------------------------------
    mu_beta = numpyro.sample("mu_beta", dist.Normal(jnp.log(0.08), 0.7))
    tau_beta = numpyro.sample("tau_beta", dist.HalfNormal(1.0))
    z_beta = numpyro.sample("z_beta", dist.Normal(0.0, 1.0).expand([n_channels]).to_event(1))
    beta = numpyro.deterministic("beta", jnp.exp(mu_beta + tau_beta * z_beta))

    contribution = beta[None, :] * response

    # --- Yom Kippur closure applies to total demand, not just the baseline ---------------
    open_fraction = 1.0 - closure * jnp.asarray(data.yom_kippur)
    mu = (baseline + contribution.sum(axis=-1)) * open_fraction

    # Reported in currency units so downstream code never has to redo the scaling.
    numpyro.deterministic(
        "roas",
        (contribution * open_fraction[:, None]).sum(axis=0)
        * data.revenue_scale
        / (spend.sum(axis=0) * jnp.asarray(data.spend_scale)),
    )

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))
    obs = None if data.revenue is None else jnp.asarray(data.revenue)
    numpyro.sample("revenue", dist.LogNormal(jnp.log(mu), sigma), obs=obs)


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
