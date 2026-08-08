"""JAX implementations of the media transforms, for use inside NumPyro models.

The NumPy transforms in :mod:`liftlab.transforms` are the readable reference implementation
and the thing tests reason about. They cannot run inside NUTS: the sampler needs traceable,
differentiable operations, and the geometric recursion has to become a :func:`jax.lax.scan`
rather than a Python loop.

Keeping two implementations is a real cost, so it is paid down by testing them against each
other: :mod:`tests.test_jax_transforms` asserts agreement to floating-point tolerance for
every transform. If they ever diverge, the tests say so.

All functions here take spend shaped ``(n_periods, n_channels)`` and per-channel parameters
shaped ``(n_channels,)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "geometric_adstock",
    "hill_saturation",
    "weibull_adstock",
    "weibull_weights",
]


def geometric_adstock(spend: Array, decay: Array, *, normalize: bool = True) -> Array:
    """Apply geometric carryover along the time axis.

    Parameters
    ----------
    spend
        Spend shaped ``(n_periods, n_channels)``, oldest first.
    decay
        Retention rate per channel, shaped ``(n_channels,)``, each in ``[0, 1)``.
    normalize
        Multiply by ``1 - decay`` so the transform preserves total spend. Strongly
        recommended inside a model: without it, decay rescales the effect size and becomes
        confounded with the channel coefficient.

    Returns
    -------
    Array
        Adstocked spend, same shape as ``spend``.
    """

    def step(carry: Array, row: Array) -> tuple[Array, Array]:
        carried = row + decay * carry
        return carried, carried

    init = jnp.zeros(spend.shape[1], dtype=spend.dtype)
    _, out = jax.lax.scan(step, init, spend)
    return out * (1.0 - decay) if normalize else out


def weibull_weights(max_lag: int, shape: Array, scale: Array) -> Array:
    """Build normalised Weibull carryover weights.

    Parameters
    ----------
    max_lag
        Kernel length in periods. Must be a Python ``int``: it sets array shapes and so
        cannot be traced.
    shape, scale
        Weibull parameters per channel, shaped ``(n_channels,)``.

    Returns
    -------
    Array
        Weights shaped ``(max_lag, n_channels)``, summing to 1 down the lag axis.
    """
    lags = jnp.arange(max_lag, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    raw = jnp.exp(-((lags[:, None] / scale[None, :]) ** shape[None, :]))
    return raw / raw.sum(axis=0, keepdims=True)


def weibull_adstock(spend: Array, shape: Array, scale: Array, *, max_lag: int = 21) -> Array:
    """Apply Weibull-kernel carryover along the time axis.

    Parameters
    ----------
    spend
        Spend shaped ``(n_periods, n_channels)``, oldest first.
    shape, scale
        Weibull parameters per channel.
    max_lag
        Kernel length in periods; static, since it determines array shapes.

    Returns
    -------
    Array
        Adstocked spend, same shape as ``spend``.

    Notes
    -----
    Implemented as an explicit sum over lags rather than a convolution primitive. ``max_lag``
    is small and static, so the loop unrolls at trace time, and the indexing stays legible.
    """
    n_periods = spend.shape[0]
    weights = weibull_weights(max_lag, shape, scale)
    padded = jnp.pad(spend, ((max_lag - 1, 0), (0, 0)))

    out = jnp.zeros_like(spend)
    for lag in range(max_lag):
        start = max_lag - 1 - lag
        out = out + weights[lag][None, :] * jax.lax.dynamic_slice(
            padded, (start, 0), (n_periods, spend.shape[1])
        )
    return out


def hill_saturation(spend: Array, half_saturation: Array, slope: Array) -> Array:
    """Apply a Hill saturation curve, mapping spend to a response in ``[0, 1)``.

    Parameters
    ----------
    spend
        Non-negative spend shaped ``(n_periods, n_channels)``.
    half_saturation, slope
        Hill parameters per channel, shaped ``(n_channels,)``. Both must be positive.

    Returns
    -------
    Array
        Saturated response, same shape as ``spend``.

    Notes
    -----
    A small epsilon guards ``0 ** slope``, whose gradient is undefined at zero and which
    otherwise produces NaNs that surface much later as silent sampler failures.
    """
    eps = 1e-12
    powered = jnp.power(jnp.maximum(spend, eps), slope[None, :])
    half = jnp.power(half_saturation[None, :], slope[None, :])
    return powered / (powered + half)
