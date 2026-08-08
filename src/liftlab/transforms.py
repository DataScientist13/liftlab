"""Media transforms: adstock (carryover) and saturation (diminishing returns).

These are the two nonlinearities that make marketing mix modeling more than a linear
regression on spend. Both are implemented in pure NumPy so they can be unit-tested and
reasoned about independently of any probabilistic-programming backend.

Conventions
-----------
Every function takes a 1-D spend series ordered oldest-to-newest, or a 2-D array shaped
``(n_periods, n_channels)``, and returns an array of the same shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "geometric_adstock",
    "hill_saturation",
    "weibull_adstock",
    "weibull_weights",
]

FloatArray = NDArray[np.float64]


def _as_2d(x: FloatArray) -> tuple[FloatArray, bool]:
    """Promote a 1-D series to a single-column 2-D array.

    Returns
    -------
    tuple
        The 2-D array and a flag recording whether promotion happened, so the caller can
        restore the original shape.
    """
    if x.ndim == 1:
        return x.reshape(-1, 1), True
    if x.ndim == 2:
        return x, False
    msg = f"expected a 1-D or 2-D array, got ndim={x.ndim}"
    raise ValueError(msg)


def geometric_adstock(
    spend: FloatArray,
    decay: float | FloatArray,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Apply geometric (exponential) carryover to a spend series.

    The effect of spend decays by a constant fraction each period::

        y[t] = x[t] + decay * y[t - 1]

    Parameters
    ----------
    spend
        Spend series shaped ``(n_periods,)`` or ``(n_periods, n_channels)``, oldest first.
    decay
        Retention rate in ``[0, 1)``. A scalar applies to every channel; an array must have
        one entry per channel. ``decay=0`` reduces to no carryover.
    normalize
        If ``True``, divide by ``1 / (1 - decay)`` so the transform preserves total spend.
        This decouples the decay parameter from the overall effect size, which materially
        improves identifiability when decay and the channel coefficient are both free.

    Returns
    -------
    FloatArray
        Adstocked series with the same shape as ``spend``.

    Notes
    -----
    The recursion is applied in a Python loop over periods rather than vectorised via
    ``scipy.signal.lfilter``. Marketing series are short (weeks or days over a few years),
    so the loop is not a bottleneck and the intent stays readable.
    """
    x, was_1d = _as_2d(np.asarray(spend, dtype=np.float64))
    rate = np.broadcast_to(np.asarray(decay, dtype=np.float64), (x.shape[1],)).copy()

    if np.any(rate < 0.0) or np.any(rate >= 1.0):
        msg = "decay must lie in [0, 1)"
        raise ValueError(msg)

    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, x.shape[0]):
        out[t] = x[t] + rate * out[t - 1]

    if normalize:
        out = out * (1.0 - rate)

    return out.ravel() if was_1d else out


def weibull_weights(max_lag: int, shape: float, scale: float) -> FloatArray:
    """Build normalised Weibull carryover weights over ``0..max_lag - 1``.

    Unlike geometric decay, a Weibull kernel can peak *after* lag zero, which is the
    realistic shape for channels with a delayed response (TV, out-of-home, brand search
    spillover).

    Parameters
    ----------
    max_lag
        Number of lags in the kernel, including lag zero. Must be at least 1.
    shape
        Weibull shape ``k``. ``k = 1`` gives exponential decay; ``k > 1`` gives a delayed
        peak; ``k < 1`` gives a sharp initial spike with a long tail.
    scale
        Weibull scale ``lambda``, in periods. Controls how far out the mass sits.

    Returns
    -------
    FloatArray
        Weights of length ``max_lag`` summing to 1.
    """
    if max_lag < 1:
        msg = "max_lag must be at least 1"
        raise ValueError(msg)
    if shape <= 0.0 or scale <= 0.0:
        msg = "shape and scale must be positive"
        raise ValueError(msg)

    lags = np.arange(max_lag, dtype=np.float64)
    # Survival-function form: mass remaining at each lag, which is the standard
    # parameterisation used for adstock in the MMM literature.
    weights = np.exp(-((lags / scale) ** shape))
    total = weights.sum()
    return np.asarray(weights / total, dtype=np.float64)


def weibull_adstock(
    spend: FloatArray,
    shape: float,
    scale: float,
    *,
    max_lag: int = 12,
) -> FloatArray:
    """Apply Weibull-kernel carryover to a spend series.

    Parameters
    ----------
    spend
        Spend series shaped ``(n_periods,)`` or ``(n_periods, n_channels)``, oldest first.
    shape, scale
        Weibull kernel parameters; see :func:`weibull_weights`.
    max_lag
        Kernel length in periods.

    Returns
    -------
    FloatArray
        Adstocked series with the same shape as ``spend``. Weights are normalised, so total
        spend is preserved up to truncation at the start of the series.
    """
    x, was_1d = _as_2d(np.asarray(spend, dtype=np.float64))
    weights = weibull_weights(max_lag, shape, scale)

    out = np.zeros_like(x)
    n_periods = x.shape[0]
    for lag, w in enumerate(weights):
        if lag >= n_periods:
            break
        out[lag:] += w * x[: n_periods - lag]

    return out.ravel() if was_1d else out


def hill_saturation(
    spend: FloatArray,
    half_saturation: float | FloatArray,
    slope: float | FloatArray,
) -> FloatArray:
    """Apply a Hill saturation curve, mapping spend to a response in ``[0, 1)``.

    ::

        y = x**slope / (x**slope + half_saturation**slope)

    Parameters
    ----------
    spend
        Non-negative spend, any shape.
    half_saturation
        Spend level at which the response reaches half its asymptote. Reported on the same
        scale as ``spend``, which is what makes this parameter interpretable to a marketer
        and therefore prior-elicitable.
    slope
        Hill coefficient controlling steepness. Must be positive; ``slope > 1`` produces an
        S-curve with a genuine threshold, ``slope <= 1`` a concave curve.

    Returns
    -------
    FloatArray
        Saturated response, same shape as ``spend``, in ``[0, 1)``.
    """
    x = np.asarray(spend, dtype=np.float64)
    k = np.asarray(half_saturation, dtype=np.float64)
    s = np.asarray(slope, dtype=np.float64)

    if np.any(x < 0.0):
        msg = "spend must be non-negative"
        raise ValueError(msg)
    if np.any(k <= 0.0):
        msg = "half_saturation must be positive"
        raise ValueError(msg)
    if np.any(s <= 0.0):
        msg = "slope must be positive"
        raise ValueError(msg)

    x_s = np.power(x, s)
    return np.asarray(x_s / (x_s + np.power(k, s)), dtype=np.float64)
