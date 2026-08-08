"""Cross-implementation agreement between the NumPy and JAX transforms.

Two implementations of the same maths is a standing risk: they drift, and the model silently
stops matching the reference the tests reason about. These tests exist to make that drift
loud. The NumPy version is the reference; the JAX version must match it.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from liftlab import jax_transforms as jt
from liftlab import transforms as npt

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


@pytest.fixture
def spend():
    rng = np.random.default_rng(42)
    base = rng.gamma(shape=2.0, scale=1_000.0, size=(180, 3))
    # Include exact zeros: dark periods are common in real media plans and are where
    # naive power-law implementations produce NaNs.
    base[20:30, 1] = 0.0
    return base


class TestGeometricAdstock:
    @pytest.mark.parametrize("normalize", [False, True])
    def test_matches_numpy(self, spend, normalize):
        decay = np.array([0.0, 0.45, 0.85])
        expected = npt.geometric_adstock(spend, decay, normalize=normalize)
        actual = jt.geometric_adstock(spend, decay, normalize=normalize)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=TOL, atol=TOL)

    def test_is_differentiable(self, spend):
        def total(decay):
            return jt.geometric_adstock(spend, decay).sum()

        grad = jax.grad(total)(np.array([0.2, 0.5, 0.7]))
        assert np.all(np.isfinite(np.asarray(grad)))


class TestWeibullAdstock:
    def test_weights_match_numpy(self):
        shape = np.array([1.5, 2.2])
        scale = np.array([3.0, 6.0])
        actual = np.asarray(jt.weibull_weights(21, shape, scale))
        for channel in range(2):
            expected = npt.weibull_weights(21, float(shape[channel]), float(scale[channel]))
            np.testing.assert_allclose(actual[:, channel], expected, rtol=TOL, atol=TOL)

    def test_matches_numpy(self, spend):
        shape = np.array([1.5, 2.2, 0.8])
        scale = np.array([3.0, 6.0, 2.0])
        actual = np.asarray(jt.weibull_adstock(spend, shape, scale, max_lag=21))
        for channel in range(spend.shape[1]):
            expected = npt.weibull_adstock(
                spend[:, channel],
                float(shape[channel]),
                float(scale[channel]),
                max_lag=21,
            )
            np.testing.assert_allclose(actual[:, channel], expected, rtol=TOL, atol=TOL)

    def test_is_differentiable(self, spend):
        def total(scale):
            return jt.weibull_adstock(spend, np.array([2.0, 2.0, 2.0]), scale, max_lag=10).sum()

        grad = jax.grad(total)(np.array([3.0, 5.0, 7.0]))
        assert np.all(np.isfinite(np.asarray(grad)))


class TestHillSaturation:
    def test_matches_numpy(self, spend):
        half = np.array([500.0, 2_000.0, 5_000.0])
        slope = np.array([1.0, 1.5, 2.2])
        actual = np.asarray(jt.hill_saturation(spend, half, slope))
        for channel in range(spend.shape[1]):
            expected = npt.hill_saturation(
                spend[:, channel], float(half[channel]), float(slope[channel])
            )
            np.testing.assert_allclose(actual[:, channel], expected, rtol=1e-8, atol=1e-8)

    def test_zero_spend_is_finite_and_differentiable(self):
        """Zero spend must not produce NaN gradients.

        ``0 ** slope`` has an undefined derivative, and a NaN here does not fail loudly —
        it surfaces much later as a sampler that will not initialise.
        """
        zeros = np.zeros((10, 2))
        half = np.array([100.0, 200.0])

        def total(slope):
            return jt.hill_saturation(zeros, half, slope).sum()

        value = jt.hill_saturation(zeros, half, np.array([1.5, 2.0]))
        assert np.all(np.isfinite(np.asarray(value)))
        assert np.all(np.asarray(value) < 1e-6)
        assert np.all(np.isfinite(np.asarray(jax.grad(total)(np.array([1.5, 2.0])))))
