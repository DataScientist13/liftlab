"""Unit tests for the media transforms.

These check properties that must hold for the transforms to be usable inside an MMM:
shape preservation, mass preservation under normalisation, interpretability of the
half-saturation parameter, and loud failure on invalid parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from liftlab import (
    geometric_adstock,
    hill_saturation,
    weibull_adstock,
    weibull_weights,
)


class TestGeometricAdstock:
    def test_zero_decay_is_identity(self):
        spend = np.array([1.0, 4.0, 0.0, 2.5])
        np.testing.assert_allclose(geometric_adstock(spend, 0.0), spend)

    def test_known_recursion(self):
        spend = np.array([1.0, 0.0, 0.0])
        # 1, 0.5, 0.25
        np.testing.assert_allclose(geometric_adstock(spend, 0.5), [1.0, 0.5, 0.25])

    def test_normalize_preserves_total_spend(self):
        rng = np.random.default_rng(11)
        decay, n = 0.6, 2000
        spend = rng.gamma(shape=2.0, scale=100.0, size=n)
        out = geometric_adstock(spend, decay, normalize=True)

        # Preservation is exact only for a series with an infinite past. Assuming zero spend
        # before t=0 loses carryover into the first few periods; that head deficit is
        # O(decay / ((1 - decay) * n)) relative to the total, so the tolerance has to scale
        # with series length rather than being a fixed constant.
        head_deficit = decay / ((1.0 - decay) * n)
        assert out.sum() == pytest.approx(spend.sum(), rel=2 * head_deficit)
        assert out.sum() < spend.sum()  # the deficit is one-sided

    def test_preserves_shape_for_multichannel(self):
        spend = np.ones((10, 3))
        out = geometric_adstock(spend, np.array([0.0, 0.3, 0.9]))
        assert out.shape == (10, 3)
        # A higher decay must accumulate more by the final period.
        assert out[-1, 0] < out[-1, 1] < out[-1, 2]

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_rejects_out_of_range_decay(self, bad):
        with pytest.raises(ValueError, match="decay"):
            geometric_adstock(np.ones(5), bad)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="ndim"):
            geometric_adstock(np.ones((2, 2, 2)), 0.5)


class TestWeibullWeights:
    def test_weights_sum_to_one(self):
        w = weibull_weights(12, shape=2.0, scale=4.0)
        assert w.shape == (12,)
        assert w.sum() == pytest.approx(1.0)

    def test_larger_scale_pushes_mass_later(self):
        short = weibull_weights(20, shape=2.0, scale=2.0)
        long = weibull_weights(20, shape=2.0, scale=8.0)
        lags = np.arange(20)
        assert (short * lags).sum() < (long * lags).sum()

    @pytest.mark.parametrize(
        ("max_lag", "shape", "scale", "match"),
        [
            (0, 2.0, 4.0, "max_lag"),
            (12, 0.0, 4.0, "positive"),
            (12, 2.0, -1.0, "positive"),
        ],
    )
    def test_rejects_invalid_parameters(self, max_lag, shape, scale, match):
        with pytest.raises(ValueError, match=match):
            weibull_weights(max_lag, shape, scale)


class TestWeibullAdstock:
    def test_preserves_total_when_tail_fits_in_series(self):
        spend = np.zeros(50)
        spend[0] = 100.0
        out = weibull_adstock(spend, shape=2.0, scale=3.0, max_lag=10)
        assert out.sum() == pytest.approx(100.0)

    def test_spreads_a_single_pulse_across_lags(self):
        spend = np.zeros(20)
        spend[0] = 1.0
        out = weibull_adstock(spend, shape=2.0, scale=4.0, max_lag=8)
        assert out[0] < 1.0
        assert np.count_nonzero(out) == 8

    def test_kernel_longer_than_series_is_truncated_not_an_error(self):
        out = weibull_adstock(np.ones(3), shape=1.5, scale=2.0, max_lag=50)
        assert out.shape == (3,)

    def test_preserves_shape_for_multichannel(self):
        out = weibull_adstock(np.ones((30, 2)), shape=2.0, scale=3.0, max_lag=6)
        assert out.shape == (30, 2)


class TestHillSaturation:
    def test_half_saturation_returns_one_half(self):
        assert hill_saturation(np.array([50.0]), 50.0, 1.0)[0] == pytest.approx(0.5)

    def test_is_monotonically_increasing(self):
        x = np.linspace(0.0, 500.0, 100)
        y = hill_saturation(x, 100.0, 2.0)
        assert np.all(np.diff(y) > 0.0)

    def test_bounded_in_unit_interval(self):
        y = hill_saturation(np.array([0.0, 1.0, 1e9]), 100.0, 1.5)
        assert y[0] == 0.0
        assert np.all(y < 1.0)
        assert y[-1] > 0.999

    @pytest.mark.parametrize(
        ("spend", "k", "s", "match"),
        [
            (np.array([-1.0]), 10.0, 1.0, "non-negative"),
            (np.array([1.0]), 0.0, 1.0, "half_saturation"),
            (np.array([1.0]), 10.0, 0.0, "slope"),
        ],
    )
    def test_rejects_invalid_parameters(self, spend, k, s, match):
        with pytest.raises(ValueError, match=match):
            hill_saturation(spend, k, s)
