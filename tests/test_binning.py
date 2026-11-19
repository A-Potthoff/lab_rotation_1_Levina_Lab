"""Tests for src.utils.binning.

Run:  python -m pytest tests/test_binning.py -v
"""

import numpy as np
import pytest

from src.utils.binning import binning


def binning_reference(x, fuse):
    """Brute force: overlap of each unit step [t, t+1) with each bin [k*fuse, (k+1)*fuse)."""
    T = len(x)
    n_bins = int(np.floor(T / fuse + 1e-9))
    sums = np.zeros(n_bins)
    for k in range(n_bins):
        start, end = k * fuse, (k + 1) * fuse
        for t in range(T):
            overlap = max(0.0, min(end, t + 1) - max(start, t))
            sums[k] += overlap * x[t]
    return sums


@pytest.mark.parametrize("fuse", [1, 2, 3, 7])
def test_integer_fuse_is_reshape_sum(fuse):
    x = np.random.default_rng(seed=0).poisson(lam=5, size=50)
    n_bins = 50 // fuse
    expected = x[: n_bins * fuse].reshape(n_bins, fuse).sum(axis=1)
    np.testing.assert_array_equal(binning(x, fuse=fuse), expected)
    np.testing.assert_allclose(binning(x, fuse=fuse, mode="mean"), expected / fuse)


def test_fuse_one_is_identity_as_float():
    x = np.arange(10)
    result = binning(x, fuse=1)
    np.testing.assert_array_equal(result, x)
    assert result.dtype == float


def test_fractional_fuse_hand_example():
    x = np.array([1, 2, 4, 8, 16, 32])
    # bins [0, 2.5), [2.5, 5); last 1 step dropped
    np.testing.assert_allclose(binning(x, fuse=2.5), [1 + 2 + 0.5 * 4, 0.5 * 4 + 8 + 16])


def test_fuse_below_one_splits_steps():
    x = np.array([2.0, 6.0])
    np.testing.assert_allclose(binning(x, fuse=0.5), [1, 1, 3, 3])
    np.testing.assert_allclose(binning(x, fuse=0.5, mode="mean"), [2, 2, 6, 6])


@pytest.mark.parametrize("fuse", [0.3, 0.7, 1.5, 2.5, 4.01, 10.1, 20.5, 33.3])
def test_matches_brute_force_reference(fuse):
    x = np.random.default_rng(seed=1).poisson(lam=3, size=101).astype(float)
    np.testing.assert_allclose(binning(x, fuse=fuse), binning_reference(x, fuse), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("fuse", [1, 2.5, 4.01, 10])
def test_sum_conserved_over_covered_range(fuse):
    x = np.random.default_rng(seed=2).random(200)
    n_bins = int(np.floor(200 / fuse))
    covered = n_bins * fuse  # in time steps
    full_steps = int(np.floor(covered))
    expected_total = x[:full_steps].sum() + (covered - full_steps) * (x[full_steps] if full_steps < 200 else 0)
    np.testing.assert_allclose(binning(x, fuse=fuse).sum(), expected_total)


@pytest.mark.parametrize("fuse", [1, 2.5, 4.01, 0.4])
def test_mean_of_constant_is_constant(fuse):
    np.testing.assert_allclose(binning(np.full(100, 3.0), fuse=fuse, mode="mean"), 3.0)


def test_2d_matches_rowwise():
    x = np.random.default_rng(seed=3).poisson(lam=4, size=(5, 60))
    result = binning(x, fuse=4.01, mode="mean")
    assert result.shape == (5, 14)
    for row, row_result in zip(x, result):
        np.testing.assert_allclose(row_result, binning(row, fuse=4.01, mode="mean"))


def test_bin_count_float_precision():
    # 3 / 0.3 = 9.999... in floating point; still 10 bins
    assert binning(np.ones(3), fuse=0.3).shape == (10,)
    assert binning(np.ones(10), fuse=0.1).shape == (100,)


def test_trailing_partial_bin_dropped():
    assert binning(np.ones(10), fuse=3).shape == (3,)
    assert binning(np.ones(10), fuse=2.5).shape == (4,)


def test_fuse_longer_than_series_gives_empty():
    assert binning(np.ones(3), fuse=5).shape == (0,)
    assert binning(np.ones((2, 3)), fuse=5).shape == (2, 0)
    assert binning(np.ones(0), fuse=1).shape == (0,)


def test_accepts_list():
    np.testing.assert_array_equal(binning([1, 2, 3, 4], fuse=2), [3, 7])


@pytest.mark.parametrize("fuse", [0, -1, np.nan])
def test_invalid_fuse(fuse):
    with pytest.raises(ValueError):
        binning(np.ones(5), fuse=fuse)


def test_invalid_mode_and_scalar():
    with pytest.raises(ValueError):
        binning(np.ones(5), mode="median")
    with pytest.raises(ValueError):
        binning(np.float64(1.0))
