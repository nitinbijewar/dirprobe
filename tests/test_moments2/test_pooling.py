"""Tests for dirprobe.moments2.pooling."""

import numpy as np
import pytest

from dirprobe.moments2.pooling import concat_directions, pool_covariances
from dirprobe.moments2.tensor import compute_site_covariance


# ── helpers ──────────────────────────────────────────────────────────

def _make_site_covs(n_sites=4, n_frames=200, seed=42):
    """Generate site covariances from random directions."""
    rng = np.random.default_rng(seed)
    covs = []
    dirs_list = []
    for i in range(n_sites):
        dirs = rng.normal(0, 1, (n_frames, 3))
        dirs[:, i % 3] *= 3.0  # bias one axis
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        covs.append(compute_site_covariance(dirs))
        dirs_list.append(dirs)
    return covs, dirs_list


# ── pool_covariances ─────────────────────────────────────────────────

class TestPoolCovariances:
    def test_trace_normalised(self):
        covs, _ = _make_site_covs()
        result = pool_covariances(covs)
        assert abs(np.trace(result) - 1.0) < 1e-6

    def test_equal_weights(self):
        covs, _ = _make_site_covs(n_sites=3)
        r1 = pool_covariances(covs)
        r2 = pool_covariances(covs, weights=np.ones(3))
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    def test_explicit_weights(self):
        covs, _ = _make_site_covs(n_sites=2)
        w = np.array([1.0, 0.0])
        result = pool_covariances(covs, weights=w)
        # Should be trace-normalised version of covs[0]
        expected = covs[0] / np.trace(covs[0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_nan_sites_excluded(self):
        covs, _ = _make_site_covs(n_sites=3)
        covs.append(np.full((3, 3), np.nan))
        result = pool_covariances(covs)
        assert np.isfinite(result).all()
        assert abs(np.trace(result) - 1.0) < 1e-6

    def test_nan_weight_renormalisation(self):
        covs, _ = _make_site_covs(n_sites=2)
        covs_with_nan = covs + [np.full((3, 3), np.nan)]
        r1 = pool_covariances(covs)
        r2 = pool_covariances(covs_with_nan)
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    def test_all_invalid_returns_nan(self):
        covs = [np.full((3, 3), np.nan), np.full((3, 3), np.nan)]
        result = pool_covariances(covs)
        assert np.isnan(result).all()

    def test_empty_list(self):
        result = pool_covariances([])
        assert np.isnan(result).all()


# ── concat_directions ────────────────────────────────────────────────

class TestConcatDirections:
    def test_output_length(self):
        dirs = [np.ones((100, 3)), np.ones((50, 3))]
        result = concat_directions(dirs)
        assert result.shape == (150, 3)

    def test_ndim_always_2(self):
        dirs = [np.ones((1, 3))]
        result = concat_directions(dirs)
        assert result.ndim == 2

    def test_empty_skipped(self):
        dirs = [np.empty((0, 3)), np.ones((10, 3))]
        result = concat_directions(dirs)
        assert result.shape == (10, 3)
