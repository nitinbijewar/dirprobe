"""Tests for dirprobe.moments2.coherence."""

import numpy as np
import pytest

from dirprobe.moments2.coherence import (
    _valid_site_axes,
    compute_delta_coh,
    compute_mean_sij,
    compute_s_align,
)
from dirprobe.moments2.tensor import compute_site_covariance
from dirprobe._constants import EPS_DEGENERACY


# ── helpers ──────────────────────────────────────────────────────────

def _aligned_site_dirs(n_sites=8, n_frames=200, noise=0.01, seed=42):
    """All sites share the same axis (z), slight noise."""
    rng = np.random.default_rng(seed)
    sites = []
    for _ in range(n_sites):
        dirs = np.zeros((n_frames, 3))
        dirs[:, 2] = 1.0
        dirs += rng.normal(0, noise, dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        sites.append(dirs)
    return sites


def _orthogonal_site_dirs(n_frames=200, noise=0.01, seed=42):
    """Two sites with orthogonal preferred axes (z and x)."""
    rng = np.random.default_rng(seed)
    sites = []
    for axis_idx in [2, 0]:
        dirs = np.zeros((n_frames, 3))
        dirs[:, axis_idx] = 1.0
        dirs += rng.normal(0, noise, dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        sites.append(dirs)
    return sites


def _random_site_covs(n_sites=20, seed=42):
    """Sites with randomly oriented axes."""
    rng = np.random.default_rng(seed)
    covs = []
    for _ in range(n_sites):
        dirs = rng.normal(0, 1, (200, 3))
        # Bias one axis randomly to avoid degeneracy
        axis = rng.integers(0, 3)
        dirs[:, axis] *= 3.0
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        covs.append(compute_site_covariance(dirs))
    return covs


def _make_degenerate_cov():
    """Return a covariance matrix with |lambda_1 - lambda_2| < eps."""
    return np.diag([1 / 3 + 0.001, 1 / 3, 1 / 3 - 0.001])


def _make_nonfinite_cov():
    """Return a non-finite covariance matrix."""
    return np.full((3, 3), np.nan)


# ── delta_coh ────────────────────────────────────────────────────────

class TestDeltaCoh:
    def test_aligned_delta_near_zero(self):
        sites = _aligned_site_dirs(n_sites=8)
        result = compute_delta_coh(sites)
        assert abs(result["delta_coh"]) < 0.1

    def test_orthogonal_delta_positive(self):
        sites = _orthogonal_site_dirs()
        result = compute_delta_coh(sites)
        assert result["delta_coh"] > 0.0

    def test_return_keys(self):
        sites = _aligned_site_dirs(n_sites=3)
        result = compute_delta_coh(sites)
        expected = {
            "d_dir_site_mean", "d_dir_pooled", "delta_coh",
            "per_site_d_dir", "pooled_C", "site_Cs",
        }
        assert set(result.keys()) == expected

    def test_default_weights_equal(self):
        sites = _aligned_site_dirs(n_sites=4)
        r1 = compute_delta_coh(sites)
        r2 = compute_delta_coh(sites, weights=np.ones(4))
        assert abs(r1["delta_coh"] - r2["delta_coh"]) < 1e-10

    def test_uses_covariance_pooling(self):
        """Verify pooling is on covariances, not vector concatenation."""
        sites = _orthogonal_site_dirs(n_frames=100)
        result = compute_delta_coh(sites)
        # pooled_C should be trace-normalised
        assert abs(np.trace(result["pooled_C"]) - 1.0) < 1e-6
        # site_Cs should match per_site_d_dir
        for c, d in zip(result["site_Cs"], result["per_site_d_dir"]):
            if np.isfinite(d):
                from dirprobe.moments2.tensor import (
                    compute_d_dir,
                    decompose_covariance,
                )
                evals, _ = decompose_covariance(c)
                assert abs(compute_d_dir(evals) - d) < 1e-10

    def test_empty_site_handled(self):
        sites = [np.empty((0, 3)), _aligned_site_dirs(n_sites=1)[0]]
        result = compute_delta_coh(sites)
        assert np.isnan(result["per_site_d_dir"][0])
        assert np.isfinite(result["per_site_d_dir"][1])


# ── S_align ──────────────────────────────────────────────────────────

class TestSAlign:
    def test_all_aligned(self):
        sites = _aligned_site_dirs(n_sites=8)
        covs = [compute_site_covariance(s) for s in sites]
        val, info = compute_s_align(covs)
        assert val > 0.9

    def test_random_orientation(self):
        covs = _random_site_covs(n_sites=20)
        val, info = compute_s_align(covs)
        assert abs(val - 1 / 3) < 0.15

    def test_nan_when_few_valid(self):
        covs = [_make_degenerate_cov()]
        val, info = compute_s_align(covs)
        assert np.isnan(val)
        assert info["n_valid_sites"] < 2

    def test_info_keys(self):
        covs = _random_site_covs(n_sites=5)
        _, info = compute_s_align(covs)
        expected = {"n_axis_degenerate", "axis_degenerate_idx", "n_valid_sites"}
        assert set(info.keys()) == expected

    def test_invariant_count(self):
        """n_axis_degenerate == len(axis_degenerate_idx)."""
        covs = _random_site_covs(n_sites=5)
        _, info = compute_s_align(covs)
        assert info["n_axis_degenerate"] == len(info["axis_degenerate_idx"])

    def test_invariant_total(self):
        """n_axis_degenerate + n_valid_sites == len(site_covariances)."""
        covs = _random_site_covs(n_sites=5)
        _, info = compute_s_align(covs)
        assert (
            info["n_axis_degenerate"] + info["n_valid_sites"]
            == len(covs)
        )


# ── mean_sij ─────────────────────────────────────────────────────────

class TestMeanSij:
    def test_all_aligned(self):
        sites = _aligned_site_dirs(n_sites=8)
        covs = [compute_site_covariance(s) for s in sites]
        val, info = compute_mean_sij(covs)
        assert val > 0.9

    def test_nan_when_few_valid(self):
        covs = [_make_degenerate_cov()]
        val, info = compute_mean_sij(covs)
        assert np.isnan(val)

    def test_info_keys_match_s_align(self):
        covs = _random_site_covs(n_sites=5)
        _, info_s = compute_s_align(covs)
        _, info_m = compute_mean_sij(covs)
        assert set(info_s.keys()) == set(info_m.keys())


# ── Identical exclusion logic ────────────────────────────────────────

class TestExclusionConsistency:
    def test_same_exclusions(self):
        """s_align and mean_sij exclude the same sites."""
        covs = _random_site_covs(n_sites=5)
        covs.append(_make_degenerate_cov())
        covs.append(_make_nonfinite_cov())
        _, info_s = compute_s_align(covs)
        _, info_m = compute_mean_sij(covs)
        assert info_s["axis_degenerate_idx"] == info_m["axis_degenerate_idx"]
        assert info_s["n_valid_sites"] == info_m["n_valid_sites"]

    def test_invariant_with_degenerate(self):
        """Invariant holds with 1 degenerate site."""
        covs = _random_site_covs(n_sites=4)
        covs.append(_make_degenerate_cov())
        _, info = compute_s_align(covs)
        assert info["n_axis_degenerate"] == len(info["axis_degenerate_idx"])
        assert info["n_axis_degenerate"] + info["n_valid_sites"] == 5

    def test_invariant_with_nonfinite(self):
        """Invariant holds with 1 non-finite site."""
        covs = _random_site_covs(n_sites=4)
        covs.append(_make_nonfinite_cov())
        _, info = compute_s_align(covs)
        assert info["n_axis_degenerate"] == len(info["axis_degenerate_idx"])
        assert info["n_axis_degenerate"] + info["n_valid_sites"] == 5

    def test_invariant_mixed(self):
        """Invariant holds with mixed degenerate + non-finite."""
        covs = _random_site_covs(n_sites=3)
        covs.append(_make_degenerate_cov())
        covs.append(_make_nonfinite_cov())
        _, info = compute_s_align(covs)
        assert info["n_axis_degenerate"] == len(info["axis_degenerate_idx"])
        assert info["n_axis_degenerate"] + info["n_valid_sites"] == 5
