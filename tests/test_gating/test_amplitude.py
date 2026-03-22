"""Tests for dirprobe.gating.amplitude."""

import numpy as np
import pytest

from dirprobe.gating.amplitude import apply_amplitude_gate, compute_off_centring
from dirprobe.moments2.tensor import compute_site_covariance, is_valid_directions
from dirprobe.moments2.coherence import (
    compute_delta_coh,
    compute_s_align,
)
from dirprobe.time.persistence import compute_persistence_metrics


# ── apply_amplitude_gate ─────────────────────────────────────────────

class TestAmplitudeGate:
    def test_known_fraction(self):
        vecs = np.array([
            [0.05, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.35, 0.0, 0.0],
        ])
        result = apply_amplitude_gate(vecs, threshold=0.20)
        assert result.shape[0] == 2

    def test_survivors_unit_normalised(self):
        rng = np.random.default_rng(7)
        vecs = rng.normal(0, 1, (100, 3))
        result = apply_amplitude_gate(vecs, threshold=0.5)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_all_excluded_shape(self):
        vecs = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
        result = apply_amplitude_gate(vecs, threshold=1.0)
        assert result.shape == (0, 3)

    def test_all_excluded_ndim_2(self):
        vecs = np.random.default_rng(0).standard_normal((20, 3))
        result = apply_amplitude_gate(vecs, threshold=1e6)
        assert result.ndim == 2

    def test_zero_threshold(self):
        rng = np.random.default_rng(8)
        vecs = rng.normal(0, 1, (50, 3))
        result = apply_amplitude_gate(vecs, threshold=0.0)
        assert result.shape[0] == 50

    def test_return_mask(self):
        vecs = np.array([
            [0.05, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ])
        unit, mask = apply_amplitude_gate(vecs, threshold=0.20, return_mask=True)
        assert mask.dtype == bool
        assert mask.tolist() == [False, True]
        assert unit.shape == (1, 3)

    def test_gate_on_raw_not_normalised(self):
        """Gate must use raw amplitudes, not post-normalisation norms."""
        vecs = np.array([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = apply_amplitude_gate(vecs, threshold=1.0)
        assert result.shape[0] == 1
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-12)


# ── compute_off_centring ─────────────────────────────────────────────

class TestOffCentring:
    def test_basic(self):
        site = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        cage = np.array([
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        disp = compute_off_centring(site, cage)
        assert disp.shape == (2, 3)
        np.testing.assert_allclose(disp[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(disp[1], [2.0, 0.0, 0.0])


# ── End-to-end all-excluded propagation ──────────────────────────────

class TestAllExcludedPropagation:
    def test_full_pipeline_no_crash(self):
        """8 sites, all frames below gate → graceful NaN propagation."""
        rng = np.random.default_rng(99)
        n_sites = 8
        # Tiny vectors: all norms < 0.01
        raw_sites = [rng.normal(0, 0.001, (100, 3)) for _ in range(n_sites)]
        threshold = 0.20

        # Gate all sites
        gated = [apply_amplitude_gate(v, threshold) for v in raw_sites]
        # All should be empty
        for g in gated:
            assert g.shape[0] == 0

        # compute_delta_coh with empty arrays
        result = compute_delta_coh(gated)
        assert all(np.isnan(d) for d in result["per_site_d_dir"])
        assert np.isnan(result["d_dir_pooled"])
        assert np.isnan(result["pooled_C"]).all()

        # compute_s_align with NaN covariances
        nan_covs = [np.full((3, 3), np.nan)] * n_sites
        val, info = compute_s_align(nan_covs)
        assert np.isnan(val)
        assert info["n_valid_sites"] == 0

        # compute_persistence_metrics with empty arrays
        pm = compute_persistence_metrics(gated, window_size=50)
        assert pm["per_site_class"] == ["INSUFFICIENT"] * n_sites
