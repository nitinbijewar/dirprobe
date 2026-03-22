"""End-to-end pipeline tests on synthetic data."""

import numpy as np
import pytest

from dirprobe.pipeline import run_pipeline
from dirprobe.synthetic.vmf import sample_vmf


class TestPipelineBasic:
    def test_isotropic_ddir_near_3(self):
        rng = np.random.default_rng(42)
        mu = np.array([0.0, 0.0, 1.0])
        disps = np.stack([sample_vmf(mu, 0.0, 2000, rng) for _ in range(8)], axis=1)
        r = run_pipeline(disps, gating_threshold=0.0)
        assert abs(r["d_dir_site_mean"] - 3.0) < 0.15

    def test_uniaxial_ddir_near_1(self):
        rng = np.random.default_rng(43)
        mu = np.array([0.0, 0.0, 1.0])
        disps = np.stack([sample_vmf(mu, 50.0, 2000, rng) for _ in range(8)], axis=1)
        r = run_pipeline(disps, gating_threshold=0.0)
        assert abs(r["d_dir_site_mean"] - 1.08) < 0.15

    def test_incoherent_large_dcoh(self):
        rng = np.random.default_rng(44)
        cube = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
                         [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]], dtype=float)
        cube /= np.linalg.norm(cube, axis=1, keepdims=True)
        disps = np.stack([sample_vmf(cube[i], 50.0, 2000, rng) for i in range(8)], axis=1)
        r = run_pipeline(disps, gating_threshold=0.0)
        assert r["delta_coh"] > 1.5

    def test_return_keys(self):
        rng = np.random.default_rng(45)
        disps = np.stack([sample_vmf(np.array([0,0,1.0]), 50.0, 200, rng) for _ in range(4)], axis=1)
        r = run_pipeline(disps, gating_threshold=0.0, persistence_windows=2)
        expected_keys = {"d_dir_site", "d_dir_site_mean", "d_dir_pooled",
                         "delta_coh", "s_align", "a_mean", "b_mean",
                         "classification", "robustness", "gated_counts",
                         "site_directions"}
        assert set(r.keys()) == expected_keys

    def test_gating_reduces_frames(self):
        rng = np.random.default_rng(46)
        # Create displacements with varying amplitudes
        dirs = np.stack([sample_vmf(np.array([0,0,1.0]), 50.0, 200, rng) for _ in range(4)], axis=1)
        amps = rng.uniform(0, 0.3, (200, 4, 1))
        disps = dirs * amps
        r = run_pipeline(disps, gating_threshold=0.20)
        assert all(c < 200 for c in r["gated_counts"])
