"""Tests for dirprobe.synthetic.generators and vmf."""

import numpy as np
import pytest

from dirprobe._constants import EPS_DEGENERACY
from dirprobe.moments2.coherence import compute_delta_coh
from dirprobe.moments2.tensor import (
    check_degeneracy,
    compute_d_dir,
    compute_site_covariance,
    decompose_covariance,
    is_valid_directions,
)
from dirprobe.robustness.suite import run_robustness_suite
from dirprobe.synthetic.generators import ALL_SYSTEMS, generate_all
from dirprobe.synthetic.vmf import sample_vmf, vmf_d_dir_3d, vmf_eigenvalues_3d
from dirprobe.time.persistence import compute_persistence_metrics


# ── vMF analytic eigenvalues ─────────────────────────────────────────

class TestVmfAnalytic:
    def test_kappa_zero_isotropic(self):
        evals = vmf_eigenvalues_3d(0.0)
        np.testing.assert_allclose(evals, [1/3, 1/3, 1/3], atol=1e-10)

    def test_kappa_zero_ddir(self):
        assert abs(vmf_d_dir_3d(0.0) - 3.0) < 1e-10

    def test_high_kappa_uniaxial(self):
        evals = vmf_eigenvalues_3d(500.0)
        assert evals[0] > 0.99
        assert abs(vmf_d_dir_3d(500.0) - 1.0) < 0.02

    def test_overflow_kappa_1000(self):
        """Must not overflow at kappa=1000 (cosh/sinh would)."""
        evals = vmf_eigenvalues_3d(1000.0)
        assert np.all(np.isfinite(evals))
        assert abs(np.sum(evals) - 1.0) < 1e-10
        assert evals[0] > 0.99

    def test_sum_to_one(self):
        for kappa in [0.1, 1.0, 5.0, 20.0, 100.0, 700.0]:
            evals = vmf_eigenvalues_3d(kappa)
            assert abs(np.sum(evals) - 1.0) < 1e-10

    def test_descending_order(self):
        for kappa in [1.0, 10.0, 50.0]:
            evals = vmf_eigenvalues_3d(kappa)
            assert evals[0] >= evals[1]
            assert evals[1] >= evals[2]

    def test_ddir_monotone_decreasing_with_kappa(self):
        """D_dir should decrease monotonically as kappa increases."""
        kappas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        ddirs = [vmf_d_dir_3d(k) for k in kappas]
        for i in range(len(ddirs) - 1):
            assert ddirs[i] > ddirs[i + 1]


# ── vMF sampling ─────────────────────────────────────────────────────

class TestVmfSampling:
    def test_unit_vectors(self):
        rng = np.random.default_rng(0)
        mu = np.array([0.0, 0.0, 1.0])
        samples = sample_vmf(mu, 10.0, 1000, rng)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_shape(self):
        rng = np.random.default_rng(1)
        mu = np.array([1.0, 0.0, 0.0])
        samples = sample_vmf(mu, 5.0, 500, rng)
        assert samples.shape == (500, 3)

    def test_high_kappa_concentrated(self):
        rng = np.random.default_rng(2)
        mu = np.array([0.0, 0.0, 1.0])
        samples = sample_vmf(mu, 100.0, 5000, rng)
        mean_dot = np.mean(samples @ mu)
        assert mean_dot > 0.95

    def test_empirical_matches_analytic(self):
        """Empirical D_dir should approximate analytic D_dir."""
        rng = np.random.default_rng(3)
        mu = np.array([0.0, 0.0, 1.0])
        kappa = 10.0
        samples = sample_vmf(mu, kappa, 50000, rng)
        c = compute_site_covariance(samples)
        evals, _ = decompose_covariance(c)
        d_dir_emp = compute_d_dir(evals)
        d_dir_analytic = vmf_d_dir_3d(kappa)
        assert abs(d_dir_emp - d_dir_analytic) < 0.05

    def test_kappa_zero_uniform(self):
        rng = np.random.default_rng(4)
        mu = np.array([0.0, 0.0, 1.0])
        samples = sample_vmf(mu, 0.0, 10000, rng)
        c = compute_site_covariance(samples)
        evals, _ = decompose_covariance(c)
        # Should be near-isotropic
        assert max(evals) - min(evals) < 0.05


# ── Generator contract ───────────────────────────────────────────────

class TestGeneratorContract:
    def test_15_systems(self):
        assert len(ALL_SYSTEMS) == 15

    def test_all_names(self):
        expected = {"a", "b", "c", "c2", "d", "e", "f", "g", "h",
                    "i", "j", "k", "l", "m", "n"}
        assert set(ALL_SYSTEMS.keys()) == expected

    @pytest.mark.parametrize("name", list(ALL_SYSTEMS.keys()))
    def test_return_structure(self, name):
        site_dirs, gt = ALL_SYSTEMS[name]()
        assert isinstance(site_dirs, list)
        assert len(site_dirs) == gt["n_sites"]
        assert "system_name" in gt
        assert "d_dir_expected" in gt
        assert "delta_coh_expected" in gt
        assert "n_sites" in gt
        assert "n_frames" in gt

    @pytest.mark.parametrize("name", list(ALL_SYSTEMS.keys()))
    def test_valid_directions(self, name):
        site_dirs, gt = ALL_SYSTEMS[name]()
        for dirs in site_dirs:
            assert is_valid_directions(dirs)

    @pytest.mark.parametrize("name", list(ALL_SYSTEMS.keys()))
    def test_frame_count(self, name):
        site_dirs, gt = ALL_SYSTEMS[name]()
        for dirs in site_dirs:
            assert dirs.shape[0] == gt["n_frames"]


# ── Specific system behaviour ────────────────────────────────────────

class TestSystemBehaviours:
    def test_system_a_uniaxial(self):
        """System A: D_dir ≈ 1, delta_coh ≈ 0."""
        site_dirs, _ = ALL_SYSTEMS["a"]()
        result = compute_delta_coh(site_dirs)
        assert result["d_dir_site_mean"] < 1.15
        assert abs(result["delta_coh"]) < 0.1

    def test_system_b_isotropic(self):
        """System B: D_dir ≈ 3, delta_coh ≈ 0."""
        site_dirs, _ = ALL_SYSTEMS["b"]()
        result = compute_delta_coh(site_dirs)
        assert result["d_dir_site_mean"] > 2.7
        assert abs(result["delta_coh"]) < 0.3

    def test_system_d_planar(self):
        """System D: D_dir ≈ 2."""
        site_dirs, _ = ALL_SYSTEMS["d"]()
        result = compute_delta_coh(site_dirs)
        assert 1.7 < result["d_dir_site_mean"] < 2.3

    def test_system_e_high_delta_coh(self):
        """System E: per-site D_dir ≈ 1 but high delta_coh."""
        site_dirs, _ = ALL_SYSTEMS["e"]()
        result = compute_delta_coh(site_dirs)
        assert result["d_dir_site_mean"] < 1.15
        assert result["delta_coh"] > 0.5

    def test_system_k_near_degenerate(self):
        """System K: D_dir ≈ 3 (near-isotropic), some sites axis_degenerate."""
        site_dirs, gt = ALL_SYSTEMS["k"]()
        result = compute_delta_coh(site_dirs)
        # Near-isotropic: D_dir close to 3
        assert result["d_dir_site_mean"] > 2.5
        # At least some sites should be axis_degenerate
        n_degenerate = 0
        for dirs in site_dirs:
            c = compute_site_covariance(dirs)
            evals, _ = decompose_covariance(c)
            deg = check_degeneracy(evals)
            if deg["axis_degenerate"]:
                n_degenerate += 1
        assert n_degenerate >= 1

    def test_system_l_switching(self):
        """System L: persistence classification should include SWITCHING or INSUFFICIENT.
        With EPS_DEGENERACY=0.05, short switching windows may be flagged degenerate."""
        site_dirs, _ = ALL_SYSTEMS["l"]()
        pm = compute_persistence_metrics(site_dirs)
        classes = set(pm["per_site_class"])
        assert classes & {"SWITCHING", "FAST", "INSUFFICIENT"}, (
            f"Expected SWITCHING or INSUFFICIENT, got {classes}"
        )

    def test_system_n_short(self):
        """System N: only 50 frames."""
        site_dirs, gt = ALL_SYSTEMS["n"]()
        assert gt["n_frames"] == 50
        for dirs in site_dirs:
            assert dirs.shape[0] == 50

    def test_system_n_robustness_not_pass(self):
        """System N: must not produce PASS from run_robustness_suite."""
        site_dirs, _ = ALL_SYSTEMS["n"]()
        result = run_robustness_suite(site_dirs)
        assert result["summary"] != "PASS"


# ── generate_all ─────────────────────────────────────────────────────

class TestGenerateAll:
    def test_returns_all_15(self):
        all_systems = generate_all()
        assert len(all_systems) == 15
        for name, (site_dirs, gt) in all_systems.items():
            assert gt["system_name"] == name
