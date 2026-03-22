"""Tests for dirprobe.robustness.suite."""

import numpy as np
import pytest

from dirprobe.robustness import suite as _suite

# Alias to avoid pytest collecting these as test functions
run_robustness_suite = _suite.run_robustness_suite
_test_gating = _suite.test_gating_sensitivity
_test_pooling = _suite.test_pooling_sensitivity
_test_temporal = _suite.test_temporal_stability
_test_failure = _suite.test_synthetic_failure_detection


# ── helpers ──────────────────────────────────────────────────────────

def _stable_site_dirs(n_sites=8, t=800, noise=0.05, seed=42):
    """Tight uniaxial directions along z for all sites."""
    rng = np.random.default_rng(seed)
    sites = []
    for _ in range(n_sites):
        dirs = np.zeros((t, 3))
        dirs[:, 2] = 1.0
        dirs += rng.normal(0, noise, dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        sites.append(dirs)
    return sites


def _stable_by_gate(n_configs=5, n_sites=8, t=200, seed=42):
    """Stable rank ordering across 3 thresholds.

    Configs have increasing inter-site axis spread (increasing delta_coh)
    so rank ordering is preserved across thresholds.
    """
    rng = np.random.default_rng(seed)
    thresholds = [0.05, 0.10, 0.20]
    # Axis tilts per config: cfg0 all aligned, cfg4 maximally spread
    axes_per_config = []
    for ci in range(n_configs):
        spread = ci * 0.3  # radians of tilt from z
        site_axes = []
        for si in range(n_sites):
            theta = spread * (si / max(n_sites - 1, 1))
            phi = 2 * np.pi * si / n_sites
            axis = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            site_axes.append(axis)
        axes_per_config.append(site_axes)

    data: dict[float, dict[str, list[np.ndarray]]] = {}
    for th in thresholds:
        data[th] = {}
        for ci in range(n_configs):
            sites = []
            for si in range(n_sites):
                mu = axes_per_config[ci][si]
                dirs = np.tile(mu, (t, 1))
                dirs += rng.normal(0, 0.05, dirs.shape)
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                sites.append(dirs)
            data[th][f"cfg{ci}"] = sites
    return data


# ── Contract: return key structure ───────────────────────────────────

class TestReturnKeys:
    def test_gating_keys(self):
        data = _stable_by_gate()
        result = _test_gating(data)
        assert set(result.keys()) == {
            "status", "pairwise_rho", "n_insufficient_pairs",
        }

    def test_pooling_keys(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = _test_pooling(dirs, site_n_frames=np.ones(4))
        assert set(result.keys()) == {"status", "diff"}

    def test_temporal_keys(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = _test_temporal(dirs)
        assert set(result.keys()) == {"status", "mean_rel_range"}

    def test_failure_keys(self):
        result = _test_failure({})
        assert set(result.keys()) == {
            "status", "expected_failures", "actual_failures",
        }

    def test_suite_keys(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = run_robustness_suite(dirs)
        assert set(result.keys()) == {
            "summary", "flag_reasons", "test_gating", "test_pooling",
            "test_temporal", "test_failure",
        }


# ── Stable data → all PASS ──────────────────────────────────────────

class TestStablePass:
    def test_full_suite_pass(self):
        dirs = _stable_site_dirs(n_sites=8, t=800)
        by_gate = _stable_by_gate()
        n_frames = np.array([800] * 8, dtype=float)
        result = run_robustness_suite(
            dirs,
            site_directions_by_gate=by_gate,
            site_n_frames=n_frames,
        )
        # Test 4 has no known_failures → INSUFFICIENT → promotes to FLAG
        # So suite summary should be FLAG (not PASS) due to Test 4
        # But Test 1,2,3 should all be PASS
        assert result["test_gating"]["status"] == "PASS"
        assert result["test_pooling"]["status"] == "PASS"
        assert result["test_temporal"]["status"] == "PASS"

    def test_gating_stable_pass(self):
        data = _stable_by_gate()
        result = _test_gating(data)
        assert result["status"] == "PASS"


# ── None inputs → INSUFFICIENT ───────────────────────────────────────

class TestInsufficient:
    def test_pooling_none_frames(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = _test_pooling(dirs, site_n_frames=None)
        assert result["status"] == "INSUFFICIENT"
        assert np.isnan(result["diff"])

    def test_gating_none(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = run_robustness_suite(dirs, site_directions_by_gate=None)
        assert result["test_gating"]["status"] == "INSUFFICIENT"

    def test_pooling_none_in_suite(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = run_robustness_suite(dirs, site_n_frames=None)
        assert result["test_pooling"]["status"] == "INSUFFICIENT"

    def test_insufficient_promotes_to_flag(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = run_robustness_suite(dirs)
        assert result["summary"] in ("FLAG", "FAIL")
        assert len(result["flag_reasons"]) > 0


# ── Spearman (not Pearson) ───────────────────────────────────────────

class TestSpearman:
    def test_monotone_rank_preserved(self):
        """5 configs with monotone delta_coh rank → PASS."""
        data = _stable_by_gate(n_configs=5, seed=99)
        result = _test_gating(data)
        assert result["status"] == "PASS"


# ── Rank inversion → FLAG or FAIL ───────────────────────────────────

class TestGatingFlag:
    def test_rank_inversion_flags(self):
        """Configs where delta_coh rank order inverts between thresholds."""
        rng = np.random.default_rng(77)

        def _make_sites(axis_idx, n_sites=8, t=200):
            sites = []
            for _ in range(n_sites):
                dirs = np.zeros((t, 3))
                dirs[:, axis_idx] = 1.0
                dirs += rng.normal(0, 0.01, dirs.shape)
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                sites.append(dirs)
            return sites

        # At threshold 0.05: cfg0 aligned, cfg1 diverse
        # At threshold 0.20: cfg0 diverse, cfg1 aligned (inverted)
        data: dict[float, dict[str, list[np.ndarray]]] = {}
        data[0.05] = {
            "cfg0": _make_sites(2),  # z-axis → low delta_coh
            "cfg1": [rng.standard_normal((200, 3)) for _ in range(8)],
            "cfg2": _make_sites(0),
            "cfg3": _make_sites(1),
        }
        # Normalise random ones
        for cid in data[0.05]:
            data[0.05][cid] = [
                d / np.linalg.norm(d, axis=1, keepdims=True)
                for d in data[0.05][cid]
            ]
        data[0.20] = {
            "cfg0": [rng.standard_normal((200, 3)) for _ in range(8)],
            "cfg1": _make_sites(2),
            "cfg2": _make_sites(1),
            "cfg3": _make_sites(0),
        }
        for cid in data[0.20]:
            data[0.20][cid] = [
                d / np.linalg.norm(d, axis=1, keepdims=True)
                for d in data[0.20][cid]
            ]

        result = _test_gating(data)
        assert result["status"] in ("FLAG", "FAIL")


# ── Short trajectory → Test 3 flags ─────────────────────────────────

class TestTemporalShort:
    def test_short_insufficient(self):
        """T=20, n_segments=4 → 5 frames each < MIN_SEGMENT_FRAMES."""
        rng = np.random.default_rng(55)
        dirs = [rng.standard_normal((20, 3)) for _ in range(8)]
        dirs = [d / np.linalg.norm(d, axis=1, keepdims=True) for d in dirs]
        result = _test_temporal(dirs, n_segments=4)
        assert result["status"] == "INSUFFICIENT"


# ── flag_reasons populated ───────────────────────────────────────────

class TestFlagReasons:
    def test_non_pass_has_reasons(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = run_robustness_suite(dirs)
        if result["summary"] != "PASS":
            assert len(result["flag_reasons"]) > 0

    def test_reasons_contain_test_name(self):
        dirs = _stable_site_dirs(n_sites=4, t=100)
        result = run_robustness_suite(dirs)
        for reason in result["flag_reasons"]:
            assert any(
                name in reason
                for name in [
                    "test_gating", "test_pooling",
                    "test_temporal", "test_failure",
                ]
            )
