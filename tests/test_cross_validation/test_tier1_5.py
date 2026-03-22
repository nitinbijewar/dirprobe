"""Tier 1.5: Degeneracy parity.

Category A: Fixture-derived parity (cfg6/cfg8 have atomic tensors).
Category B: All-config counts from degeneracy_flags.json.
Category C: Convention verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from dirprobe.moments2.tensor import check_degeneracy, decompose_covariance
from dirprobe.moments2.coherence import compute_s_align


# ── Category A: Fixture-only parity ────────────────────────────────


class TestDegeneracyFromTensors:
    """Recompute degeneracy flags from fixture tensors and compare."""

    @staticmethod
    def _flags_from_tensors(tensors, eps):
        flags = []
        for C in tensors:
            evals, _ = decompose_covariance(C)
            deg = check_degeneracy(evals, eps=eps)
            flags.append(deg["axis_degenerate"])
        return flags

    def test_cfg6_eps001(self, cfg6_site_tensors, degeneracy_flags):
        """Degeneracy flags for cfg6 at eps=0.01 match fixture."""
        flags_dp = self._flags_from_tensors(cfg6_site_tensors, eps=0.01)
        flags_fix = degeneracy_flags["6"]
        assert flags_dp == flags_fix, (
            f"Mismatch: dirprobe={flags_dp}, fixture={flags_fix}"
        )

    def test_cfg8_eps001(self, cfg8_site_tensors, degeneracy_flags):
        """Degeneracy flags for cfg8 at eps=0.01 match fixture."""
        flags_dp = self._flags_from_tensors(cfg8_site_tensors, eps=0.01)
        flags_fix = degeneracy_flags["8"]
        assert flags_dp == flags_fix, (
            f"Mismatch: dirprobe={flags_dp}, fixture={flags_fix}"
        )

    def test_cfg6_eps005(self, cfg6_site_tensors, degeneracy_flags):
        """Degeneracy flags for cfg6 at eps=0.05 also match fixture.
        (min spectral gap > 0.05 for all KNN sites)"""
        flags_dp = self._flags_from_tensors(cfg6_site_tensors, eps=0.05)
        flags_fix = degeneracy_flags["6"]
        assert flags_dp == flags_fix

    def test_cfg8_eps005(self, cfg8_site_tensors, degeneracy_flags):
        flags_dp = self._flags_from_tensors(cfg8_site_tensors, eps=0.05)
        flags_fix = degeneracy_flags["8"]
        assert flags_dp == flags_fix


# ── Category B: All-config counts ──────────────────────────────────


class TestDegeneracyCounts:
    def test_per_config_count(self, degeneracy_flags):
        """Per-config degenerate site count matches fixture."""
        for cfg_id_str in [str(i) for i in range(1, 11)]:
            if cfg_id_str not in degeneracy_flags:
                pytest.skip(f"Config {cfg_id_str} not in fixture")
            flags = degeneracy_flags[cfg_id_str]
            assert isinstance(flags, list)
            assert len(flags) == 8, f"Config {cfg_id_str}: expected 8 sites, got {len(flags)}"

    def test_total_degenerate(self, degeneracy_flags):
        """Total degenerate sites matches fixture metadata."""
        expected_total = degeneracy_flags.get("degenerate_count", None)
        if expected_total is None:
            pytest.skip("No degenerate_count in fixture")
        actual = sum(
            sum(degeneracy_flags[str(i)])
            for i in range(1, 11)
            if str(i) in degeneracy_flags
        )
        assert actual == expected_total, (
            f"Total degenerate: computed={actual}, fixture={expected_total}"
        )


# ── Category C: Convention verification ─────────────────────────────


class TestEpsilonEquivalence:
    """On KNN data, eps=0.01 and eps=0.05 produce identical flags
    because the minimum spectral gap across all 80 sites is ~0.06."""

    def _min_gap(self, tensors):
        gaps = []
        for C in tensors:
            evals, _ = decompose_covariance(C)
            evals_sorted = np.sort(evals)[::-1]
            gaps.append(evals_sorted[0] - evals_sorted[1])
        return min(gaps)

    def test_cfg6_min_gap_above_005(self, cfg6_site_tensors):
        """Minimum spectral gap in cfg6 exceeds 0.05."""
        gap = self._min_gap(cfg6_site_tensors)
        assert gap > 0.05, f"Min gap = {gap:.4f}, must be > 0.05"

    def test_cfg8_min_gap_above_005(self, cfg8_site_tensors):
        """Minimum spectral gap in cfg8 exceeds 0.05."""
        gap = self._min_gap(cfg8_site_tensors)
        assert gap > 0.05, f"Min gap = {gap:.4f}, must be > 0.05"

    def test_eps_equivalence_cfg6(self, cfg6_site_tensors):
        """eps=0.01 and eps=0.05 give identical flags for cfg6."""
        flags_01 = TestDegeneracyFromTensors._flags_from_tensors(cfg6_site_tensors, 0.01)
        flags_05 = TestDegeneracyFromTensors._flags_from_tensors(cfg6_site_tensors, 0.05)
        assert flags_01 == flags_05

    def test_eps_equivalence_cfg8(self, cfg8_site_tensors):
        flags_01 = TestDegeneracyFromTensors._flags_from_tensors(cfg8_site_tensors, 0.01)
        flags_05 = TestDegeneracyFromTensors._flags_from_tensors(cfg8_site_tensors, 0.05)
        assert flags_01 == flags_05


class TestSAlignExcludesDegenerate:
    def test_cfg6_s_align_with_degeneracy(self, cfg6_site_tensors):
        """S_align excludes degenerate sites (0 degenerate for cfg6)."""
        s_val, info = compute_s_align(list(cfg6_site_tensors), degeneracy_eps=0.01)
        assert info["n_valid_sites"] == 8
        assert info["n_axis_degenerate"] == 0
        assert np.isfinite(s_val)
