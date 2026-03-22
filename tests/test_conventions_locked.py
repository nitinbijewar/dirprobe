"""Convention enforcement tests — one per CONVENTIONS.md section.

These tests are the executable counterpart of CONVENTIONS.md.
Changing any convention requires updating this file, CONVENTIONS.md,
and the manuscript simultaneously.
"""

from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "dirprobe"


# ── Convention 1: Covariance-level pooling ──────────────────────────


class TestConvention1CovariancePooling:
    def test_pool_is_arithmetic_mean(self):
        """Pooled tensor equals arithmetic mean of site tensors."""
        from dirprobe.moments2.pooling import pool_covariances

        C1 = np.array([[0.6, 0.1, 0.0],
                        [0.1, 0.3, 0.0],
                        [0.0, 0.0, 0.1]])
        C2 = np.array([[0.4, 0.0, 0.1],
                        [0.0, 0.5, 0.0],
                        [0.1, 0.0, 0.1]])
        result = pool_covariances([C1, C2])
        expected = (C1 + C2) / 2.0
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_pooled_trace_preserved(self):
        """Pooling trace-normalised tensors preserves Tr = 1."""
        from dirprobe.moments2.pooling import pool_covariances

        C1 = np.diag([0.5, 0.3, 0.2])
        C2 = np.diag([0.4, 0.4, 0.2])
        result = pool_covariances([C1, C2])
        assert abs(np.trace(result) - 1.0) < 1e-15


# ── Convention 2: Absolute amplitude gate ───────────────────────────


class TestConvention2AbsoluteGate:
    def test_gate_uses_absolute_threshold(self):
        """Gate at 0.20 Å keeps only vectors with |x| >= 0.20."""
        from dirprobe.gating.amplitude import apply_amplitude_gate

        vectors = np.array([
            [0.01, 0.0, 0.0],   # |x| = 0.01 → excluded
            [0.15, 0.0, 0.0],   # |x| = 0.15 → excluded
            [0.30, 0.0, 0.0],   # |x| = 0.30 → kept
        ])
        result = apply_amplitude_gate(vectors, threshold=0.20)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(
            result[0], [1.0, 0.0, 0.0], atol=1e-10
        )


# ── Convention 3: Squared-dot nematic alignment ─────────────────────


class TestConvention3SquaredDot:
    def test_antiparallel_sij_is_one(self):
        """Antiparallel axes: S_ij = |ê · (-ê)|² = 1.0."""
        from dirprobe.moments2.coherence import compute_mean_sij
        from dirprobe.moments2.tensor import compute_site_covariance

        e1 = np.array([[1.0, 0.0, 0.0]] * 50)
        e2 = np.array([[-1.0, 0.0, 0.0]] * 50)
        C1 = compute_site_covariance(e1)
        C2 = compute_site_covariance(e2)
        val, _ = compute_mean_sij([C1, C2], degeneracy_eps=0.01)
        assert abs(val - 1.0) < 1e-10

    def test_orthogonal_sij_is_zero(self):
        """Orthogonal axes: S_ij = 0.0."""
        from dirprobe.moments2.coherence import compute_mean_sij
        from dirprobe.moments2.tensor import compute_site_covariance

        e1 = np.array([[1.0, 0.0, 0.0]] * 50)
        e2 = np.array([[0.0, 1.0, 0.0]] * 50)
        C1 = compute_site_covariance(e1)
        C2 = compute_site_covariance(e2)
        val, _ = compute_mean_sij([C1, C2], degeneracy_eps=0.01)
        assert abs(val) < 1e-10

    def test_45deg_sij_is_half(self):
        """45-degree axes: S_ij = |cos(45°)|² = 0.5."""
        from dirprobe.moments2.coherence import compute_mean_sij
        from dirprobe.moments2.tensor import compute_site_covariance

        e1 = np.array([[1.0, 0.0, 0.0]] * 50)
        e2_dir = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        e2 = np.tile(e2_dir, (50, 1))
        C1 = compute_site_covariance(e1)
        C2 = compute_site_covariance(e2)
        val, _ = compute_mean_sij([C1, C2], degeneracy_eps=0.01)
        assert abs(val - 0.5) < 1e-10


# ── Convention 4: Cartesian coordinates ─────────────────────────────


class TestConvention4Cartesian:
    def test_gating_preserves_cartesian_direction(self):
        """Gating + normalisation preserves Cartesian direction."""
        from dirprobe.gating.amplitude import apply_amplitude_gate

        direction = np.array([0.1, 0.2, 0.3])
        vectors = direction[np.newaxis, :]  # (1, 3)
        result = apply_amplitude_gate(vectors, threshold=0.0)
        expected = direction / np.linalg.norm(direction)
        np.testing.assert_allclose(result[0], expected, atol=1e-15)


# ── Convention 5: EPS_DEGENERACY = 0.05 ─────────────────────────────


class TestConvention5Epsilon:
    def test_constant_value(self):
        from dirprobe._constants import EPS_DEGENERACY
        assert EPS_DEGENERACY == 0.05

    def test_below_threshold_flagged(self):
        """Gap = 0.04 < 0.05 → degenerate."""
        from dirprobe.moments2.tensor import check_degeneracy
        evals = np.array([0.37, 0.33, 0.30])
        result = check_degeneracy(evals, eps=0.05)
        assert result["axis_degenerate"] is True

    def test_above_threshold_not_flagged(self):
        """Gap = 0.06 > 0.05 → not degenerate."""
        from dirprobe.moments2.tensor import check_degeneracy
        evals = np.array([0.40, 0.34, 0.26])
        result = check_degeneracy(evals, eps=0.05)
        assert result["axis_degenerate"] is False


# ── Convention 6: Persistence accepts explicit W ─────────────────────


class TestConvention6PersistenceParameterized:
    def test_windowed_alignment_accepts_window_size(self):
        """compute_windowed_alignment accepts window_size parameter."""
        from dirprobe.time.persistence import compute_windowed_alignment
        import inspect
        sig = inspect.signature(compute_windowed_alignment)
        assert "window_size" in sig.parameters

    def test_block_to_full_accepts_n_blocks(self):
        """compute_block_to_full_alignment accepts n_blocks parameter."""
        from dirprobe.time.persistence import compute_block_to_full_alignment
        import inspect
        sig = inspect.signature(compute_block_to_full_alignment)
        assert "n_blocks" in sig.parameters

    def test_constant_axis_high_persistence_w2(self):
        """Constant axis → A ≈ 1, B ≈ 1 with W=2."""
        from dirprobe.time.persistence import (
            compute_block_to_full_alignment,
            compute_windowed_alignment,
        )
        rng = np.random.default_rng(42)
        mu = np.array([1.0, 0.0, 0.0])
        noise = rng.normal(0, 0.01, (200, 3))
        dirs = mu + noise
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        a_arr = compute_windowed_alignment(dirs, window_size=100, degeneracy_eps=0.05)
        b_arr = compute_block_to_full_alignment(dirs, n_blocks=2, min_block_size=2, degeneracy_eps=0.05)
        assert float(np.nanmean(a_arr)) > 0.99
        assert float(np.nanmean(b_arr)) > 0.99

    def test_constant_axis_high_persistence_w4(self):
        """Constant axis → A ≈ 1, B ≈ 1 with W=4."""
        from dirprobe.time.persistence import (
            compute_block_to_full_alignment,
            compute_windowed_alignment,
        )
        rng = np.random.default_rng(42)
        mu = np.array([1.0, 0.0, 0.0])
        noise = rng.normal(0, 0.01, (200, 3))
        dirs = mu + noise
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        a_arr = compute_windowed_alignment(dirs, window_size=50, degeneracy_eps=0.05)
        b_arr = compute_block_to_full_alignment(dirs, n_blocks=4, min_block_size=2, degeneracy_eps=0.05)
        assert float(np.nanmean(a_arr)) > 0.99
        assert float(np.nanmean(b_arr)) > 0.99


# ── Convention 7: No knn_coherence imports ───────────────────────────


class TestConvention7Independence:
    def test_no_knn_coherence_imports(self):
        """No source file imports from knn_coherence."""
        violations = []
        for path in SRC_DIR.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                if "import knn_coherence" in line or "from knn_coherence" in line:
                    violations.append(f"{path}:{i}: {line.strip()}")
        assert violations == [], f"knn_coherence imports found:\n" + "\n".join(violations)

    def test_no_hardcoded_knn_paths(self):
        """No source file contains hardcoded knn_coherence or subprobe paths."""
        forbidden = ["/home/nitinb/knn_coherence", "/home/nitinb/subprobe"]
        violations = []
        for path in SRC_DIR.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                for pattern in forbidden:
                    if pattern in line:
                        violations.append(f"{path}:{i}: {line.strip()}")
        assert violations == [], f"Hardcoded paths found:\n" + "\n".join(violations)


# ── Convention 8: Bundled data checksums ─────────────────────────────


class TestConvention8BundledData:
    def test_checksum_file_exists(self):
        checksum_path = SRC_DIR / "data" / "DATA_SHA256SUMS.txt"
        assert checksum_path.exists(), f"{checksum_path} not found"

    def test_all_checksums_match(self):
        checksum_path = SRC_DIR / "data" / "DATA_SHA256SUMS.txt"
        text = checksum_path.read_text()
        for line in text.strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            expected_hash, fname = parts
            # Try knn/ first, then synthetic/
            fpath = SRC_DIR / "data" / "knn" / fname
            if not fpath.exists():
                fpath = SRC_DIR / "data" / "synthetic" / fname
            assert fpath.exists(), f"File not found: {fname} (checked knn/ and synthetic/)"
            actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
            assert actual == expected_hash, (
                f"{fname}: expected {expected_hash[:16]}..., got {actual[:16]}..."
            )

    def test_verify_bundled_data_function(self):
        from dirprobe.data import verify_bundled_data
        ok, details = verify_bundled_data()
        assert ok is True, f"verify_bundled_data failed: {details}"
