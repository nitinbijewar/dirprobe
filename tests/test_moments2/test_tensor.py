"""Tests for dirprobe.moments2.tensor."""

import inspect

import numpy as np
import pytest

from dirprobe._constants import EPS_DEGENERACY
from dirprobe.moments2.tensor import (
    check_degeneracy,
    compute_d_dir,
    compute_site_covariance,
    decompose_covariance,
    is_valid_directions,
)


# ── helpers ──────────────────────────────────────────────────────────

def _uniaxial_directions(n=1000, noise=0.01, seed=42):
    """Vectors along z with small noise."""
    rng = np.random.default_rng(seed)
    dirs = np.zeros((n, 3))
    dirs[:, 2] = 1.0
    dirs += rng.normal(0, noise, dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def _planar_directions(n=1000, noise=0.01, seed=42):
    """Uniform on xy great circle with small noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    dirs = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    dirs += rng.normal(0, noise, dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def _isotropic_directions(n=2000, seed=42):
    """Uniform on S^2."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(0, 1, (n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


# ── D_dir limits ─────────────────────────────────────────────────────

class TestDdirLimits:
    def test_uniaxial(self):
        dirs = _uniaxial_directions()
        c = compute_site_covariance(dirs)
        evals, _ = decompose_covariance(c)
        d = compute_d_dir(evals)
        assert abs(d - 1.0) < 0.1

    def test_planar(self):
        dirs = _planar_directions()
        c = compute_site_covariance(dirs)
        evals, _ = decompose_covariance(c)
        d = compute_d_dir(evals)
        assert abs(d - 2.0) < 0.2

    def test_isotropic(self):
        dirs = _isotropic_directions()
        c = compute_site_covariance(dirs)
        evals, _ = decompose_covariance(c)
        d = compute_d_dir(evals)
        assert abs(d - 3.0) < 0.15


# ── Trace normalisation ─────────────────────────────────────────────

class TestTraceNormalisation:
    def test_trace_uniaxial(self):
        c = compute_site_covariance(_uniaxial_directions())
        assert abs(np.trace(c) - 1.0) < 1e-6

    def test_trace_planar(self):
        c = compute_site_covariance(_planar_directions())
        assert abs(np.trace(c) - 1.0) < 1e-6

    def test_trace_isotropic(self):
        c = compute_site_covariance(_isotropic_directions())
        assert abs(np.trace(c) - 1.0) < 1e-6


# ── Dimension-agnostic ───────────────────────────────────────────────

class TestDimensionAgnostic:
    def test_d2(self):
        rng = np.random.default_rng(99)
        dirs = rng.normal(0, 1, (500, 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        c = compute_site_covariance(dirs)
        assert c.shape == (2, 2)
        assert abs(np.trace(c) - 1.0) < 1e-6

    def test_d3(self):
        c = compute_site_covariance(_isotropic_directions())
        assert c.shape == (3, 3)

    def test_d4(self):
        rng = np.random.default_rng(101)
        dirs = rng.normal(0, 1, (500, 4))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        c = compute_site_covariance(dirs)
        assert c.shape == (4, 4)
        evals, _ = decompose_covariance(c)
        d = compute_d_dir(evals)
        assert 1.0 <= d <= 4.0 + 0.1


# ── Input validation ─────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_input(self):
        with pytest.raises(ValueError, match="empty or non-2D"):
            compute_site_covariance(np.empty((0, 3)))

    def test_non_unit_vectors(self):
        dirs = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        with pytest.raises(ValueError, match="norms far from 1"):
            compute_site_covariance(dirs)

    def test_1d_input(self):
        with pytest.raises(ValueError, match="empty or non-2D"):
            compute_site_covariance(np.array([1.0, 0.0, 0.0]))


# ── check_degeneracy ────────────────────────────────────────────────

class TestDegeneracy:
    def test_top_pair_degenerate(self):
        d = check_degeneracy(np.array([0.5, 0.5, 0.0]))
        assert d["axis_degenerate"] is True
        assert d["is_degenerate"] is True

    def test_bottom_pair_degenerate_axis_ok(self):
        d = check_degeneracy(np.array([0.6, 0.2, 0.2]))
        assert d["axis_degenerate"] is False
        assert d["is_degenerate"] is True

    def test_well_separated(self):
        d = check_degeneracy(np.array([0.8, 0.1, 0.1]))
        assert d["axis_degenerate"] is False
        assert abs(d["min_adjacent_gap"] - 0.0) < 0.01  # bottom gap ~ 0

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_degeneracy(np.array([np.nan, 0.5, 0.5]))

    def test_non_normalised_raises(self):
        with pytest.raises(ValueError, match="not trace-normalised"):
            check_degeneracy(np.array([2.0, 1.0, 0.5]))

    def test_default_eps_is_constant(self):
        sig = inspect.signature(check_degeneracy)
        assert sig.parameters["eps"].default == EPS_DEGENERACY


# ── is_valid_directions ─────────────────────────────────────────────

class TestIsValidDirections:
    def test_valid(self):
        assert is_valid_directions(_uniaxial_directions()) is True

    def test_empty(self):
        assert is_valid_directions(np.empty((0, 3))) is False

    def test_non_unit(self):
        arr = np.array([[10.0, 0.0, 0.0]])
        assert is_valid_directions(arr) is False
