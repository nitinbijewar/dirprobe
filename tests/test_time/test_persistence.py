"""Tests for dirprobe.time.persistence."""

import inspect

import numpy as np
import pytest

from dirprobe._constants import (
    DEFAULT_WINDOW_SIZE,
    PERSISTENCE_LOCKED,
    PERSISTENCE_FAST,
)
from dirprobe.time.persistence import (
    classify_persistence,
    compute_block_to_full_alignment,
    compute_persistence_metrics,
    compute_windowed_alignment,
)


# ── helpers ──────────────────────────────────────────────────────────

def _constant_axis(t=500, noise=0.01, seed=42):
    """Vectors along z with small noise."""
    rng = np.random.default_rng(seed)
    dirs = np.zeros((t, 3))
    dirs[:, 2] = 1.0
    dirs += rng.normal(0, noise, dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def _alternating_axes(t=500, window_size=50, seed=42):
    """Alternate between z-axis and x-axis per window."""
    rng = np.random.default_rng(seed)
    dirs = np.zeros((t, 3))
    n_windows = t // window_size
    for w in range(n_windows):
        s = w * window_size
        e = s + window_size
        axis = 2 if w % 2 == 0 else 0
        dirs[s:e, axis] = 1.0
        dirs[s:e] += rng.normal(0, 0.01, (window_size, 3))
        norms = np.linalg.norm(dirs[s:e], axis=1, keepdims=True)
        dirs[s:e] /= norms
    return dirs


def _random_directions(t=500, seed=42):
    """Uniform random on S^2."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(0, 1, (t, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


# ── windowed alignment ───────────────────────────────────────────────

class TestWindowedAlignment:
    def test_constant_axis_high_alignment(self):
        dirs = _constant_axis(t=500)
        a = compute_windowed_alignment(dirs, window_size=50)
        assert len(a) == 9  # 10 windows, 9 pairs
        assert np.nanmean(a) > 0.95

    def test_alternating_axes_low_alignment(self):
        dirs = _alternating_axes(t=500, window_size=50)
        a = compute_windowed_alignment(dirs, window_size=50)
        assert np.nanmean(a) < 0.1

    def test_short_trace_empty(self):
        dirs = _random_directions(t=10)
        a = compute_windowed_alignment(dirs, window_size=50)
        assert len(a) == 0

    def test_one_window_empty(self):
        dirs = _constant_axis(t=50)
        a = compute_windowed_alignment(dirs, window_size=50)
        assert len(a) == 0

    def test_non_overlapping_drop_tail(self):
        """T=110, window_size=50 → 2 windows (0-49, 50-99), drop 100-109."""
        dirs = _random_directions(t=110)
        a = compute_windowed_alignment(dirs, window_size=50)
        assert len(a) == 1  # 2 windows → 1 consecutive pair

    def test_sign_flip_invariance(self):
        """Negating axis in alternating windows doesn't change A."""
        dirs = _constant_axis(t=500, noise=0.01)
        a_normal = compute_windowed_alignment(dirs, window_size=50)
        # Forcibly negate odd windows
        dirs_flipped = dirs.copy()
        for w in range(1, 10, 2):
            s = w * 50
            dirs_flipped[s : s + 50] *= -1
        a_flipped = compute_windowed_alignment(dirs_flipped, window_size=50)
        np.testing.assert_allclose(a_normal, a_flipped, atol=1e-6)


# ── block-to-full alignment ─────────────────────────────────────────

class TestBlockToFull:
    def test_constant_axis_high(self):
        dirs = _constant_axis(t=500)
        b = compute_block_to_full_alignment(dirs, n_blocks=2)
        assert np.nanmean(b) > 0.95

    def test_short_returns_empty(self):
        dirs = _random_directions(t=20)
        b = compute_block_to_full_alignment(
            dirs, n_blocks=2, min_block_size=50
        )
        assert len(b) == 0


# ── classify_persistence ─────────────────────────────────────────────

class TestClassify:
    def test_insufficient_nan_a(self):
        assert classify_persistence(np.nan, 0.9) == "INSUFFICIENT"

    def test_insufficient_nan_b(self):
        assert classify_persistence(0.9, np.nan) == "INSUFFICIENT"

    def test_locked(self):
        assert classify_persistence(0.9, 0.9) == "LOCKED"

    def test_fast(self):
        assert classify_persistence(0.3, 0.9) == "FAST"

    def test_switching(self):
        assert classify_persistence(0.6, 0.6) == "SWITCHING"

    def test_defaults_match_constants(self):
        sig = inspect.signature(classify_persistence)
        assert sig.parameters["threshold_locked"].default == PERSISTENCE_LOCKED
        assert sig.parameters["threshold_fast"].default == PERSISTENCE_FAST


# ── compute_persistence_metrics ──────────────────────────────────────

class TestPersistenceMetrics:
    def test_return_keys(self):
        sites = [_constant_axis(t=200)]
        result = compute_persistence_metrics(sites, window_size=50)
        expected = {
            "per_site_A", "per_site_B", "mean_A", "mean_B",
            "per_site_class",
        }
        assert set(result.keys()) == expected

    def test_constant_axis_locked(self):
        sites = [_constant_axis(t=500)]
        result = compute_persistence_metrics(sites, window_size=50)
        assert result["per_site_class"][0] == "LOCKED"

    def test_random_drifting(self):
        sites = [_random_directions(t=500)]
        result = compute_persistence_metrics(sites, window_size=50)
        # With EPS_DEGENERACY=0.05, random directions in 50-frame windows
        # may have spectral gap < 0.05 → INSUFFICIENT is also valid.
        assert result["per_site_class"][0] in ("SWITCHING", "FAST", "INSUFFICIENT")

    def test_default_window_size(self):
        sig = inspect.signature(compute_persistence_metrics)
        assert sig.parameters["window_size"].default == DEFAULT_WINDOW_SIZE
