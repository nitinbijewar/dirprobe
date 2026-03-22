"""Temporal persistence diagnostics: windowed alignment (A) and block-to-full (B).

Extracted from v11 compute_phase2c.py:270 (compute_sector_persistence).
Degeneracy handling and generalised N-window support are NEW.
"""

from __future__ import annotations

import numpy as np

from dirprobe._constants import (
    DEFAULT_MIN_BLOCK_SIZE,
    DEFAULT_N_BLOCKS,
    DEFAULT_WINDOW_SIZE,
    EPS_DEGENERACY,
    PERSISTENCE_FAST,
    PERSISTENCE_LOCKED,
)
from dirprobe.moments2.tensor import (
    check_degeneracy,
    compute_site_covariance,
    decompose_covariance,
)


def _safe_nanmean(arr: np.ndarray) -> float:
    """nanmean that returns np.nan for empty or all-NaN arrays without warning."""
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _principal_axis(directions: np.ndarray, eps: float) -> np.ndarray | None:
    """Return principal eigenvector, or None if degenerate/invalid."""
    if directions.ndim != 2 or len(directions) < 1:
        return None
    if not np.isfinite(directions).all():
        return None
    c = compute_site_covariance(directions)
    evals, evecs = decompose_covariance(c)
    deg = check_degeneracy(evals, eps=eps)
    if deg["axis_degenerate"]:
        return None
    return evecs[:, 0]


def compute_windowed_alignment(
    directions: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    degeneracy_eps: float = EPS_DEGENERACY,
) -> np.ndarray:
    """Principal-axis alignment across consecutive non-overlapping windows.

    Parameters
    ----------
    directions : (T, d) unit vectors.
    window_size : int
        Frames per window. Tail remainder frames are dropped.
    degeneracy_eps : float
        Threshold for axis degeneracy.

    Returns
    -------
    alignments : 1D array of |e_w . e_{w+1}|^2 per consecutive pair.
        Empty array if < 2 windows.
    """
    t = len(directions)
    if t < window_size:
        return np.array([])

    # Non-overlapping windows; tail remainder dropped
    starts = list(range(0, t - window_size + 1, window_size))
    if len(starts) < 2:
        return np.array([])

    # Compute principal axis per window
    axes: list[np.ndarray | None] = []
    for s in starts:
        window = directions[s : s + window_size]
        axes.append(_principal_axis(window, degeneracy_eps))

    # Consecutive-pair alignments
    alignments: list[float] = []
    for i in range(len(axes) - 1):
        if axes[i] is None or axes[i + 1] is None:
            alignments.append(np.nan)
        else:
            dot = np.dot(axes[i], axes[i + 1])
            alignments.append(float(abs(dot) ** 2))

    return np.array(alignments)


def compute_block_to_full_alignment(
    directions: np.ndarray,
    n_blocks: int = DEFAULT_N_BLOCKS,
    min_block_size: int = DEFAULT_MIN_BLOCK_SIZE,
    degeneracy_eps: float = EPS_DEGENERACY,
) -> np.ndarray:
    """Alignment of each block's principal axis with full-trajectory axis.

    Parameters
    ----------
    directions : (T, d) unit vectors.
    n_blocks : int
        Number of blocks (uses np.array_split, no remainder dropped).
    min_block_size : int
        Minimum block length; if shortest block < this, return empty.
    degeneracy_eps : float
        Threshold for axis degeneracy.

    Returns
    -------
    alignments : 1D array of |e_block . e_full|^2 per block.
        Empty array if blocks too short.
    """
    blocks = np.array_split(directions, n_blocks)
    block_len_min = min(len(b) for b in blocks)
    if block_len_min < min_block_size:
        return np.array([])

    full_axis = _principal_axis(directions, degeneracy_eps)

    alignments: list[float] = []
    for block in blocks:
        if full_axis is None:
            alignments.append(np.nan)
            continue
        block_axis = _principal_axis(block, degeneracy_eps)
        if block_axis is None:
            alignments.append(np.nan)
        else:
            dot = np.dot(block_axis, full_axis)
            alignments.append(float(abs(dot) ** 2))

    return np.array(alignments)


def classify_persistence(
    A: float,
    B: float,
    threshold_locked: float = PERSISTENCE_LOCKED,
    threshold_fast: float = PERSISTENCE_FAST,
) -> str:
    """Classify persistence from A and B metrics (manuscript §2.3).

    Parameters
    ----------
    A : float
        Mean consecutive-window alignment (or np.nan).
    B : float
        Mean block-to-full alignment (or np.nan).
    threshold_locked : float
        Both A and B must exceed this for LOCKED.
    threshold_fast : float
        A below this yields FAST.

    Returns
    -------
    str
        One of 'LOCKED', 'SWITCHING', 'FAST', 'INSUFFICIENT'.
    """
    # 1. NaN check FIRST
    if np.isnan(A) or np.isnan(B):
        return "INSUFFICIENT"
    # 2. LOCKED: A > 0.8 AND B > 0.8
    if A > threshold_locked and B > threshold_locked:
        return "LOCKED"
    # 3. FAST: A < 0.4
    if A < threshold_fast:
        return "FAST"
    # 4. SWITCHING: 0.4 <= A <= 0.8
    return "SWITCHING"


def compute_persistence_metrics(
    site_directions: list[np.ndarray],
    window_size: int = DEFAULT_WINDOW_SIZE,
    n_blocks: int = DEFAULT_N_BLOCKS,
    degeneracy_eps: float = EPS_DEGENERACY,
) -> dict:
    """Per-site A (consecutive alignment) and B (block-to-full).

    Parameters
    ----------
    site_directions : list of (T_i, d) unit-vector arrays per site.
    window_size : int
        Window size for windowed alignment.
    n_blocks : int
        Number of blocks for block-to-full.
    degeneracy_eps : float
        Threshold for axis degeneracy.

    Returns
    -------
    dict
        Keys exactly: ``per_site_A``, ``per_site_B``, ``mean_A``,
        ``mean_B``, ``per_site_class``.
    """
    per_site_a: list[float] = []
    per_site_b: list[float] = []
    per_site_class: list[str] = []

    for dirs in site_directions:
        # Windowed alignment → A
        a_arr = compute_windowed_alignment(
            dirs, window_size=window_size, degeneracy_eps=degeneracy_eps
        )
        a_val = _safe_nanmean(a_arr)

        # Block-to-full → B
        b_arr = compute_block_to_full_alignment(
            dirs,
            n_blocks=n_blocks,
            min_block_size=DEFAULT_MIN_BLOCK_SIZE,
            degeneracy_eps=degeneracy_eps,
        )
        b_val = _safe_nanmean(b_arr)

        per_site_a.append(a_val)
        per_site_b.append(b_val)
        per_site_class.append(classify_persistence(a_val, b_val))

    valid_a = [v for v in per_site_a if np.isfinite(v)]
    valid_b = [v for v in per_site_b if np.isfinite(v)]
    mean_a = float(np.mean(valid_a)) if valid_a else np.nan
    mean_b = float(np.mean(valid_b)) if valid_b else np.nan

    return {
        "per_site_A": per_site_a,
        "per_site_B": per_site_b,
        "mean_A": mean_a,
        "mean_B": mean_b,
        "per_site_class": per_site_class,
    }
