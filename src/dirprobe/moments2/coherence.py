"""Coherence metrics: delta_coh, S_align, mean_sij.

Extracted from v11 compute_secondary.py (S_align, mean_Sij)
and compute_persite_ddir.py (delta_coh). Degeneracy handling is NEW.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dirprobe._constants import EPS_DEGENERACY
from dirprobe.moments2.tensor import (
    check_degeneracy,
    compute_d_dir,
    compute_site_covariance,
    decompose_covariance,
)
from dirprobe.moments2.pooling import pool_covariances


def _valid_site_axes(
    site_covariances: list[np.ndarray],
    degeneracy_eps: float,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract valid principal axes, excluding degenerate/non-finite sites.

    Parameters
    ----------
    site_covariances : list of (d, d) covariance matrices.
    degeneracy_eps : float
        Threshold for axis degeneracy (|lambda_1 - lambda_2| < eps).

    Returns
    -------
    axes : list of (d,) principal eigenvectors for valid sites.
    excluded_idx : list of int, indices of excluded sites.
    """
    axes: list[np.ndarray] = []
    excluded_idx: list[int] = []

    for i, c in enumerate(site_covariances):
        if not np.isfinite(c).all():
            excluded_idx.append(i)
            continue
        evals, evecs = decompose_covariance(c)
        if not np.isfinite(evals).all():
            excluded_idx.append(i)
            continue
        deg = check_degeneracy(evals, eps=degeneracy_eps)
        if deg["axis_degenerate"]:
            excluded_idx.append(i)
            continue
        axes.append(evecs[:, 0])

    return axes, excluded_idx


def compute_delta_coh(
    site_directions: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> dict:
    """Coherence gap = D_dir^pooled - <D_dir^site>.

    Parameters
    ----------
    site_directions : list of (T_i, d) unit-vector arrays per site.
        Empty arrays (T=0) are handled gracefully.
    weights : (N_sites,) array or None
        Pooling weights. None = equal weights over valid sites.

    Returns
    -------
    dict
        Keys exactly: ``d_dir_site_mean``, ``d_dir_pooled``,
        ``delta_coh``, ``per_site_d_dir``, ``pooled_C``, ``site_Cs``.
    """
    n_sites = len(site_directions)
    d = _infer_dimension(site_directions)

    site_cs: list[np.ndarray] = []
    per_site_d_dir: list[float] = []

    for dirs in site_directions:
        if dirs.ndim != 2 or len(dirs) == 0:
            site_cs.append(np.full((d, d), np.nan))
            per_site_d_dir.append(np.nan)
            continue
        c = compute_site_covariance(dirs)
        evals, _ = decompose_covariance(c)
        site_cs.append(c)
        per_site_d_dir.append(compute_d_dir(evals))

    pooled_c = pool_covariances(site_cs, weights=weights)

    if np.isfinite(pooled_c).all():
        p_evals, _ = decompose_covariance(pooled_c)
        d_dir_pooled = compute_d_dir(p_evals)
    else:
        d_dir_pooled = np.nan

    valid_d = [v for v in per_site_d_dir if np.isfinite(v)]
    d_dir_site_mean = float(np.mean(valid_d)) if valid_d else np.nan

    if np.isfinite(d_dir_pooled) and np.isfinite(d_dir_site_mean):
        delta_coh = d_dir_pooled - d_dir_site_mean
    else:
        delta_coh = np.nan

    return {
        "d_dir_site_mean": d_dir_site_mean,
        "d_dir_pooled": d_dir_pooled,
        "delta_coh": delta_coh,
        "per_site_d_dir": per_site_d_dir,
        "pooled_C": pooled_c,
        "site_Cs": site_cs,
    }


def compute_s_align(
    site_covariances: list[np.ndarray],
    degeneracy_eps: float = EPS_DEGENERACY,
) -> tuple[float, dict]:
    """Nematic alignment: max eigenvalue of (1/N_valid) sum(e_i e_i^T).

    Parameters
    ----------
    site_covariances : list of (d, d) covariance matrices.
    degeneracy_eps : float
        Threshold for axis degeneracy exclusion.

    Returns
    -------
    value : float or np.nan
        S_align. np.nan if < 2 valid sites.
    info : dict
        Keys exactly: ``n_axis_degenerate``, ``axis_degenerate_idx``,
        ``n_valid_sites``.
    """
    axes, excluded_idx = _valid_site_axes(site_covariances, degeneracy_eps)
    n_valid = len(axes)
    n_excluded = len(excluded_idx)

    info = {
        "n_axis_degenerate": n_excluded,
        "axis_degenerate_idx": excluded_idx,
        "n_valid_sites": n_valid,
    }

    if n_valid < 2:
        return np.nan, info

    d = len(axes[0])
    nematic = np.zeros((d, d))
    for e in axes:
        nematic += np.outer(e, e)
    nematic /= n_valid

    s_align = float(np.max(np.linalg.eigvalsh(nematic)))
    return s_align, info


def compute_mean_sij(
    site_covariances: list[np.ndarray],
    degeneracy_eps: float = EPS_DEGENERACY,
) -> tuple[float, dict]:
    """Mean pairwise alignment: mean(|e_i . e_j|^2) over all i<j valid pairs.

    Parameters
    ----------
    site_covariances : list of (d, d) covariance matrices.
    degeneracy_eps : float
        Threshold for axis degeneracy exclusion.

    Returns
    -------
    value : float or np.nan
        mean(S_ij). np.nan if < 2 valid sites.
    info : dict
        Keys exactly: ``n_axis_degenerate``, ``axis_degenerate_idx``,
        ``n_valid_sites``.
    """
    axes, excluded_idx = _valid_site_axes(site_covariances, degeneracy_eps)
    n_valid = len(axes)
    n_excluded = len(excluded_idx)

    info = {
        "n_axis_degenerate": n_excluded,
        "axis_degenerate_idx": excluded_idx,
        "n_valid_sites": n_valid,
    }

    if n_valid < 2:
        return np.nan, info

    pairs: list[float] = []
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            dot = np.dot(axes[i], axes[j])
            pairs.append(dot**2)

    return float(np.mean(pairs)), info


def _infer_dimension(site_directions: list[np.ndarray]) -> int:
    """Infer spatial dimension from first non-empty direction array."""
    for dirs in site_directions:
        if dirs.ndim == 2 and dirs.shape[1] > 0:
            return dirs.shape[1]
    return 3  # fallback
