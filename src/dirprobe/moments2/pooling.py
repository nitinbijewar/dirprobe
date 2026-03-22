"""Pooling primitives for multi-site covariance tensors.

pool_covariances is the PRIMARY method (covariance-level averaging).
concat_directions is SECONDARY (convenience only, v11 compatibility).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pool_covariances(
    site_covariances: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted average of site covariance tensors, then trace-normalise.

    Parameters
    ----------
    site_covariances : list of (d, d) arrays.
        Non-finite matrices are excluded; weights are renormalised
        over valid sites.
    weights : (N_sites,) array or None
        Per-site weights. None = equal weights (1/N_valid).

    Returns
    -------
    C_pooled : (d, d) ndarray with Tr = 1.
        NaN matrix if no valid sites.
    """
    if not site_covariances:
        return np.full((3, 3), np.nan)

    d = site_covariances[0].shape[0]
    n = len(site_covariances)

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).copy()
        if len(w) != n:
            raise ValueError(
                f"weights length {len(w)} != site count {n}"
            )

    # Mask out non-finite sites
    valid_mask = np.array(
        [np.isfinite(c).all() for c in site_covariances]
    )
    w[~valid_mask] = 0.0

    w_sum = w.sum()
    if w_sum == 0:
        return np.full((d, d), np.nan)

    w_norm = w / w_sum
    c_pooled = np.zeros((d, d), dtype=float)
    for i, c in enumerate(site_covariances):
        if w_norm[i] > 0:
            c_pooled += w_norm[i] * c

    trace = np.trace(c_pooled)
    if trace <= 0 or not np.isfinite(trace):
        return np.full((d, d), np.nan)

    c_pooled /= trace
    return c_pooled


def concat_directions(
    site_directions: list[np.ndarray],
) -> np.ndarray:
    """Concatenate unit vectors across sites.

    Parameters
    ----------
    site_directions : list of (T_i, d) arrays.
        Empty arrays are skipped.

    Returns
    -------
    (T_total, d) ndarray
        Concatenated unit vectors.
    """
    non_empty = [
        d for d in site_directions
        if d.ndim == 2 and len(d) > 0
    ]
    if not non_empty:
        return np.empty((0, 3))
    return np.vstack(non_empty)
