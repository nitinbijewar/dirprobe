"""Amplitude gating: exclude low-amplitude frames, return unit vectors.

Uses absolute threshold gating (not percentile-based).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_amplitude_gate(
    vectors: np.ndarray,
    threshold: float,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Exclude frames where |vector| < threshold. Return unit vectors.

    Parameters
    ----------
    vectors : (T, d) raw displacement vectors.
    threshold : float
        Amplitude gate. Frames with norm < threshold are excluded.
    return_mask : bool
        If True, also return the boolean mask.

    Returns
    -------
    unit_vectors : (N_surviving, d) normalised survivors. Shape (0, d) if
        all excluded.
    mask : (T,) bool array (only if return_mask=True).
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected 2D array, got ndim={vectors.ndim}")

    d = vectors.shape[1]
    norms = np.linalg.norm(vectors, axis=1)
    mask = norms >= threshold

    surviving = vectors[mask]
    if len(surviving) > 0:
        surviving = surviving / norms[mask, np.newaxis]
    else:
        surviving = np.empty((0, d))

    if return_mask:
        return surviving, mask
    return surviving


def compute_off_centring(
    site_positions: np.ndarray,
    cage_positions: np.ndarray,
) -> np.ndarray:
    """Off-centring displacement = site_position - mean(cage_positions).

    Parameters
    ----------
    site_positions : (T, 3) positions of the B-site atom over time.
    cage_positions : (T, N_cage, 3) positions of cage atoms over time.

    Returns
    -------
    displacements : (T, 3) off-centring vectors in Cartesian coordinates.
    """
    centroid = np.mean(cage_positions, axis=1)  # (T, 3)
    return site_positions - centroid
