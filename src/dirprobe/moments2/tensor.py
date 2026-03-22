"""Rank-2 second-moment tensor: covariance, eigendecomposition, D_dir.

Extracted from v11 compute_persite_ddir.py:220 (compute_ddir).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dirprobe._constants import EPS_DEGENERACY


def is_valid_directions(arr: np.ndarray) -> bool:
    """Check whether *arr* is a valid unit-direction array.

    Parameters
    ----------
    arr : ndarray
        Candidate direction array.

    Returns
    -------
    bool
        True iff arr.ndim == 2, len(arr) > 0, all entries finite,
        and all row norms within [1-1e-2, 1+1e-2].
    """
    if arr.ndim != 2 or len(arr) == 0:
        return False
    if not np.isfinite(arr).all():
        return False
    norms = np.linalg.norm(arr, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= 1e-2))


def compute_site_covariance(directions: np.ndarray) -> np.ndarray:
    """Trace-normalised second-moment tensor C = <u (x) u> with Tr(C) = 1.

    Parameters
    ----------
    directions : (T, d) array of unit vectors, T >= 1.

    Returns
    -------
    C : (d, d) symmetric positive semi-definite, Tr(C) = 1.

    Raises
    ------
    ValueError
        On empty input, non-finite values, near-zero vectors,
        or norms far from 1.
    """
    # 1. empty or non-2D
    if directions.ndim != 2 or len(directions) == 0:
        raise ValueError("empty or non-2D input")

    # 2. non-finite
    if not np.isfinite(directions).all():
        raise ValueError("non-finite values in directions")

    norms = np.linalg.norm(directions, axis=1)

    # 3. near-zero vectors
    if np.any(norms < 1e-12):
        raise ValueError("near-zero vector; likely upstream error")

    # 4. norms far from 1
    max_dev = np.max(np.abs(norms - 1.0))
    if max_dev > 1e-2:
        raise ValueError(
            "expected unit vectors; got norms far from 1. "
            "Did you pass raw displacements?"
        )

    # 5. silent re-normalisation for small drift
    if max_dev > 1e-6:
        directions = directions / norms[:, None]

    # Raw second moment (NOT np.cov which subtracts mean)
    t = len(directions)
    c_raw = directions.T @ directions / t
    trace = np.trace(c_raw)

    # 6. post-computation guard
    if not np.isfinite(c_raw).all() or trace <= 0:
        raise ValueError(
            "covariance computation produced non-finite or non-positive trace"
        )

    c = c_raw / trace
    return c


def decompose_covariance(
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of symmetric covariance tensor.

    Parameters
    ----------
    C : (d, d) symmetric matrix.

    Returns
    -------
    eigenvalues : (d,) descending order, sum ~ 1.
    eigenvectors : (d, d) columns are eigenvectors; evecs[:, i] for
        eigenvalue i.
    """
    c_sym = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(c_sym)
    # eigh returns ascending; reverse to descending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    return evals, evecs


def compute_d_dir(eigenvalues: np.ndarray) -> float:
    """Directional dimensionality D_dir = 1 / sum(lambda_i^2).

    Parameters
    ----------
    eigenvalues : (d,) trace-normalised eigenvalues (sum ~ 1).

    Returns
    -------
    float
        D_dir in [1, d].
    """
    return float(1.0 / np.sum(eigenvalues**2))


def check_degeneracy(
    eigenvalues: np.ndarray,
    eps: float = EPS_DEGENERACY,
) -> dict:
    """Check eigenvalue degeneracy on trace-normalised spectrum.

    Parameters
    ----------
    eigenvalues : (d,) finite, trace-normalised (sum ~ 1).
    eps : float
        Gap threshold for degeneracy detection.

    Returns
    -------
    dict
        Keys exactly: ``is_degenerate``, ``degenerate_pairs``,
        ``min_adjacent_gap``, ``axis_degenerate``.

    Raises
    ------
    ValueError
        If eigenvalues are non-finite or not trace-normalised.
    """
    if not np.isfinite(eigenvalues).all():
        raise ValueError("non-finite eigenvalues")
    if abs(np.sum(eigenvalues) - 1.0) > 1e-6:
        raise ValueError(
            f"eigenvalues not trace-normalised: sum={np.sum(eigenvalues)}"
        )

    # Sort descending (defensive — should already be sorted)
    idx = np.argsort(eigenvalues)[::-1]
    evals = eigenvalues[idx]

    gaps = []
    degenerate_pairs: list[tuple[int, int]] = []
    for i in range(len(evals) - 1):
        gap = evals[i] - evals[i + 1]
        gaps.append(gap)
        if abs(gap) < eps:
            degenerate_pairs.append((i, i + 1))

    min_gap = float(min(gaps)) if gaps else float("inf")

    # axis_degenerate: ONLY if top pair |lambda_1 - lambda_2| < eps
    axis_degenerate = bool(
        len(evals) >= 2 and abs(evals[0] - evals[1]) < eps
    )

    return {
        "is_degenerate": bool(len(degenerate_pairs) > 0),
        "degenerate_pairs": degenerate_pairs,
        "min_adjacent_gap": min_gap,
        "axis_degenerate": axis_degenerate,
    }
