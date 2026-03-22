"""Single rank-2 pipeline assembler.

All reproduce scripts and generator-mode verification call this function.
No ad-hoc pipeline assembly elsewhere.
"""

from __future__ import annotations

import numpy as np

from dirprobe._constants import EPS_DEGENERACY
from dirprobe.gating.amplitude import apply_amplitude_gate
from dirprobe.moments2.coherence import compute_delta_coh, compute_s_align
from dirprobe.moments2.tensor import compute_site_covariance
from dirprobe.robustness.suite import run_robustness_suite
from dirprobe.time.persistence import (
    classify_persistence,
    compute_block_to_full_alignment,
    compute_windowed_alignment,
)


def _safe_nanmean(arr: np.ndarray) -> float:
    """nanmean that returns np.nan for empty or all-NaN arrays without warning."""
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def run_pipeline(
    displacements: np.ndarray,
    gating_threshold: float = 0.05,
    persistence_windows: int = 4,
    degeneracy_eps: float | None = None,
    pooling_weights: np.ndarray | None = None,
) -> dict:
    """Run the full rank-2 diagnostic pipeline on displacement data.

    Parameters
    ----------
    displacements : (T, N, 3) array
        Pre-gating displacement vectors per frame per site.
    gating_threshold : float
        Amplitude gate threshold in angstrom.
    persistence_windows : int
        Number of windows W for persistence metrics.
    degeneracy_eps : float or None
        Degeneracy threshold. None = use EPS_DEGENERACY default.
    pooling_weights : (N,) array or None
        Pooling weights. None = equal weights.

    Returns
    -------
    dict with keys:
        d_dir_site: list of per-site D_dir
        d_dir_site_mean: float
        d_dir_pooled: float
        delta_coh: float
        s_align: float (or np.nan)
        a_mean: float (or np.nan)
        b_mean: float (or np.nan)
        classification: str
        robustness: str
        gated_counts: list of int
        site_directions: list of (T_gated, 3) arrays
    """
    if degeneracy_eps is None:
        degeneracy_eps = EPS_DEGENERACY

    T, N, d = displacements.shape

    # 1. Gate + normalise per site
    site_directions: list[np.ndarray] = []
    gated_counts: list[int] = []

    for site in range(N):
        site_disp = displacements[:, site, :]
        amps = np.linalg.norm(site_disp, axis=1)
        mask = amps >= gating_threshold

        n_gated = int(mask.sum())
        gated_counts.append(n_gated)

        if n_gated < 3:
            site_directions.append(np.zeros((0, d)))
            continue

        unit_vecs = apply_amplitude_gate(site_disp, gating_threshold)
        site_directions.append(unit_vecs)

    # 2. Compute delta_coh (site tensors + pooling + D_dir)
    dcoh_result = compute_delta_coh(site_directions, weights=pooling_weights)

    # 3. S_align from site covariances
    site_covs = dcoh_result["site_Cs"]
    valid_covs = [C for C in site_covs if np.isfinite(C).all()]
    if len(valid_covs) >= 2:
        s_val, s_info = compute_s_align(valid_covs, degeneracy_eps=degeneracy_eps)
    else:
        s_val = np.nan

    # 4. Persistence
    window_size = max(1, T // persistence_windows) if T > 0 else 1
    # Use per-site gated frame count for window sizing
    per_site_A: list[float] = []
    per_site_B: list[float] = []

    for dirs in site_directions:
        if len(dirs) < 4:
            per_site_A.append(np.nan)
            per_site_B.append(np.nan)
            continue

        site_T = len(dirs)
        ws = site_T // persistence_windows
        if ws < 2:
            per_site_A.append(np.nan)
            per_site_B.append(np.nan)
            continue

        a_arr = compute_windowed_alignment(
            dirs, window_size=ws, degeneracy_eps=degeneracy_eps
        )
        a_val = _safe_nanmean(a_arr)

        b_arr = compute_block_to_full_alignment(
            dirs, n_blocks=persistence_windows, min_block_size=2,
            degeneracy_eps=degeneracy_eps,
        )
        b_val = _safe_nanmean(b_arr)

        per_site_A.append(a_val)
        per_site_B.append(b_val)

    valid_A = [v for v in per_site_A if np.isfinite(v)]
    valid_B = [v for v in per_site_B if np.isfinite(v)]
    a_mean = float(np.mean(valid_A)) if valid_A else np.nan
    b_mean = float(np.mean(valid_B)) if valid_B else np.nan

    classification = classify_persistence(a_mean, b_mean)

    # 5. Robustness (basic — no multi-gate data)
    rob = run_robustness_suite(site_directions)
    robustness = rob.get("summary", "FLAG")

    return {
        "d_dir_site": dcoh_result["per_site_d_dir"],
        "d_dir_site_mean": dcoh_result["d_dir_site_mean"],
        "d_dir_pooled": dcoh_result["d_dir_pooled"],
        "delta_coh": dcoh_result["delta_coh"],
        "s_align": s_val,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "classification": classification,
        "robustness": robustness,
        "gated_counts": gated_counts,
        "site_directions": site_directions,
    }
