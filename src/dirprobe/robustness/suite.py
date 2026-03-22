"""Four-test robustness suite for directional diagnostics.

Test 1: Gating sensitivity (Spearman rank preservation of delta_coh)
Test 2: Pooling sensitivity (equal vs frame-weighted covariance pooling)
Test 3: Temporal stability (D_dir relative range across time segments)
Test 4: Synthetic failure detection (adversarial inputs correctly flagged)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from dirprobe._constants import (
    MIN_SEGMENT_FRAMES,
    POOLING_FLAG,
    POOLING_PASS,
    RHO_FLAG,
    RHO_PASS,
    TEMPORAL_FLAG,
    TEMPORAL_PASS,
)
from dirprobe.moments2.coherence import compute_delta_coh
from dirprobe.moments2.tensor import (
    compute_d_dir,
    compute_site_covariance,
    decompose_covariance,
)


# ── Test 1: Gating sensitivity ──────────────────────────────────────


def test_gating_sensitivity(
    site_directions_by_gate: dict[float, dict[str, list[np.ndarray]]],
) -> dict:
    """Test 1: Spearman rank ordering of delta_coh across gate thresholds.

    Parameters
    ----------
    site_directions_by_gate : dict
        site_directions_by_gate[threshold][config_id] = list of (T_i, d)
        arrays, already gated and unit-normalised.

    Returns
    -------
    dict
        Keys exactly: ``status``, ``pairwise_rho``, ``n_insufficient_pairs``.
    """
    thresholds = sorted(site_directions_by_gate.keys())

    if len(thresholds) < 2:
        return {
            "status": "INSUFFICIENT",
            "pairwise_rho": {},
            "n_insufficient_pairs": 0,
        }

    # Compute delta_coh per config per threshold
    dcoh_by_threshold: dict[float, dict[str, float]] = {}
    for t in thresholds:
        dcoh_by_threshold[t] = {}
        for cid, site_dirs in site_directions_by_gate[t].items():
            result = compute_delta_coh(site_dirs)
            dcoh_by_threshold[t][cid] = result["delta_coh"]

    # Pairwise Spearman
    pairwise_rho: dict[str, float | str] = {}
    n_insufficient = 0

    for i in range(len(thresholds)):
        for j in range(i + 1, len(thresholds)):
            ti, tj = thresholds[i], thresholds[j]
            key = f"({ti}, {tj})"
            rho = _pairwise_spearman(
                dcoh_by_threshold[ti], dcoh_by_threshold[tj]
            )
            pairwise_rho[key] = rho
            if rho == "INSUFFICIENT":
                n_insufficient += 1

    status = _gating_verdict(pairwise_rho, n_insufficient)

    return {
        "status": status,
        "pairwise_rho": pairwise_rho,
        "n_insufficient_pairs": n_insufficient,
    }


def _pairwise_spearman(
    dcoh_a: dict[str, float], dcoh_b: dict[str, float]
) -> float | str:
    """Spearman rho between two delta_coh vectors, excluding NaN configs."""
    common = set(dcoh_a.keys()) & set(dcoh_b.keys())
    vals_a = []
    vals_b = []
    for cid in sorted(common):
        a, b = dcoh_a[cid], dcoh_b[cid]
        if np.isfinite(a) and np.isfinite(b):
            vals_a.append(a)
            vals_b.append(b)

    if len(vals_a) < 3:
        return "INSUFFICIENT"

    arr_a = np.array(vals_a)
    arr_b = np.array(vals_b)

    # Constant-array edge cases
    a_const = np.all(arr_a == arr_a[0])
    b_const = np.all(arr_b == arr_b[0])
    if a_const and b_const:
        if np.allclose(arr_a, arr_b):
            return 1.0
        return "INSUFFICIENT"
    if a_const or b_const:
        return "INSUFFICIENT"

    rho, _ = sp_stats.spearmanr(arr_a, arr_b)
    return float(rho)


def _gating_verdict(
    pairwise_rho: dict[str, float | str], n_insufficient: int
) -> str:
    """Determine Test 1 verdict from pairwise rho values."""
    numeric = [v for v in pairwise_rho.values() if isinstance(v, (int, float))]
    if not numeric:
        return "INSUFFICIENT"
    min_rho = min(numeric)
    if min_rho < RHO_FLAG:
        return "FAIL"
    if min_rho < RHO_PASS:
        return "FLAG"
    return "PASS"


# ── Test 2: Pooling sensitivity ──────────────────────────────────────


def test_pooling_sensitivity(
    site_directions: list[np.ndarray],
    site_n_frames: np.ndarray | None = None,
) -> dict:
    """Test 2: delta_coh stability across pooling conventions.

    Parameters
    ----------
    site_directions : list of (T_i, d) arrays per site.
    site_n_frames : (N_sites,) post-gating frame counts, or None.

    Returns
    -------
    dict
        Keys exactly: ``status``, ``diff``.
    """
    if site_n_frames is None:
        return {"status": "INSUFFICIENT", "diff": np.nan}

    result_equal = compute_delta_coh(site_directions, weights=None)
    dcoh_equal = result_equal["delta_coh"]

    frame_weights = np.asarray(site_n_frames, dtype=float)
    result_frame = compute_delta_coh(site_directions, weights=frame_weights)
    dcoh_frame = result_frame["delta_coh"]

    if not np.isfinite(dcoh_equal) or not np.isfinite(dcoh_frame):
        return {"status": "INSUFFICIENT", "diff": np.nan}

    diff = abs(dcoh_equal - dcoh_frame)
    status = _threshold_verdict(diff, POOLING_PASS, POOLING_FLAG)

    return {"status": status, "diff": float(diff)}


# ── Test 3: Temporal stability ───────────────────────────────────────


def test_temporal_stability(
    site_directions: list[np.ndarray],
    n_segments: int = 4,
) -> dict:
    """Test 3: D_dir relative range across time segments.

    Parameters
    ----------
    site_directions : list of (T_i, d) arrays per site.
    n_segments : int
        Number of contiguous segments per site.

    Returns
    -------
    dict
        Keys exactly: ``status``, ``mean_rel_range``.
    """
    rel_ranges: list[float] = []

    for dirs in site_directions:
        rr = _site_rel_range(dirs, n_segments)
        rel_ranges.append(rr)

    arr = np.array(rel_ranges)
    if np.all(np.isnan(arr)):
        return {"status": "INSUFFICIENT", "mean_rel_range": np.nan}

    mean_rr = float(np.nanmean(arr))

    if np.isnan(mean_rr):
        status = "INSUFFICIENT"
    elif mean_rr < TEMPORAL_PASS:
        status = "PASS"
    elif mean_rr < TEMPORAL_FLAG:
        status = "FLAG"
    else:
        status = "FAIL"

    return {"status": status, "mean_rel_range": mean_rr}


def _site_rel_range(dirs: np.ndarray, n_segments: int) -> float:
    """Relative range of D_dir across segments for one site."""
    if dirs.ndim != 2 or len(dirs) == 0:
        return np.nan

    segments = np.array_split(dirs, n_segments)
    d_dirs: list[float] = []

    for seg in segments:
        if len(seg) < MIN_SEGMENT_FRAMES:
            d_dirs.append(np.nan)
            continue
        c = compute_site_covariance(seg)
        evals, _ = decompose_covariance(c)
        d_dirs.append(compute_d_dir(evals))

    arr = np.array(d_dirs)
    if np.any(np.isnan(arr)):
        return np.nan

    mean_val = np.mean(arr)
    if mean_val == 0:
        return np.nan
    return float((np.max(arr) - np.min(arr)) / mean_val)


# ── Test 4: Synthetic failure detection ──────────────────────────────


def test_synthetic_failure_detection(
    results: dict[str, dict],
    known_failures: dict[str, list[str]] | None = None,
) -> dict:
    """Test 4: Pipeline correctly flags adversarial inputs.

    Parameters
    ----------
    results : dict mapping system names to run_robustness_suite() output.
    known_failures : maps system names to expected failing subtests.

    Returns
    -------
    dict
        Keys exactly: ``status``, ``expected_failures``, ``actual_failures``.
    """
    if known_failures is None:
        return {
            "status": "INSUFFICIENT",
            "expected_failures": {},
            "actual_failures": _collect_actual_failures(results),
        }

    actual = _collect_actual_failures(results)
    all_match = True

    for sys_name, expected_tests in known_failures.items():
        actual_tests = actual.get(sys_name, [])
        for et in expected_tests:
            if et not in actual_tests:
                all_match = False
                break
        if not all_match:
            break

    return {
        "status": "PASS" if all_match else "FAIL",
        "expected_failures": dict(known_failures),
        "actual_failures": actual,
    }


def _collect_actual_failures(results: dict[str, dict]) -> dict[str, list[str]]:
    """Extract which subtests FAILed for each system."""
    actual: dict[str, list[str]] = {}
    test_keys = [
        ("test_gating", "test_gating_sensitivity"),
        ("test_pooling", "test_pooling_sensitivity"),
        ("test_temporal", "test_temporal_stability"),
        ("test_failure", "test_synthetic_failure_detection"),
    ]
    for sys_name, suite_result in results.items():
        fails = []
        for result_key, test_name in test_keys:
            sub = suite_result.get(result_key, {})
            if sub.get("status") == "FAIL":
                fails.append(test_name)
        actual[sys_name] = fails
    return actual


# ── Suite runner ─────────────────────────────────────────────────────


def run_robustness_suite(
    site_directions: list[np.ndarray],
    site_directions_by_gate: (
        dict[float, dict[str, list[np.ndarray]]] | None
    ) = None,
    n_segments: int = 4,
    site_n_frames: np.ndarray | None = None,
) -> dict:
    """Run all four tests and return per-test results + suite summary.

    Parameters
    ----------
    site_directions : per-site arrays for this config (Tests 2, 3, 4).
    site_directions_by_gate : multi-config gated input for Test 1.
    n_segments : temporal segments for Test 3.
    site_n_frames : per-site post-gating frame counts for Test 2.

    Returns
    -------
    dict
        Keys exactly: ``summary``, ``flag_reasons``, ``test_gating``,
        ``test_pooling``, ``test_temporal``, ``test_failure``.
    """
    # Test 1
    if site_directions_by_gate is not None:
        t1 = test_gating_sensitivity(site_directions_by_gate)
    else:
        t1 = {
            "status": "INSUFFICIENT",
            "pairwise_rho": {},
            "n_insufficient_pairs": 0,
        }

    # Test 2
    t2 = test_pooling_sensitivity(site_directions, site_n_frames)

    # Test 3
    t3 = test_temporal_stability(site_directions, n_segments)

    # Test 4 (default: no known_failures)
    this_result = {
        "summary": "PENDING",
        "flag_reasons": [],
        "test_gating": t1,
        "test_pooling": t2,
        "test_temporal": t3,
        "test_failure": {},
    }
    t4 = test_synthetic_failure_detection(
        {"this_config": this_result}, known_failures=None
    )

    # Suite summary
    statuses = [t1["status"], t2["status"], t3["status"], t4["status"]]
    flag_reasons: list[str] = []

    for name, sub in [
        ("test_gating", t1),
        ("test_pooling", t2),
        ("test_temporal", t3),
        ("test_failure", t4),
    ]:
        st = sub["status"]
        if st == "FAIL":
            detail = _detail_string(name, sub)
            flag_reasons.append(f"{name}: FAIL{detail}")
        elif st == "FLAG":
            detail = _detail_string(name, sub)
            flag_reasons.append(f"{name}: FLAG{detail}")
        elif st == "INSUFFICIENT":
            detail = _detail_string(name, sub)
            flag_reasons.append(f"{name}: INSUFFICIENT{detail}")

    if any(s == "FAIL" for s in statuses):
        summary = "FAIL"
    elif any(s == "FLAG" for s in statuses):
        summary = "FLAG"
    elif any(s == "INSUFFICIENT" for s in statuses):
        summary = "FLAG"
    else:
        summary = "PASS"

    return {
        "summary": summary,
        "flag_reasons": flag_reasons,
        "test_gating": t1,
        "test_pooling": t2,
        "test_temporal": t3,
        "test_failure": t4,
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _threshold_verdict(
    value: float, pass_threshold: float, flag_threshold: float
) -> str:
    """PASS / FLAG / FAIL based on two thresholds."""
    if value < pass_threshold:
        return "PASS"
    if value < flag_threshold:
        return "FLAG"
    return "FAIL"


def _detail_string(name: str, sub: dict) -> str:
    """Build human-readable detail for flag_reasons."""
    if "diff" in sub and np.isfinite(sub.get("diff", np.nan)):
        return f" (diff={sub['diff']:.4f})"
    if "mean_rel_range" in sub and np.isfinite(
        sub.get("mean_rel_range", np.nan)
    ):
        return f" (mean_rel_range={sub['mean_rel_range']:.4f})"
    if sub["status"] == "INSUFFICIENT":
        return " (no data provided)" if name != "test_failure" else ""
    return ""
