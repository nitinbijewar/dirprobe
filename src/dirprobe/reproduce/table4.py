"""Reproduce Table 4: KNN demonstration.

Default mode reads bundled JSON (frozen manuscript data).
With --from-displacements: runs full pipeline from bundled NPZ displacement data.
With --from-trajectories: reads XDATCAR files (requires dirprobe.io.vasp).
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files

import numpy as np

from dirprobe.reproduce.formatting import fmt, render_table, sort_key_delta_coh
from dirprobe.reproduce.jsonio import _none_to_nan


def _log(msg: str) -> None:
    """Print progress to stderr."""
    print(msg, file=sys.stderr)


# ── build rows from bundled JSON ─────────────────────────────────────


def build_table4_rows() -> tuple[list[str], list[list[str]]]:
    """Build Table 4 from bundled KNN JSON data.

    Returns (headers, rows) sorted by ascending Delta_coh.
    """
    headers = [
        "Config", "D_pooled", "<D_site>", "Δ_coh",
        "A", "B", "S_align", "Robust?",
    ]

    data_dir = files("dirprobe.data.knn")

    persite = _none_to_nan(
        json.loads((data_dir / "persite_results.json").read_text(encoding="utf-8")),
        numeric_keys=None,
    )
    secondary = _none_to_nan(
        json.loads((data_dir / "secondary_results.json").read_text(encoding="utf-8")),
        numeric_keys=None,
    )
    persistence = _none_to_nan(
        json.loads((data_dir / "sector_persistence.json").read_text(encoding="utf-8")),
        numeric_keys=None,
    )
    gating = _none_to_nan(
        json.loads((data_dir / "gating_sensitivity.json").read_text(encoding="utf-8")),
        numeric_keys=None,
    )

    s_align_map = secondary.get("S_align", {})

    persistence_a: dict[str, float] = {}
    persistence_b: dict[str, float] = {}
    for ckey, pdata in persistence.items():
        persistence_a[ckey] = pdata.get("mean_consecutive_alignment", np.nan)
        persistence_b[ckey] = pdata.get("mean_block_to_full", np.nan)

    robust_map = _compute_robustness_from_json(gating, persite, persistence)

    row_dicts: list[dict] = []
    for config_key, cdata in persite.items():
        config_num = config_key.replace("config_", "")
        row_dicts.append({
            "config": config_num,
            "d_pooled": cdata.get("D_pooled", np.nan),
            "d_site_mean": cdata.get("mean_ddir_site", np.nan),
            "delta_coh": cdata.get("Dcoh", np.nan),
            "a": persistence_a.get(config_key, np.nan),
            "b": persistence_b.get(config_key, np.nan),
            "s_align": s_align_map.get(config_key, np.nan),
            "robust": robust_map.get(config_key, "NA"),
        })

    row_dicts.sort(key=sort_key_delta_coh)

    rows = []
    for rd in row_dicts:
        rows.append([
            rd["config"],
            fmt(rd["d_pooled"]),
            fmt(rd["d_site_mean"]),
            fmt(rd["delta_coh"]),
            fmt(rd["a"]),
            fmt(rd["b"]),
            fmt(rd["s_align"]),
            rd["robust"],
        ])

    return headers, rows


# ── build rows from displacement NPZ ────────────────────────────────


def build_table4_rows_from_displacements(
    directory: str | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Build Table 4 by running the full pipeline on displacement NPZ data.

    Parameters
    ----------
    directory : path or None
        Directory containing config_XX.npz files. None = bundled data.

    Progress output goes to stderr; table data returned for stdout.
    """
    from dirprobe.gating.amplitude import apply_amplitude_gate
    from dirprobe.io.npz import load_all_configs
    from dirprobe.moments2.coherence import compute_s_align
    from dirprobe.moments2.pooling import pool_covariances
    from dirprobe.moments2.tensor import (
        compute_d_dir,
        compute_site_covariance,
        decompose_covariance,
    )
    from dirprobe.time.persistence import (
        compute_block_to_full_alignment,
        compute_windowed_alignment,
    )

    DELTA_0 = 0.20
    EPS_PERSIST = 0.01  # match reference implementation for persistence

    _log("Loading displacements...")
    configs = load_all_configs(directory)
    _log(f"  {len(configs)} configs loaded")

    headers = [
        "Config", "D_pooled", "<D_site>", "Δ_coh",
        "A", "B", "S_align", "Robust?",
    ]

    row_dicts: list[dict] = []
    total_raw = 0
    total_gated = 0

    for cfg_id in sorted(configs.keys()):
        cfg_data = configs[cfg_id]
        displacements = cfg_data["displacements"]
        ordering = cfg_data["ordering"]
        n_frames = displacements.shape[0]

        _log(f"  Config {cfg_id:2d} ({ordering}): {n_frames} frames")

        site_dirs: list[np.ndarray] = []
        site_covs: list[np.ndarray] = []
        n_gated_list: list[int] = []

        for site in range(8):
            site_disp = displacements[:, site, :]
            total_raw += len(site_disp)
            unit_vecs, mask = apply_amplitude_gate(
                site_disp, DELTA_0, return_mask=True
            )
            n_gated = int(mask.sum())
            n_gated_list.append(n_gated)
            total_gated += n_gated

            if n_gated < 3:
                site_dirs.append(np.zeros((0, 3)))
                continue

            site_dirs.append(unit_vecs)
            site_covs.append(compute_site_covariance(unit_vecs))

        # D_dir per site
        site_ddirs = []
        for C in site_covs:
            evals, _ = decompose_covariance(C)
            site_ddirs.append(compute_d_dir(evals))

        # Frame-weighted pooling (manuscript convention)
        weights = np.array(n_gated_list, dtype=float)
        C_pool = pool_covariances(site_covs, weights=weights)
        evals_pool, _ = decompose_covariance(C_pool)
        d_pool = compute_d_dir(evals_pool)
        d_site = float(np.mean(site_ddirs))
        dcoh = d_pool - d_site

        # S_align
        s_val, _ = compute_s_align(site_covs)

        # Persistence (hero pair only: configs 6 and 8)
        a_val = np.nan
        b_val = np.nan
        if cfg_id in (6, 8):
            site_A, site_B = [], []
            for dirs in site_dirs:
                if len(dirs) < 4:
                    continue
                T = len(dirs)
                ws = T // 2
                a_arr = compute_windowed_alignment(
                    dirs, window_size=ws, degeneracy_eps=EPS_PERSIST
                )
                if len(a_arr) > 0:
                    site_A.append(float(np.nanmean(a_arr)))
                b_arr = compute_block_to_full_alignment(
                    dirs, n_blocks=2, min_block_size=2,
                    degeneracy_eps=EPS_PERSIST,
                )
                if len(b_arr) > 0:
                    site_B.append(float(np.nanmean(b_arr)))
            if site_A:
                a_val = float(np.mean(site_A))
            if site_B:
                b_val = float(np.mean(site_B))

        row_dicts.append({
            "config": str(cfg_id),
            "d_pooled": d_pool,
            "d_site_mean": d_site,
            "delta_coh": dcoh,
            "a": a_val,
            "b": b_val,
            "s_align": s_val,
            "robust": "NA",  # robustness requires multi-gate data
        })

    _log(f"\nPipeline summary:")
    _log(f"  Gating: {total_raw} -> {total_gated} retained "
         f"({100 * total_gated / total_raw:.1f}%)")

    # Compare against frozen
    _log("\nComparing against manuscript:")
    data_dir = files("dirprobe.data.knn")
    frozen_ps = _none_to_nan(
        json.loads((data_dir / "persite_results.json").read_text(encoding="utf-8")),
        numeric_keys=None,
    )
    max_diff = {"D_site": 0, "D_pool": 0, "Dcoh": 0}
    for rd in row_dicts:
        fk = f"config_{rd['config']}"
        fp = frozen_ps.get(fk, {})
        for label, rd_key, fp_key in [
            ("D_site", "d_site_mean", "mean_ddir_site"),
            ("D_pool", "d_pooled", "D_pooled"),
            ("Dcoh", "delta_coh", "Dcoh"),
        ]:
            diff = abs(rd[rd_key] - fp.get(fp_key, rd[rd_key]))
            max_diff[label] = max(max_diff[label], diff)

    for label, md in max_diff.items():
        status = "✓" if md < 0.001 else "✗"
        _log(f"  {label:8s}: max |Δ| = {md:.4f}  {status}")

    sorted_cfgs = [rd["config"] for rd in sorted(row_dicts, key=sort_key_delta_coh)]
    _log(f"  Rank ordering: {sorted_cfgs}")

    row_dicts.sort(key=sort_key_delta_coh)

    rows = []
    for rd in row_dicts:
        rows.append([
            rd["config"],
            fmt(rd["d_pooled"]),
            fmt(rd["d_site_mean"]),
            fmt(rd["delta_coh"]),
            fmt(rd["a"]),
            fmt(rd["b"]),
            fmt(rd["s_align"]),
            rd["robust"],
        ])

    return headers, rows


# ── robustness from JSON ─────────────────────────────────────────────


def _compute_robustness_from_json(
    gating: dict, persite: dict, persistence: dict,
) -> dict[str, str]:
    """Compute robustness from pre-computed JSON (gating test only)."""
    from scipy import stats as sp_stats
    from dirprobe._constants import RHO_FLAG, RHO_PASS

    configs_data = gating.get("configs", {})
    gates = gating.get("gates", [])

    dcoh_by_threshold: dict[float, dict[str, float]] = {}
    for gate_val in gates:
        gate_key = str(gate_val)
        dcoh_by_threshold[gate_val] = {}
        for config_key, cdata in configs_data.items():
            gate_data = cdata.get("gates", {}).get(gate_key, {})
            dcoh_by_threshold[gate_val][config_key] = gate_data.get("Dcoh", np.nan)

    thresholds = sorted(dcoh_by_threshold.keys())
    min_rho = float("inf")
    has_numeric = False

    for i in range(len(thresholds)):
        for j in range(i + 1, len(thresholds)):
            ti, tj = thresholds[i], thresholds[j]
            common = set(dcoh_by_threshold[ti]) & set(dcoh_by_threshold[tj])
            vals_a, vals_b = [], []
            for cid in sorted(common):
                a, b = dcoh_by_threshold[ti][cid], dcoh_by_threshold[tj][cid]
                if np.isfinite(a) and np.isfinite(b):
                    vals_a.append(a)
                    vals_b.append(b)
            if len(vals_a) >= 3:
                arr_a, arr_b = np.array(vals_a), np.array(vals_b)
                if not (np.all(arr_a == arr_a[0]) or np.all(arr_b == arr_b[0])):
                    rho, _ = sp_stats.spearmanr(arr_a, arr_b)
                    min_rho = min(min_rho, rho)
                    has_numeric = True

    if has_numeric:
        t1 = "PASS" if min_rho >= RHO_PASS else ("FLAG" if min_rho >= RHO_FLAG else "FAIL")
    else:
        t1 = "INSUFFICIENT"

    statuses = [t1, "INSUFFICIENT", "INSUFFICIENT", "INSUFFICIENT"]
    if any(s == "FAIL" for s in statuses):
        summary = "FAIL"
    elif any(s in ("FLAG", "INSUFFICIENT") for s in statuses):
        summary = "FLAG"
    else:
        summary = "PASS"

    return {ck: summary for ck in persite}


# ── main ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce Table 4: KNN demonstration")
    parser.add_argument("--from-displacements", nargs="?", const="BUNDLED", default=None,
                        metavar="DIR", help="Run pipeline from NPZ displacements (default: bundled)")
    parser.add_argument("--from-trajectories", metavar="DIR", default=None,
                        help="Run from XDATCAR files (requires dirprobe.io.vasp)")
    args = parser.parse_args(argv)

    if args.from_trajectories is not None:
        raise NotImplementedError("--from-trajectories requires dirprobe.io.vasp")

    if args.from_displacements is not None:
        directory = None if args.from_displacements == "BUNDLED" else args.from_displacements
        headers, rows = build_table4_rows_from_displacements(directory)
    else:
        headers, rows = build_table4_rows()

    print("\nTable 4: KNN Demonstration\n")
    print(render_table(headers, rows))
    print()


if __name__ == "__main__":
    main()
