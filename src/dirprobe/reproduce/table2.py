"""Table 2: Synthetic validation — D_dir, Delta_coh, robustness summary.

Default mode reads frozen bundled manuscript data.
With --from-generators: runs CMS systems A-N through pipeline.
"""

from __future__ import annotations

import json
import sys
from importlib.resources import files

from dirprobe.reproduce.formatting import fmt, render_table, sort_key_delta_coh


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def build_table2_rows() -> tuple[list[str], list[list[str]]]:
    """Build Table 2 from frozen bundled manuscript results."""
    headers = [
        "System", "Description",
        "D_dir", "D_dir_pooled", "Δ_coh", "Robust",
    ]

    data_dir = files("dirprobe.data.synthetic")
    raw = json.loads((data_dir / "cms_table2.json").read_text(encoding="utf-8"))

    row_dicts: list[dict] = []
    for entry in raw:
        row_dicts.append({
            "system": entry["system"],
            "description": entry.get("description", ""),
            "d_dir": float(entry.get("D_dir_measured", entry.get("D_dir", 0))),
            "d_dir_pooled": float(entry.get("D_dir_pooled", 0)),
            "delta_coh": float(entry.get("Delta_coh", 0)),
            "robust": entry.get("robust", "NA"),
        })

    row_dicts.sort(key=sort_key_delta_coh)

    rows = []
    for rd in row_dicts:
        rows.append([
            rd["system"], rd["description"],
            fmt(rd["d_dir"], 4), fmt(rd["d_dir_pooled"], 4),
            fmt(rd["delta_coh"], 6), rd["robust"],
        ])

    return headers, rows


_DESCRIPTIONS = {
    "A": "Isotropic (uniform on S²)",
    "B": "Uniaxial (vMF, κ=50, aligned)",
    "C": "Planar (biaxial, same plane)",
    "D": "Incoherent (cube vertices, κ=50)",
    "E": "Interpolation (tunable κ_inter)",
    "F": "Mixed κ (20/50/100)",
    "G": "Static locked axes (rate=0)",
    "H": "Axis switching (rate sweep)",
    "I": "Axis drift (continuous)",
    "J": "Mixed confinement (per-site)",
    "K": "Near-isotropic",
    "L": "Finite-size (N=4-64)",
    "M": "Gating-sensitive (low amp.)",
    "N": "Temporally unstable (high CV)",
}


def build_table2_rows_from_generators() -> tuple[list[str], list[list[str]]]:
    """Build Table 2 by running CMS systems A-N through the pipeline.

    Ground truth comparison is against analytic/convention expectations,
    NOT frozen JSON.
    """
    from dirprobe.pipeline import run_pipeline
    from dirprobe.synthetic.systems import generate_all_table2

    headers = [
        "System", "Description",
        "D_dir", "D_dir_pooled", "Δ_coh", "Robust",
    ]

    _log("Generating CMS manuscript systems A-N...")
    all_systems = generate_all_table2()
    _log(f"  {len(all_systems)} systems generated")

    row_dicts: list[dict] = []

    for i, (label, sys_data) in enumerate(sorted(all_systems.items())):
        _log(f"  System {label}: running pipeline [{i+1}/14]")

        # Use gating_threshold=0.0 for unit-vector systems (no amplitude)
        # except M which has physical amplitudes
        gt = sys_data.get("gating_threshold", 0.0)
        if label == "M":
            gt = 0.05

        result = run_pipeline(sys_data["displacements"], gating_threshold=gt)

        row_dicts.append({
            "system": label,
            "description": _DESCRIPTIONS.get(label, ""),
            "d_dir": result["d_dir_site_mean"],
            "d_dir_pooled": result["d_dir_pooled"],
            "delta_coh": result["delta_coh"],
            "robust": result["robustness"],
        })

    _log("\nPipeline complete.")

    row_dicts.sort(key=sort_key_delta_coh)

    rows = []
    for rd in row_dicts:
        rows.append([
            rd["system"], str(rd["description"]),
            fmt(rd["d_dir"], 4), fmt(rd["d_dir_pooled"], 4),
            fmt(rd["delta_coh"], 6), rd["robust"],
        ])

    return headers, rows


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Table 2")
    parser.add_argument("--from-generators", action="store_true",
                        help="Run CMS systems through pipeline")
    args = parser.parse_args(argv)

    if args.from_generators:
        headers, rows = build_table2_rows_from_generators()
        print("\nTable 2: Synthetic Validation (from generators)\n")
    else:
        headers, rows = build_table2_rows()
        print("\nTable 2: Synthetic Validation (manuscript)\n")

    print(render_table(headers, rows))
    print()


if __name__ == "__main__":
    main()
