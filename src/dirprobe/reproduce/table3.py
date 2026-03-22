"""Table 3: Persistence calibration — A, B, classification.

Default mode reads frozen bundled manuscript data.
With --from-generators: runs CMS systems G, H, I through pipeline.
"""

from __future__ import annotations

import json
import sys
from importlib.resources import files

from dirprobe.reproduce.formatting import fmt, render_table


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def build_table3_rows() -> tuple[list[str], list[list[str]]]:
    """Build Table 3 from frozen bundled manuscript persistence data."""
    headers = ["System", "Control", "A_mean", "B_mean", "Classification"]

    data_dir = files("dirprobe.data.synthetic")
    raw = json.loads((data_dir / "cms_table3.json").read_text(encoding="utf-8"))

    rows = []
    for entry in raw:
        rows.append([
            entry.get("system", "?"),
            entry.get("control_parameter", ""),
            fmt(float(entry["A_mean"]), 6),
            fmt(float(entry["B_mean"]), 6),
            entry.get("classification", "?"),
        ])

    return headers, rows


def build_table3_rows_from_generators() -> tuple[list[str], list[list[str]]]:
    """Build Table 3 by running G, H, I through pipeline with W=4."""
    from dirprobe.pipeline import run_pipeline
    from dirprobe.synthetic.systems import generate_all_table3
    from dirprobe.time.persistence import classify_persistence

    headers = ["System", "Control", "A_mean", "B_mean", "Classification"]

    _log("Generating CMS Table 3 systems...")
    table3_rows = generate_all_table3()
    _log(f"  {len(table3_rows)} rows to compute")

    rows = []
    for i, row_data in enumerate(table3_rows):
        label = row_data["label"]
        ctrl = row_data["control_value"]
        sys_data = row_data["system_data"]

        _log(f"  Row {i+1}/{len(table3_rows)}: {label} (control={ctrl})")

        result = run_pipeline(
            sys_data["displacements"],
            gating_threshold=0.0,  # unit vectors
            persistence_windows=4,
        )

        cls = classify_persistence(result["a_mean"], result["b_mean"])

        ctrl_str = f"rate=0 (static)" if ctrl == 0.0 else str(ctrl)
        rows.append([
            label, ctrl_str,
            fmt(result["a_mean"], 6),
            fmt(result["b_mean"], 6),
            cls,
        ])

    _log("Pipeline complete.")
    return headers, rows


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Table 3")
    parser.add_argument("--from-generators", action="store_true",
                        help="Run CMS systems through pipeline")
    args = parser.parse_args(argv)

    if args.from_generators:
        headers, rows = build_table3_rows_from_generators()
        print("\nTable 3: Persistence Calibration (from generators)\n")
    else:
        headers, rows = build_table3_rows()
        print("\nTable 3: Persistence Calibration (manuscript)\n")

    print(render_table(headers, rows))
    print()


if __name__ == "__main__":
    main()
