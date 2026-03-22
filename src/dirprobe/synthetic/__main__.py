"""CLI for synthetic system generation and verification.

Usage:
    python -m dirprobe.synthetic --smoke-test
    python -m dirprobe.synthetic --save-to <dir>
    python -m dirprobe.synthetic --verify <dir>
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic system tools")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true",
                       help="Quick sanity check on systems A, B, D")
    group.add_argument("--save-to", metavar="DIR",
                       help="Generate all systems and save bundles")
    group.add_argument("--verify", metavar="DIR",
                       help="Verify bundles against pipeline")
    args = parser.parse_args(argv)

    if args.smoke_test:
        _smoke_test()
    elif args.save_to:
        _save_all(args.save_to)
    elif args.verify:
        _verify_all(args.verify)


def _smoke_test() -> None:
    from dirprobe.pipeline import run_pipeline
    from dirprobe.synthetic.systems import system_A, system_B, system_D

    _log("Smoke test: generating A, B, D...")

    a = system_A(T=1000)
    r = run_pipeline(a["displacements"], gating_threshold=0.0)
    d_dir_a = r["d_dir_site_mean"]
    _log(f"  System A (isotropic): D_dir = {d_dir_a:.2f} (expect ~3.0)")
    assert abs(d_dir_a - 3.0) < 0.15, f"A failed: {d_dir_a}"

    b = system_B(T=1000)
    r = run_pipeline(b["displacements"], gating_threshold=0.0)
    d_dir_b = r["d_dir_site_mean"]
    _log(f"  System B (uniaxial):  D_dir = {d_dir_b:.2f} (expect ~1.08)")
    assert abs(d_dir_b - 1.08) < 0.15, f"B failed: {d_dir_b}"

    d = system_D(T=1000)
    r = run_pipeline(d["displacements"], gating_threshold=0.0)
    dcoh = r["delta_coh"]
    _log(f"  System D (incoherent): Delta_coh = {dcoh:.2f} (expect > 1.5)")
    assert dcoh > 1.0, f"D failed: {dcoh}"

    print("Smoke test PASSED")


def _save_all(directory: str) -> None:
    from dirprobe.pipeline import run_pipeline
    from dirprobe.synthetic.bundle import save_bundle
    from dirprobe.synthetic.systems import ALL_CMS_SYSTEMS

    _log(f"Saving all 14 systems to {directory}...")
    for label, fn in ALL_CMS_SYSTEMS.items():
        _log(f"  {label}...")
        data = fn()
        save_bundle(data, directory, pipeline_fn=lambda d: run_pipeline(d, gating_threshold=0.0))
    print(f"Saved 14 systems to {directory}")


def _verify_all(directory: str) -> None:
    from dirprobe.pipeline import run_pipeline
    from dirprobe.synthetic.bundle import verify_bundle
    from dirprobe.synthetic.systems import ALL_CMS_SYSTEMS

    _log(f"Verifying bundles in {directory}...")
    all_ok = True
    for label in ALL_CMS_SYSTEMS:
        ok, checks = verify_bundle(
            directory, label,
            pipeline_fn=lambda d: run_pipeline(d, gating_threshold=0.0),
        )
        status = "PASS" if ok else "FAIL"
        _log(f"  {label}: {status}")
        if not ok:
            all_ok = False
            for k, v in checks.items():
                if v.get("status") == "FAIL":
                    _log(f"    {k}: actual={v['actual']:.4f} expected={v['expected']:.4f}")

    if all_ok:
        print("All verifications PASSED")
    else:
        print("Some verifications FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
