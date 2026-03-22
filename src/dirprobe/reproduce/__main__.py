"""Run all reproduction tables.

Run: python -m dirprobe.reproduce
Run: python -m dirprobe.reproduce --ignore-data-integrity
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Run data verification then all reproduction tables."""
    parser = argparse.ArgumentParser(
        description="Run all dirprobe reproduction tables",
    )
    parser.add_argument(
        "--ignore-data-integrity",
        action="store_true",
        help="Continue even if bundled data checksums fail",
    )
    parser.add_argument(
        "--from-displacements",
        nargs="?",
        const="BUNDLED",
        default=None,
        metavar="DIR",
        help="Run Table 4 from NPZ displacements (default: bundled)",
    )
    parser.add_argument(
        "--from-generators",
        action="store_true",
        help="Run Tables 2/3 from CMS system generators",
    )
    args = parser.parse_args(argv)

    # 1. Verify bundled data
    from dirprobe.data import verify_bundled_data

    ok, details = verify_bundled_data()
    if not ok:
        print("ERROR: Bundled data integrity check failed:", file=sys.stderr)
        for fname, d in details.items():
            if not d["checksum_ok"]:
                print(f"  {fname}: checksum FAILED", file=sys.stderr)
        if not args.ignore_data_integrity:
            sys.exit(1)
        print(
            "WARNING: Continuing with --ignore-data-integrity",
            file=sys.stderr,
        )

    # 2. Run tables (inline imports to prevent io/ trigger)
    from dirprobe.reproduce.table2 import main as table2_main

    t2_argv = ["--from-generators"] if args.from_generators else []
    table2_main(t2_argv)

    from dirprobe.reproduce.table3 import main as table3_main

    t3_argv = ["--from-generators"] if args.from_generators else []
    table3_main(t3_argv)

    from dirprobe.reproduce.table4 import main as table4_main

    t4_argv = []
    if args.from_displacements is not None:
        if args.from_displacements == "BUNDLED":
            t4_argv = ["--from-displacements"]
        else:
            t4_argv = ["--from-displacements", args.from_displacements]
    table4_main(t4_argv)


if __name__ == "__main__":
    main()
