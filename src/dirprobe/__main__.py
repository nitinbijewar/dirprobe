"""Entry point for `python -m dirprobe` and `dirprobe` console command."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dirprobe",
        description="Directional diagnostics for site-resolved trajectory data",
    )
    sub = parser.add_subparsers(dest="command")

    # reproduce subcommand
    rep = sub.add_parser("reproduce", help="Reproduce manuscript tables")
    rep.add_argument("--from-generators", action="store_true",
                     help="Tables 2/3 from CMS system generators")
    rep.add_argument("--from-displacements", nargs="?", const="BUNDLED",
                     default=None, metavar="DIR",
                     help="Table 4 from NPZ displacements")
    rep.add_argument("--ignore-data-integrity", action="store_true",
                     help="Continue if bundled data checksums fail")

    # synthetic subcommand
    syn = sub.add_parser("synthetic", help="Synthetic system tools")
    syn_group = syn.add_mutually_exclusive_group(required=True)
    syn_group.add_argument("--smoke-test", action="store_true")
    syn_group.add_argument("--save-to", metavar="DIR")
    syn_group.add_argument("--verify", metavar="DIR")

    args = parser.parse_args(argv)

    if args.command == "reproduce":
        from dirprobe.reproduce.__main__ import main as rep_main
        # Forward relevant args
        fwd = []
        if args.from_generators:
            fwd.append("--from-generators")
        if args.from_displacements is not None:
            fwd.append("--from-displacements")
            if args.from_displacements != "BUNDLED":
                fwd.append(args.from_displacements)
        if args.ignore_data_integrity:
            fwd.append("--ignore-data-integrity")
        rep_main(fwd)

    elif args.command == "synthetic":
        from dirprobe.synthetic.__main__ import main as syn_main
        fwd = []
        if args.smoke_test:
            fwd.append("--smoke-test")
        elif args.save_to:
            fwd.extend(["--save-to", args.save_to])
        elif args.verify:
            fwd.extend(["--verify", args.verify])
        syn_main(fwd)

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
