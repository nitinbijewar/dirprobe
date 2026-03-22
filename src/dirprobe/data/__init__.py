"""Bundled data access and integrity verification."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

BUNDLED_JSON_FILES = [
    "gating_sensitivity.json",
    "hero_numbers.json",
    "mlff_stats.json",
    "persite_results.json",
    "secondary_results.json",
    "sector_persistence.json",
]


def verify_bundled_data() -> tuple[bool, dict]:
    """Verify integrity of the 6 bundled JSON files.

    Checks (in order):

    1. Checksum verification (HARD GATE):
       Hash each file against DATA_SHA256SUMS.txt.

    2. Null-location audit (ADVISORY):
       Traverse loaded JSON, report null paths.

    Returns
    -------
    ok : bool
        True iff ALL 6 checksums pass.
    details : dict
        Maps filename to ``{'checksum_ok': bool, 'null_paths': list[str]}``.
    """
    data_dir = files("dirprobe.data.knn")
    checksum_file = files("dirprobe.data") / "DATA_SHA256SUMS.txt"

    # Parse expected checksums
    expected: dict[str, str] = {}
    checksum_text = checksum_file.read_text(encoding="utf-8")
    for line in checksum_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            expected[parts[1]] = parts[0]

    all_ok = True
    details: dict[str, dict] = {}

    for fname in BUNDLED_JSON_FILES:
        ref = data_dir / fname
        content = ref.read_bytes()

        # Checksum
        actual_hash = hashlib.sha256(content).hexdigest()
        checksum_ok = expected.get(fname) == actual_hash
        if not checksum_ok:
            all_ok = False

        # Null audit
        data = json.loads(content)
        null_paths: list[str] = []
        _walk_nulls(data, "", null_paths)

        details[fname] = {
            "checksum_ok": checksum_ok,
            "null_paths": null_paths,
        }

    return all_ok, details


def _walk_nulls(obj: object, path: str, out: list[str]) -> None:
    """Recursively find null values and record their JSON paths."""
    if obj is None:
        out.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_nulls(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_nulls(v, f"{path}[{i}]", out)
