"""JSON I/O with NaN ↔ null conversion.

json_dump:         write dict to JSON file, converting np.nan → null.
json_load_bundled: load bundled package JSON, converting null → np.nan.
json_load:         load arbitrary JSON with selective null → NaN conversion.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import numpy as np


class _NanEncoder(json.JSONEncoder):
    """Encode np.nan as JSON null, numpy scalars as Python scalars."""

    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        return super().encode(_nan_to_none(obj))


def _nan_to_none(obj):
    """Recursively replace np.nan/float('nan') with None."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        val = obj.item()
        return None if np.isnan(val) else val
    return obj


def _none_to_nan(obj, numeric_keys: set[str] | None = None):
    """Recursively replace None with np.nan in numeric fields.

    Parameters
    ----------
    obj : parsed JSON object.
    numeric_keys : set of key names whose None values should become np.nan.
        If None, ALL None values become np.nan (blanket mode).
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if v is None:
                if numeric_keys is None or k in numeric_keys:
                    result[k] = np.nan
                else:
                    result[k] = v
            else:
                result[k] = _none_to_nan(v, numeric_keys)
        return result
    if isinstance(obj, list):
        return [_none_to_nan(v, numeric_keys) for v in obj]
    return obj


def json_dump(data: dict, path: str | Path, indent: int = 2) -> None:
    """Write dict to JSON, converting np.nan → null.

    Parameters
    ----------
    data : dict to serialise.
    path : output file path.
    indent : JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=_NanEncoder, indent=indent, allow_nan=False)
        f.write("\n")


def json_load_bundled(
    relative_path: str,
    package: str = "dirprobe.data",
) -> dict:
    """Load a JSON file bundled in the dirprobe package (null → np.nan blanket).

    Parameters
    ----------
    relative_path : path relative to the package data directory.
    package : package containing the data directory.

    Returns
    -------
    dict with all null values replaced by np.nan.
    """
    ref = importlib.resources.files(package).joinpath(relative_path)
    text = ref.read_text(encoding="utf-8")
    raw = json.loads(text)
    return _none_to_nan(raw, numeric_keys=None)


def json_load(
    path: str | Path,
    numeric_keys: set[str],
) -> dict:
    """Load JSON file with selective null → np.nan conversion.

    Parameters
    ----------
    path : file path.
    numeric_keys : set of key names whose None values become np.nan.
        Required (not optional, no default).

    Returns
    -------
    dict with specified null values replaced by np.nan.
    """
    with open(path) as f:
        raw = json.load(f)
    return _none_to_nan(raw, numeric_keys=numeric_keys)
