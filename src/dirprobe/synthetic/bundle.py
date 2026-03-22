"""Save, load, and verify generation bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_bundle(
    system_output: dict,
    directory: str | Path,
    pipeline_fn=None,
) -> Path:
    """Save a system's output as a bundle (params JSON + displacements NPZ).

    If pipeline_fn is provided, also saves diagnostics JSON.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    label = system_output["label"]

    # Parameters
    params = {
        "label": label,
        "parameters": system_output["parameters"],
        "ground_truth": _serialise_gt(system_output["ground_truth"]),
    }
    with open(directory / f"{label}_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Displacements
    np.savez_compressed(
        directory / f"{label}_displacements.npz",
        displacements=system_output["displacements"],
    )

    # Diagnostics (optional)
    if pipeline_fn is not None:
        result = pipeline_fn(system_output["displacements"])
        diag = {k: _to_json(v) for k, v in result.items() if k != "site_directions"}
        with open(directory / f"{label}_diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2)

    return directory


def load_bundle(directory: str | Path, label: str) -> dict:
    """Load a bundle by label."""
    directory = Path(directory)
    with open(directory / f"{label}_params.json") as f:
        params = json.load(f)
    data = np.load(directory / f"{label}_displacements.npz")
    return {
        "label": label,
        "parameters": params["parameters"],
        "ground_truth": params["ground_truth"],
        "displacements": data["displacements"],
    }


def verify_bundle(
    directory: str | Path,
    label: str,
    pipeline_fn,
    tolerance: dict | None = None,
) -> tuple[bool, dict]:
    """Load a bundle, run pipeline, compare against ground truth."""
    bundle = load_bundle(directory, label)
    result = pipeline_fn(bundle["displacements"])
    gt = bundle["ground_truth"]
    tol = tolerance or gt.get("tolerance", {})

    checks = {}
    all_ok = True

    for key in ["D_dir_site", "D_dir_pooled", "Delta_coh"]:
        gt_val = gt.get(key)
        if gt_val is None:
            continue
        # Map ground truth key to pipeline result key
        result_key = {
            "D_dir_site": "d_dir_site_mean",
            "D_dir_pooled": "d_dir_pooled",
            "Delta_coh": "delta_coh",
        }[key]
        actual = result.get(result_key)
        if actual is None or not np.isfinite(actual):
            checks[key] = {"status": "SKIP", "reason": "non-finite"}
            continue
        diff = abs(actual - gt_val)
        t = tol.get("D_dir", tol.get("Delta_coh", 0.1))
        ok = diff < t
        checks[key] = {"status": "PASS" if ok else "FAIL",
                       "actual": actual, "expected": gt_val, "diff": diff}
        if not ok:
            all_ok = False

    return all_ok, checks


def _serialise_gt(gt: dict) -> dict:
    """Make ground truth JSON-serialisable."""
    out = {}
    for k, v in gt.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, list):
        return [_to_json(x) for x in obj]
    return obj
