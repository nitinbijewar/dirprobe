"""Load displacement data from bundled NPZ files."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np


def load_displacements(path: str | Path) -> dict:
    """Load one config NPZ file.

    Returns dict with keys: displacements (N,8,3), config_id (int),
    ordering (str), temperature_range (2,).
    """
    data = np.load(path, allow_pickle=True)
    return {
        "displacements": data["displacements"],
        "config_id": int(data["config_id"]),
        "ordering": str(data["ordering"]),
        "temperature_range": data["temperature_range"],
    }


def load_all_configs(directory: str | Path | None = None) -> dict[int, dict]:
    """Load all config_XX.npz files.

    Parameters
    ----------
    directory : path or None
        If None, uses bundled data under dirprobe.data.knn.displacements.

    Returns dict mapping config_id -> displacement dict.
    """
    if directory is None:
        data_dir = files("dirprobe.data.knn") / "displacements"
    else:
        data_dir = Path(directory)

    configs = {}
    for cfg_id in range(1, 11):
        fname = f"config_{cfg_id:02d}.npz"
        if directory is None:
            path = data_dir / fname
            # importlib.resources Traversable → read via joinpath
            content = path.read_bytes()
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
            tmp.write(content)
            tmp.close()
            try:
                configs[cfg_id] = load_displacements(tmp.name)
            finally:
                os.unlink(tmp.name)
        else:
            fpath = data_dir / fname
            if fpath.exists():
                configs[cfg_id] = load_displacements(fpath)

    return configs
