"""Shared fixtures for cross-validation test suite.

Loads Class A (atomic) and Class B (manuscript) fixtures once per session.
All paths relative to the project root.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "cross_validation"


# ── helpers ──────────────────────────────────────────────────────────

def _load_npz(name: str) -> dict[str, np.ndarray]:
    data = np.load(FIXTURE_DIR / name, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _load_json(name: str):
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


# ── Class A: atomic fixtures ────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg6_site_tensors() -> np.ndarray:
    """(8, 3, 3) covariance tensors for config 6."""
    return _load_npz("cfg6_site_tensors.npz")["tensors"]


@pytest.fixture(scope="session")
def cfg6_principal_axes() -> np.ndarray:
    """(8, 3) principal axes for config 6. NaN rows = degenerate."""
    return _load_npz("cfg6_principal_axes.npz")["axes"]


@pytest.fixture(scope="session")
def cfg6_site0_vectors() -> np.ndarray:
    """(T, 3) gated unit vectors for config 6 site 0."""
    return _load_npz("cfg6_site0_unit_vectors.npz")["vectors"]


@pytest.fixture(scope="session")
def cfg8_site_tensors() -> np.ndarray:
    """(8, 3, 3) covariance tensors for config 8."""
    return _load_npz("cfg8_site_tensors.npz")["tensors"]


@pytest.fixture(scope="session")
def cfg8_raw_displacements() -> dict[str, np.ndarray]:
    """Per-site Cartesian displacements for config 8.
    Keys: 'site_0' .. 'site_7', each (T_total, 3).
    """
    return _load_npz("cfg8_raw_displacements.npz")


@pytest.fixture(scope="session")
def cfg8_persistence_data() -> dict:
    """Persistence fixture with resolved metadata.

    The file is named 'cfg8_persistence_W4.npz' but actually contains
    half-split (W=2) data from the reference implementation. The window_axes shape
    (8, 2, 3) confirms W=2.

    Returns dict with:
        window_axes: (8, 2, 3)
        full_axes: (8, 3)
        actual_n_windows: int (2)
        is_halfsplit: bool (True)
    """
    data = _load_npz("cfg8_persistence_W4.npz")
    n_windows = data["window_axes"].shape[1]
    return {
        "window_axes": data["window_axes"],
        "full_axes": data["full_axes"],
        "actual_n_windows": n_windows,
        "is_halfsplit": n_windows == 2,
    }


@pytest.fixture(scope="session")
def gating_masks() -> dict[str, np.ndarray]:
    """Boolean gating masks. Keys like 'delta_0.20_config_8_site_3'."""
    return _load_npz("gating_masks.npz")


@pytest.fixture(scope="session")
def degeneracy_flags() -> dict:
    """Degeneracy flags per config. Keys '1'..'10' -> list[bool],
    plus 'epsilon_used', 'total_sites', 'degenerate_count'.
    """
    return _load_json("degeneracy_flags.json")


# ── Class B: manuscript fixtures ────────────────────────────────────

@pytest.fixture(scope="session")
def expected_table_s3() -> list[dict]:
    """10 rows, one per KNN config. Keys: config, ordering, D_dir_site_mean, etc."""
    return _load_json("expected_table_s3.json")


@pytest.fixture(scope="session")
def expected_table2() -> list[dict]:
    """14 rows for synthetic validation. Keys: system, D_dir_expected, etc."""
    return _load_json("expected_table2.json")


@pytest.fixture(scope="session")
def expected_table3() -> list[dict]:
    """12 rows for persistence calibration. Keys: system, A_mean, B_mean, etc."""
    return _load_json("expected_table3.json")


@pytest.fixture(scope="session")
def expected_key_stats() -> dict:
    """Key statistics: spearman_rho_salign_dcoh, etc."""
    return _load_json("expected_key_statistics.json")


@pytest.fixture(scope="session")
def expected_hero_persist() -> dict:
    """Hero pair persistence: '6' -> {A_mean, B_mean}, '8' -> {A_mean, B_mean}."""
    return _load_json("expected_persistence_hero.json")


@pytest.fixture(scope="session")
def manifest() -> dict:
    """Fixture manifest with conventions and file lists."""
    return _load_json("fixture_manifest.json")
