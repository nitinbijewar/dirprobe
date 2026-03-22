"""Regression test for --from-displacements pipeline.

Verifies that the displacement-based pipeline matches frozen manuscript data.
"""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest

from dirprobe.reproduce.jsonio import _none_to_nan
from dirprobe.reproduce.table4 import build_table4_rows_from_displacements


@pytest.fixture(scope="module")
def disp_table():
    """Build Table 4 from displacements once per module."""
    headers, rows = build_table4_rows_from_displacements()
    row_dicts = {}
    for row in rows:
        row_dicts[row[0].strip()] = {
            "D_pooled": _parse(row[1]),
            "D_site": _parse(row[2]),
            "Delta_coh": _parse(row[3]),
            "A": _parse(row[4]),
            "B": _parse(row[5]),
            "S_align": _parse(row[6]),
        }
    return row_dicts


def _parse(s: str) -> float:
    s = s.strip()
    return np.nan if s in ("NA", "nan", "") else float(s)


@pytest.fixture(scope="module")
def frozen_ps():
    data_dir = files("dirprobe.data.knn")
    return _none_to_nan(
        json.loads((data_dir / "persite_results.json").read_text(encoding="utf-8")),
        numeric_keys=None,
    )


class TestFromDisplacements:
    def test_row_count(self, disp_table):
        assert len(disp_table) == 10

    def test_rank_ordering(self, disp_table):
        sorted_cfgs = sorted(disp_table.items(), key=lambda kv: kv[1]["Delta_coh"])
        rank = [c for c, _ in sorted_cfgs]
        assert rank == ["6", "4", "2", "10", "3", "5", "7", "1", "9", "8"]

    @pytest.mark.parametrize("cfg", [str(i) for i in range(1, 11)])
    def test_ddir_site(self, cfg, disp_table, frozen_ps):
        actual = disp_table[cfg]["D_site"]
        expected = frozen_ps[f"config_{cfg}"]["mean_ddir_site"]
        assert abs(actual - expected) < 0.006

    @pytest.mark.parametrize("cfg", [str(i) for i in range(1, 11)])
    def test_dcoh(self, cfg, disp_table, frozen_ps):
        actual = disp_table[cfg]["Delta_coh"]
        expected = frozen_ps[f"config_{cfg}"]["Dcoh"]
        assert abs(actual - expected) < 0.006

    def test_hero_persistence_cfg6(self, disp_table):
        a = disp_table["6"]["A"]
        b = disp_table["6"]["B"]
        assert np.isfinite(a) and abs(a - 0.87) < 0.02
        assert np.isfinite(b) and abs(b - 0.96) < 0.02

    def test_hero_persistence_cfg8(self, disp_table):
        a = disp_table["8"]["A"]
        b = disp_table["8"]["B"]
        assert np.isfinite(a) and abs(a - 0.59) < 0.02
        assert np.isfinite(b) and abs(b - 0.84) < 0.02
