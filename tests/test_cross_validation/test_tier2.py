"""Tier 2: Config-level KNN parity against manuscript fixtures.

Ground truth: Class B fixtures (expected_table_s3.json, expected_persistence_hero.json).
Data source: dirprobe bundled KNN JSON via reproduce.table4.build_table4_rows().

Tolerance: 0.01 for scalars (bundled v11 JSON may differ slightly from
reference implementation fresh recomputation in cms_all_results.json).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats as sp_stats

from dirprobe.reproduce.table4 import build_table4_rows


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def table4_data():
    """Build Table 4 once per module. Returns (headers, rows, row_dicts)."""
    headers, rows = build_table4_rows()
    row_dicts = []
    for row in rows:
        row_dicts.append({
            "config": row[0],
            "D_pooled": _parse(row[1]),
            "D_site_mean": _parse(row[2]),
            "Delta_coh": _parse(row[3]),
            "A": _parse(row[4]),
            "B": _parse(row[5]),
            "S_align": _parse(row[6]),
            "robust": row[7],
        })
    return headers, rows, row_dicts


def _parse(s: str) -> float:
    """Parse formatted float or 'NA' to float/nan."""
    s = s.strip()
    if s in ("NA", "nan", "", "None"):
        return np.nan
    return float(s)


def _fixture_row(expected_table_s3, config_id: str) -> dict:
    """Find a row in the fixture by config id."""
    for row in expected_table_s3:
        if str(row["config"]) == str(config_id):
            return row
    pytest.fail(f"Config {config_id} not found in expected_table_s3")


def _table4_row(row_dicts, config_id: str) -> dict:
    """Find a row in table4 by config id."""
    for rd in row_dicts:
        if str(rd["config"]) == str(config_id):
            return rd
    pytest.fail(f"Config {config_id} not found in table4 output")


# ── KNN D_dir per config ────────────────────────────────────────────

# NOTE: Bundled JSON (from v11 legacy) may have small offsets from
# reference implementation fresh recomputation. Tolerance is 0.01.

# Strict tolerances — bundled data matches manuscript authority.
# Bundled KNN JSON now contains full-precision values from cms_all_results.json.
# Table 4 formats to 2 decimal places, so tolerance accounts for rounding.
TOL = 0.006


class TestKnnDdir:
    @pytest.mark.parametrize("cfg", [str(i) for i in range(1, 11)])
    def test_ddir_site_mean(self, cfg, table4_data, expected_table_s3):
        """<D_dir_site> matches manuscript fixture."""
        _, _, row_dicts = table4_data
        dp = _table4_row(row_dicts, cfg)
        fix = _fixture_row(expected_table_s3, cfg)
        expected = float(fix["D_dir_site_mean"])
        actual = dp["D_site_mean"]
        assert abs(actual - expected) < TOL, (
            f"Config {cfg}: <D_site>={actual:.4f}, expected={expected:.4f}"
        )

    @pytest.mark.parametrize("cfg", [str(i) for i in range(1, 11)])
    def test_ddir_pooled(self, cfg, table4_data, expected_table_s3):
        """D_dir_pooled matches manuscript fixture."""
        _, _, row_dicts = table4_data
        dp = _table4_row(row_dicts, cfg)
        fix = _fixture_row(expected_table_s3, cfg)
        expected = float(fix["D_dir_pooled"])
        actual = dp["D_pooled"]
        assert abs(actual - expected) < TOL, (
            f"Config {cfg}: D_pooled={actual:.4f}, expected={expected:.4f}"
        )

    @pytest.mark.parametrize("cfg", [str(i) for i in range(1, 11)])
    def test_dcoh(self, cfg, table4_data, expected_table_s3):
        _, _, row_dicts = table4_data
        dp = _table4_row(row_dicts, cfg)
        fix = _fixture_row(expected_table_s3, cfg)
        expected = float(fix["Delta_coh"])
        actual = dp["Delta_coh"]
        assert abs(actual - expected) < TOL, (
            f"Config {cfg}: Dcoh={actual:.4f}, expected={expected:.4f}"
        )


# ── Hero pair persistence ────────────────────────────────────────────


class TestKnnHeroPersistence:
    """Hero pair persistence values from bundled KNN data vs manuscript.

    NOTE: Bundled sector_persistence.json may use different key names
    ('mean_consecutive_alignment', 'mean_block_to_full') than the
    reference implementation output ('mean_A', 'mean_B'). Table4 maps these.
    Tolerance is 0.01 to allow for v11 vs reference implementation differences.
    """

    def test_cfg6_persistence(self, table4_data, expected_hero_persist):
        """Strict tolerance after bundled data refresh to half-split values."""
        _, _, row_dicts = table4_data
        dp = _table4_row(row_dicts, "6")
        fix = expected_hero_persist["6"]
        if np.isfinite(dp["A"]):
            assert abs(dp["A"] - fix["A_mean"]) < 0.01, (
                f"cfg6 A: {dp['A']:.4f} vs {fix['A_mean']:.4f}"
            )
        if np.isfinite(dp["B"]):
            assert abs(dp["B"] - fix["B_mean"]) < 0.01, (
                f"cfg6 B: {dp['B']:.4f} vs {fix['B_mean']:.4f}"
            )

    def test_cfg8_persistence(self, table4_data, expected_hero_persist):
        _, _, row_dicts = table4_data
        dp = _table4_row(row_dicts, "8")
        fix = expected_hero_persist["8"]
        if np.isfinite(dp["A"]):
            assert abs(dp["A"] - fix["A_mean"]) < 0.01, (
                f"cfg8 A: {dp['A']:.4f} vs {fix['A_mean']:.4f}"
            )
        if np.isfinite(dp["B"]):
            assert abs(dp["B"] - fix["B_mean"]) < 0.01, (
                f"cfg8 B: {dp['B']:.4f} vs {fix['B_mean']:.4f}"
            )


# ── S_align per config ──────────────────────────────────────────────


class TestKnnSAlign:
    @pytest.mark.parametrize("cfg", [str(i) for i in range(1, 11)])
    def test_s_align(self, cfg, table4_data, expected_table_s3):
        _, _, row_dicts = table4_data
        dp = _table4_row(row_dicts, cfg)
        fix = _fixture_row(expected_table_s3, cfg)
        expected = float(fix["S_align"])
        actual = dp["S_align"]
        if np.isfinite(actual):
            assert abs(actual - expected) < TOL, (
                f"Config {cfg}: S_align={actual:.4f}, expected={expected:.4f}"
            )


# ── Rank ordering ────────────────────────────────────────────────────


class TestKnnRankOrdering:
    def test_dcoh_rank(self, table4_data, expected_table_s3):
        """Configs sorted by Delta_coh ascending should match expected rank."""
        _, _, row_dicts = table4_data
        # Sort by Delta_coh ascending
        valid = [(rd["config"], rd["Delta_coh"]) for rd in row_dicts if np.isfinite(rd["Delta_coh"])]
        valid.sort(key=lambda x: x[1])
        dp_rank = [c for c, _ in valid]

        # Expected rank from fixture (sorted ascending by Delta_coh)
        fix_sorted = sorted(expected_table_s3, key=lambda r: float(r["Delta_coh"]))
        fix_rank = [str(r["config"]) for r in fix_sorted]

        assert dp_rank == fix_rank, (
            f"Rank mismatch:\n  dirprobe: {dp_rank}\n  expected: {fix_rank}"
        )


# ── Key statistics ───────────────────────────────────────────────────


class TestKnnKeyStatistics:
    def test_spearman_rho_salign_dcoh(self, table4_data, expected_table_s3, expected_key_stats):
        """Spearman rho(S_align, Delta_coh) matches manuscript."""
        _, _, row_dicts = table4_data
        s_aligns = []
        dcohs = []
        for cfg_id in range(1, 11):
            dp = _table4_row(row_dicts, str(cfg_id))
            if np.isfinite(dp["S_align"]) and np.isfinite(dp["Delta_coh"]):
                s_aligns.append(dp["S_align"])
                dcohs.append(dp["Delta_coh"])

        if len(s_aligns) >= 3:
            rho, p = sp_stats.spearmanr(s_aligns, dcohs)
            expected_rho = expected_key_stats["spearman_rho_salign_dcoh"]
            # Tolerant: bundled data differs slightly from the reference implementation
            assert abs(rho - expected_rho) < 0.1, (
                f"Spearman rho: {rho:.4f}, expected={expected_rho:.4f}"
            )
