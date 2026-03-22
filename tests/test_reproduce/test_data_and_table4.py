"""Tests for bundled data verification, table4, and module entrypoint."""

import json
import subprocess
import sys

import numpy as np
import pytest

from dirprobe.data import BUNDLED_JSON_FILES, verify_bundled_data


# ── verify_bundled_data ──────────────────────────────────────────────

class TestVerifyBundledData:
    def test_returns_true(self):
        ok, details = verify_bundled_data()
        assert ok is True

    def test_details_has_all_6_files(self):
        _, details = verify_bundled_data()
        assert len(details) == 6
        for fname in BUNDLED_JSON_FILES:
            assert fname in details

    def test_details_structure(self):
        _, details = verify_bundled_data()
        for fname, d in details.items():
            assert "checksum_ok" in d
            assert "null_paths" in d
            assert d["checksum_ok"] is True
            assert isinstance(d["null_paths"], list)

    def test_idempotent(self):
        ok1, d1 = verify_bundled_data()
        ok2, d2 = verify_bundled_data()
        assert ok1 == ok2
        for fname in BUNDLED_JSON_FILES:
            assert d1[fname]["checksum_ok"] == d2[fname]["checksum_ok"]
            assert d1[fname]["null_paths"] == d2[fname]["null_paths"]


# ── table4 build function ───────────────────────────────────────────

class TestTable4Build:
    def test_returns_headers_and_rows(self):
        from dirprobe.reproduce.table4 import build_table4_rows
        headers, rows = build_table4_rows()
        assert isinstance(headers, list)
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_headers_contain_expected_columns(self):
        from dirprobe.reproduce.table4 import build_table4_rows
        headers, _ = build_table4_rows()
        assert "Config" in headers
        assert "Δ_coh" in headers
        assert "D_pooled" in headers
        assert "S_align" in headers
        assert "Robust?" in headers

    def test_10_knn_configs(self):
        from dirprobe.reproduce.table4 import build_table4_rows
        _, rows = build_table4_rows()
        assert len(rows) == 10

    def test_hero_pair_present(self):
        from dirprobe.reproduce.table4 import build_table4_rows
        _, rows = build_table4_rows()
        config_col = 0
        configs = [row[config_col] for row in rows]
        assert "6" in configs
        assert "8" in configs

    def test_hero_pair_delta_coh_values(self):
        """Config 6 Δ_coh ≈ 0.02, Config 8 Δ_coh ≈ 1.26."""
        from dirprobe.reproduce.table4 import build_table4_rows
        headers, rows = build_table4_rows()
        dcoh_idx = headers.index("Δ_coh")
        config_idx = headers.index("Config")
        for row in rows:
            if row[config_idx] == "6":
                val = float(row[dcoh_idx])
                assert abs(val - 0.02) < 0.05
            elif row[config_idx] == "8":
                val = float(row[dcoh_idx])
                assert abs(val - 1.26) < 0.05

    def test_sorted_by_delta_coh(self):
        from dirprobe.reproduce.table4 import build_table4_rows
        headers, rows = build_table4_rows()
        dcoh_idx = headers.index("Δ_coh")
        dcoh_vals = []
        for row in rows:
            try:
                dcoh_vals.append(float(row[dcoh_idx]))
            except ValueError:
                dcoh_vals.append(float("inf"))
        for i in range(len(dcoh_vals) - 1):
            assert dcoh_vals[i] <= dcoh_vals[i + 1]


# ── __main__ entrypoint ──────────────────────────────────────────────

class TestMainEntrypoint:
    def test_module_runs_without_crash(self):
        result = subprocess.run(
            [sys.executable, "-m", "dirprobe.reproduce"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ── io/ not imported in default mode ─────────────────────────────────

class TestLazyImport:
    def test_io_not_imported_at_module_level(self):
        import importlib
        # Ensure clean state
        mods_before = {m for m in sys.modules if "dirprobe.io" in m}
        importlib.import_module("dirprobe.reproduce.table4")
        mods_after = {m for m in sys.modules if "dirprobe.io" in m}
        new_mods = mods_after - mods_before
        assert len(new_mods) == 0, f"io/ imported at module level: {new_mods}"


# ── Float key handling ───────────────────────────────────────────────

class TestFloatKeyHandling:
    def test_gating_sensitivity_float_keys(self):
        """After loading gating_sensitivity.json, threshold keys become float."""
        from importlib.resources import files
        data_dir = files("dirprobe.data.knn")
        raw = json.loads(
            (data_dir / "gating_sensitivity.json").read_text(encoding="utf-8")
        )
        gates = raw.get("gates", [])
        # gates list should contain floats (JSON numbers)
        for g in gates:
            assert isinstance(g, (int, float))
            _ = float(g)  # must not raise

        # String keys in configs.*.gates can be converted to float
        configs = raw.get("configs", {})
        for config_key, cdata in configs.items():
            gate_keys = list(cdata.get("gates", {}).keys())
            for gk in gate_keys:
                _ = float(gk)  # must not raise
