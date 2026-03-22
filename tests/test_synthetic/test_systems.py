"""Tests for CMS manuscript synthetic systems A-N."""

import numpy as np
import pytest

from dirprobe.synthetic.systems import (
    ALL_CMS_SYSTEMS,
    system_A, system_B, system_D, system_G,
    generate_all_table2, generate_all_table3,
)
from dirprobe.pipeline import run_pipeline


class TestSystemContract:
    """Every system returns correct dict structure."""

    @pytest.mark.parametrize("label", list(ALL_CMS_SYSTEMS.keys()))
    def test_return_keys(self, label):
        fn = ALL_CMS_SYSTEMS[label]
        result = fn()
        assert "displacements" in result
        assert "parameters" in result
        assert "ground_truth" in result
        assert "label" in result
        assert result["label"] == label

    @pytest.mark.parametrize("label", list(ALL_CMS_SYSTEMS.keys()))
    def test_displacement_shape(self, label):
        fn = ALL_CMS_SYSTEMS[label]
        result = fn()
        d = result["displacements"]
        assert d.ndim == 3
        assert d.shape[2] == 3  # 3D

    def test_all_14_systems(self):
        assert len(ALL_CMS_SYSTEMS) == 14


class TestSanityChecks:
    def test_system_A_isotropic(self):
        a = system_A(T=2000)
        r = run_pipeline(a["displacements"], gating_threshold=0.0)
        assert abs(r["d_dir_site_mean"] - 3.0) < 0.15

    def test_system_B_uniaxial(self):
        b = system_B(T=2000)
        r = run_pipeline(b["displacements"], gating_threshold=0.0)
        assert abs(r["d_dir_site_mean"] - 1.08) < 0.15

    def test_system_D_incoherent(self):
        d = system_D(T=2000)
        r = run_pipeline(d["displacements"], gating_threshold=0.0)
        assert r["delta_coh"] > 1.5

    def test_system_G_locked(self):
        g = system_G(T=2000)
        r = run_pipeline(g["displacements"], gating_threshold=0.0, persistence_windows=4)
        assert r["a_mean"] > 0.9
        assert r["classification"] == "LOCKED"

    def test_determinism(self):
        a1 = system_A(T=100)
        a2 = system_A(T=100)
        np.testing.assert_array_equal(a1["displacements"], a2["displacements"])


class TestBatchGenerators:
    def test_table2_count(self):
        results = generate_all_table2()
        assert len(results) == 14
        assert set(results.keys()) == set("ABCDEFGHIJKLMN")

    def test_table3_count(self):
        rows = generate_all_table3()
        assert len(rows) == 12  # 1G + 6H + 5I
