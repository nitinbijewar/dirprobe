"""Regression test for --from-generators mode (Table 2)."""

import pytest
from dirprobe.reproduce.table2 import build_table2_rows_from_generators


@pytest.fixture(scope="module")
def gen_table2():
    return build_table2_rows_from_generators()


class TestTable2FromGenerators:
    def test_row_count(self, gen_table2):
        _, rows = gen_table2
        assert len(rows) == 14

    def test_labels_A_through_N(self, gen_table2):
        _, rows = gen_table2
        labels = {r[0] for r in rows}
        assert labels == set("ABCDEFGHIJKLMN")

    def test_system_A_ddir_near_3(self, gen_table2):
        _, rows = gen_table2
        a_row = next(r for r in rows if r[0] == "A")
        d_dir = float(a_row[2])
        assert abs(d_dir - 3.0) < 0.15

    def test_system_B_ddir_near_1(self, gen_table2):
        _, rows = gen_table2
        b_row = next(r for r in rows if r[0] == "B")
        d_dir = float(b_row[2])
        assert abs(d_dir - 1.08) < 0.15
