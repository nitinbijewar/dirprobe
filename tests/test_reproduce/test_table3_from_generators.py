"""Regression test for --from-generators mode (Table 3)."""

import pytest
from dirprobe.reproduce.table3 import build_table3_rows_from_generators


@pytest.fixture(scope="module")
def gen_table3():
    return build_table3_rows_from_generators()


class TestTable3FromGenerators:
    def test_row_count(self, gen_table3):
        _, rows = gen_table3
        assert len(rows) == 12

    def test_system_G_locked(self, gen_table3):
        _, rows = gen_table3
        g_row = rows[0]  # G is first row
        assert "G" in g_row[0]
        assert g_row[4] == "LOCKED"

    def test_H_A_decreases_overall(self, gen_table3):
        """H: first rate should have higher A than last rate."""
        _, rows = gen_table3
        h_rows = [r for r in rows if r[0].startswith("H")]
        a_first = float(h_rows[0][2])
        a_last = float(h_rows[-1][2])
        assert a_first > a_last, (
            f"H A should decrease: first={a_first}, last={a_last}"
        )
