"""Table formatting utilities for manuscript reproduction.

fmt:               scalar → display string with configurable precision.
sort_key_delta_coh: row sort key for Table 2 (ascending delta_coh).
render_table:      list-of-lists → aligned plain-text table.
"""

from __future__ import annotations

import numpy as np


def fmt(value: float | None, decimals: int = 2) -> str:
    """Format a scalar for table display.

    Parameters
    ----------
    value : float, None, or np.nan.
    decimals : int, decimal places (default 2).

    Returns
    -------
    str
        Formatted number, or ``"NA"`` for None/NaN.
    """
    if value is None:
        return "NA"
    try:
        if np.isnan(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    return f"{value:.{decimals}f}"


def sort_key_delta_coh(row: dict) -> float:
    """Sort key for Table 2 rows: ascending delta_coh.

    Parameters
    ----------
    row : dict with key ``delta_coh``.

    Returns
    -------
    float
        delta_coh value, or inf for missing/NaN (sorts last).
    """
    val = row.get("delta_coh")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("inf")
    return val


def render_table(
    headers: list[str],
    rows: list[list[str]],
    pad: int = 2,
) -> str:
    """Render a plain-text aligned table.

    Parameters
    ----------
    headers : column header strings.
    rows : list of lists of cell strings, same length as headers.
    pad : int, minimum spaces between columns.

    Returns
    -------
    str
        Multi-line table string.
    """
    n_cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for j in range(min(len(row), n_cols)):
            widths[j] = max(widths[j], len(row[j]))

    def _fmt_row(cells: list[str]) -> str:
        parts = []
        for j in range(n_cols):
            cell = cells[j] if j < len(cells) else ""
            parts.append(cell.ljust(widths[j]))
        return (" " * pad).join(parts)

    lines = [_fmt_row(headers)]
    lines.append((" " * pad).join("-" * w for w in widths))
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)
