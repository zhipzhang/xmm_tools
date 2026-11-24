"""Parser for XMM-Newton / HEASARC QDP (Quick and Dandy Plotter) files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from astropy.table import Table
import numpy as np


_LABEL_POS_RE = re.compile(
    r"^la(?:b)?\s+(\d+)\s+p\s+([+-]?\S+)\s+([+-]?\S+)\s+(.*)$",
    re.IGNORECASE,
)


@dataclass
class QDPLabel:
    """An in-plot text annotation parsed from ``la <id> p x y ...``."""

    label_id: int
    x: float
    y: float
    text: str
    justify: str = "center"


@dataclass
class QDPSeries:
    """A single Y-vector (and optional errors) paired with its X-vector."""

    vector_index: int  # 1-based vector index (2 = first Y vector)
    y_index: int  # 1-based Y-vector index (1 = first Y vector)
    name: str
    x: np.ndarray
    y: np.ndarray
    yerr_low: np.ndarray | None = None
    yerr_high: np.ndarray | None = None
    xerr_low: np.ndarray | None = None
    xerr_high: np.ndarray | None = None
    step: bool = False
    visible: bool = True
    color_index: int | None = None


@dataclass
class QDPWindow:
    """A plot window defined by ``win <n>`` and ``yplot ...`` commands."""

    window_id: int
    title: str = ""
    xlabel: str = "X"
    ylabel: str = "Y"
    log_x: bool = False
    log_y: bool = False
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    y_vectors: list[int] = field(default_factory=list)  # 1-based Y-vector indices
    labels: list[QDPLabel] = field(default_factory=list)


@dataclass
class QDPData:
    """Parsed contents of a QDP file."""

    path: Path | None
    raw_tables: list[np.ndarray]
    series: list[QDPSeries]
    windows: list[QDPWindow]
    commands: list[str]
    serr_vectors: set[int] = field(default_factory=set)
    terr_vectors: set[int] = field(default_factory=set)

    def to_astropy_tables(self) -> list[Table]:
        """Convert parsed raw numeric tables to ``astropy.table.Table`` objects."""
        tables: list[Table] = []
        for arr in self.raw_tables:
            if arr.size == 0:
                tables.append(Table())
                continue
            ncols = arr.shape[1]
            names = [f"col{i + 1}" for i in range(ncols)]
            tables.append(Table(arr, names=names, meta={"commands": self.commands}))
        return tables


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_qdp_float(token: str) -> float:
    if token.upper() == "NO":
        return float("nan")
    return float(token)


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return text.strip()


def _parse_command(
    line: str,
    state: dict[str, Any],
) -> None:
    """Update parser state from a single QDP/PLT command line."""
    tokens = line.split()
    if not tokens:
        return

    cmd = tokens[0].lower()

    # read serr / read terr
    if cmd == "read" and len(tokens) >= 3:
        sub = tokens[1].lower()
        for tok in tokens[2:]:
            if tok.isdigit():
                vec = int(tok)
                if sub == "serr":
                    state["serr"].add(vec)
                elif sub == "terr":
                    state["terr"].add(vec)
        return

    # win <id>
    if cmd == "win" and len(tokens) >= 2 and tokens[1].isdigit():
        win_id = int(tokens[1])
        state["current_win"] = win_id
        if win_id not in state["windows"]:
            state["windows"][win_id] = QDPWindow(window_id=win_id)
        return

    # yplot <y1> <y2> ...
    if cmd == "yplot" and len(tokens) >= 2:
        win = _get_active_window(state)
        for tok in tokens[1:]:
            if tok.isdigit():
                y_idx = int(tok)
                if y_idx not in win.y_vectors:
                    win.y_vectors.append(y_idx)
        return

    # line step <vec...> | line off <vec...>
    if cmd == "line" and len(tokens) >= 3:
        sub = tokens[1].lower()
        for tok in tokens[2:]:
            if tok.isdigit():
                vec = int(tok)
                if sub == "step":
                    state["step_vectors"].add(vec)
                elif sub == "off":
                    state["hidden_vectors"].add(vec)
        return

    # col <c> on <vec> | col off <vec>
    if cmd in ("col", "color") and len(tokens) >= 3:
        if tokens[1].lower() == "off":
            for tok in tokens[2:]:
                if tok.isdigit():
                    state["hidden_vectors"].add(int(tok))
        elif len(tokens) >= 4 and tokens[1].isdigit() and tokens[2].lower() == "on":
            color_idx = int(tokens[1])
            for tok in tokens[3:]:
                if tok.isdigit():
                    state["color_map"][int(tok)] = color_idx
        return

    # log x [on|off] / log y [on|off]
    if cmd == "log" and len(tokens) >= 2:
        win = _get_active_window(state)
        axis = tokens[1].lower()
        flag = True
        if len(tokens) >= 3 and tokens[2].lower() == "off":
            flag = False
        if "x" in axis:
            win.log_x = flag
        if "y" in axis:
            win.log_y = flag
        return

    # r x <min> <max> | r y1 <min> <max>
    if cmd == "r" and len(tokens) >= 4:
        axis_tok = tokens[1].lower()
        if _is_number(tokens[2]) and _is_number(tokens[3]):
            lo, hi = float(tokens[2]), float(tokens[3])
            axis_char = axis_tok[0]
            win_num_str = axis_tok[1:]
            if win_num_str.isdigit():
                win = _ensure_window(state, int(win_num_str))
            else:
                win = _get_active_window(state)
            if axis_char == "x":
                win.xlim = (lo, hi)
            elif axis_char == "y":
                win.ylim = (lo, hi)
        return

    # la / lab commands
    if cmd in ("la", "lab") and len(tokens) >= 2:
        m = _LABEL_POS_RE.match(line)
        if m:
            lbl_id = int(m.group(1))
            if _is_number(m.group(2)) and _is_number(m.group(3)):
                lx = float(m.group(2))
                ly = float(m.group(3))
                rest = m.group(4)
                justify = "center"
                rest_lower = rest.lower()
                if "just left" in rest_lower:
                    justify = "left"
                elif "just right" in rest_lower:
                    justify = "right"
                if '"' in rest:
                    text = _strip_quotes(rest[rest.index('"') :])
                else:
                    text = _strip_quotes(rest)
                win = _get_active_window(state)
                win.labels.append(
                    QDPLabel(
                        label_id=lbl_id,
                        x=lx,
                        y=ly,
                        text=text,
                        justify=justify,
                    )
                )
            return

        sub = tokens[1].lower()
        rest_text = line.split(None, 2)[2].strip() if len(tokens) >= 3 else ""
        win = _get_active_window(state)
        if sub == "x":
            win.xlabel = _strip_quotes(rest_text) or "X"
        elif sub == "y":
            win.ylabel = _strip_quotes(rest_text) or "Y"
        elif sub == "t":
            win.title = _strip_quotes(rest_text)
        return


def _ensure_window(state: dict[str, Any], win_id: int) -> QDPWindow:
    if win_id not in state["windows"]:
        state["windows"][win_id] = QDPWindow(window_id=win_id)
    return state["windows"][win_id]


def _get_active_window(state: dict[str, Any]) -> QDPWindow:
    win_id = state["current_win"]
    return _ensure_window(state, win_id)


def load_qdp(path: str | Path) -> QDPData:
    """Load and parse an XMM-Newton or standard HEASARC QDP file.

    Parameters
    ----------
    path:
        Path to the ``.qdp`` file.

    Returns
    -------
    QDPData
        Parsed tables, series with error bars, plot windows, and commands.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return parse_qdp_text(text, path=file_path)


def parse_qdp_text(text: str, path: Path | None = None) -> QDPData:
    """Parse QDP content from a string."""
    state: dict[str, Any] = {
        "serr": set(),
        "terr": set(),
        "current_win": 1,
        "windows": {},
        "step_vectors": set(),
        "hidden_vectors": set(),
        "color_map": {},
    }
    commands: list[str] = []
    tables_rows: list[list[list[float]]] = [[]]

    for raw_line in text.splitlines():
        # Strip inline comments starting with '!'
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue

        tokens = line.split()
        first = tokens[0]

        # Full-row 'NO' separator splits tables when all tokens are 'NO'
        if all(tok.upper() == "NO" for tok in tokens):
            if tables_rows[-1]:
                tables_rows.append([])
            continue

        if _is_number(first) or (
            first.upper() == "NO" and any(_is_number(t) for t in tokens[1:])
        ):
            row = [_parse_qdp_float(tok) for tok in tokens]
            tables_rows[-1].append(row)
        else:
            commands.append(line)
            _parse_command(line, state)

    raw_tables: list[np.ndarray] = []
    for rows in tables_rows:
        if not rows:
            continue
        max_cols = max(len(r) for r in rows)
        padded = [r + [float("nan")] * (max_cols - len(r)) for r in rows]
        raw_tables.append(np.asarray(padded, dtype=np.float64))

    series_list: list[QDPSeries] = []
    if raw_tables:
        # Build series from the first (or stacked) table
        arr = raw_tables[0]
        series_list = _build_series(arr, state)

    windows = [state["windows"][k] for k in sorted(state["windows"])]
    if not windows:
        windows = [
            QDPWindow(
                window_id=1,
                title=path.name if path else "QDP Plot",
                y_vectors=[s.y_index for s in series_list],
            )
        ]
    else:
        for win in windows:
            if not win.y_vectors:
                win.y_vectors = [s.y_index for s in series_list]
            if not win.title and win.labels:
                # Use the primary title label if no `la t` was given
                for lbl in win.labels:
                    if lbl.label_id == win.window_id:
                        win.title = lbl.text
                        break

        # Attach any unassigned Y-vectors (e.g. ESAS Fit/Select Lims) to a matching window
        assigned = {y for w in windows for y in w.y_vectors}
        for s in series_list:
            if s.y_index in assigned:
                continue
            valid = np.isfinite(s.x) & np.isfinite(s.y) & (s.y != -1.0)
            if not np.any(valid):
                continue
            xmin, xmax = float(np.min(s.x[valid])), float(np.max(s.x[valid]))
            target_win = windows[0]
            for w in windows:
                if w.xlim is not None and w.xlim[0] <= xmin and xmax <= w.xlim[1]:
                    target_win = w
                    break
            target_win.y_vectors.append(s.y_index)

    return QDPData(
        path=path,
        raw_tables=raw_tables,
        series=series_list,
        windows=windows,
        commands=commands,
        serr_vectors=set(state["serr"]),
        terr_vectors=set(state["terr"]),
    )


def _build_series(arr: np.ndarray, state: dict[str, Any]) -> list[QDPSeries]:
    """Map raw numeric columns to X/Y QDPSeries using ``read serr`` / ``read terr``."""
    ncols = arr.shape[1]
    if ncols < 2:
        return []

    serr: set[int] = state["serr"]
    terr: set[int] = state["terr"]

    col_idx = 0
    vec_idx = 1

    # Vector 1 is X
    x = arr[:, col_idx]
    col_idx += 1
    xerr_low: np.ndarray | None = None
    xerr_high: np.ndarray | None = None
    if vec_idx in serr and col_idx < ncols:
        err = arr[:, col_idx]
        xerr_low = err
        xerr_high = err
        col_idx += 1
    elif vec_idx in terr and col_idx + 1 < ncols:
        xerr_high = arr[:, col_idx]
        xerr_low = np.abs(arr[:, col_idx + 1])
        col_idx += 2

    series_list: list[QDPSeries] = []
    y_idx = 1
    vec_idx = 2

    while col_idx < ncols:
        y = arr[:, col_idx]
        col_idx += 1
        yerr_low: np.ndarray | None = None
        yerr_high: np.ndarray | None = None
        if vec_idx in serr and col_idx < ncols:
            err = arr[:, col_idx]
            yerr_low = err
            yerr_high = err
            col_idx += 1
        elif vec_idx in terr and col_idx + 1 < ncols:
            yerr_high = arr[:, col_idx]
            yerr_low = np.abs(arr[:, col_idx + 1])
            col_idx += 2

        series_list.append(
            QDPSeries(
                vector_index=vec_idx,
                y_index=y_idx,
                name=f"Y{y_idx} (vec {vec_idx})",
                x=x,
                y=y,
                yerr_low=yerr_low,
                yerr_high=yerr_high,
                xerr_low=xerr_low,
                xerr_high=xerr_high,
                step=vec_idx in state["step_vectors"],
                visible=vec_idx not in state["hidden_vectors"],
                color_index=state["color_map"].get(y_idx),
            )
        )
        y_idx += 1
        vec_idx += 1

    return series_list
