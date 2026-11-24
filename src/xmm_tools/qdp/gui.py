"""Dear ImGui + ImPlot viewer for XMM-Newton QDP files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

from imgui_bundle import (
    immapp,
    imgui,
    implot,
    portable_file_dialogs as pfd,
)
import numpy as np

from xmm_tools.qdp.parser import QDPData, QDPSeries, QDPWindow, load_qdp


@dataclass
class SeriesViewState:
    """Interactive display toggles for a single QDPSeries."""

    visible: bool = True
    step: bool = False
    show_errors: bool = True


@dataclass
class WindowViewState:
    """Interactive display toggles for a single QDPWindow."""

    log_x: bool = False
    log_y: bool = False


_DEFAULT_RECENT_PATH = Path.home() / ".config" / "xmm_tools" / "recent_qdp.json"
_MAX_RECENT_FILES = 10


def _load_recent_files(store_path: Path = _DEFAULT_RECENT_PATH) -> list[str]:
    """Load recent QDP file paths from disk, ignoring missing/corrupt data."""
    try:
        raw = json.loads(store_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(p) for p in raw if isinstance(p, str)][:_MAX_RECENT_FILES]
    except OSError:
        pass
    except json.JSONDecodeError:
        pass
    return []


def _save_recent_files(
    recent: list[str], store_path: Path = _DEFAULT_RECENT_PATH
) -> None:
    """Persist recent QDP file paths to disk."""
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(
            json.dumps(recent[:_MAX_RECENT_FILES], indent=2), encoding="utf-8"
        )
    except OSError:
        pass


@dataclass
class QDPViewerState:
    """Application state for the ImGui QDP viewer."""

    data: QDPData | None = None
    error_message: str = ""
    path_input: str = ""
    hide_minus_one: bool = True
    show_annotations: bool = True
    limit_reset_epoch: int = 0
    file_dialog: pfd.open_file | None = None
    recent_store_path: Path = _DEFAULT_RECENT_PATH
    recent_files: list[str] = field(default_factory=list)
    series_states: dict[int, SeriesViewState] = field(default_factory=dict)
    window_states: dict[int, WindowViewState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.recent_files:
            self.recent_files = _load_recent_files(self.recent_store_path)

    def _record_recent(self, path: Path) -> None:
        resolved = str(path.resolve())
        updated = [resolved] + [p for p in self.recent_files if p != resolved]
        self.recent_files = updated[:_MAX_RECENT_FILES]
        _save_recent_files(self.recent_files, self.recent_store_path)

    def clear_recent(self) -> None:
        self.recent_files = []
        _save_recent_files(self.recent_files, self.recent_store_path)

    def open_file(self, path: str | Path) -> None:
        """Load a QDP file and initialize per-window / per-series view state."""
        try:
            loaded = load_qdp(path)
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"Failed to load {path}: {exc}"
            return

        self.data = loaded
        self.error_message = ""
        self.path_input = str(loaded.path) if loaded.path else ""
        if loaded.path is not None:
            self._record_recent(loaded.path)
        self.limit_reset_epoch += 1
        self.series_states = {
            s.y_index: SeriesViewState(
                visible=s.visible,
                step=s.step,
                show_errors=(s.yerr_low is not None),
            )
            for s in loaded.series
        }
        self.window_states = {
            w.window_id: WindowViewState(log_x=w.log_x, log_y=w.log_y)
            for w in loaded.windows
        }


def _filter_series_arrays(
    series: QDPSeries,
    *,
    hide_minus_one: bool,
    log_x: bool,
    log_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return contiguous float64 arrays filtered for NaN/Inf, sentinel -1, and log<=0."""
    mask = np.isfinite(series.x) & np.isfinite(series.y)
    if hide_minus_one:
        mask &= series.y != -1.0
    if log_x:
        mask &= series.x > 0.0
    if log_y:
        mask &= series.y > 0.0

    xs = np.ascontiguousarray(series.x[mask], dtype=np.float64)
    ys = np.ascontiguousarray(series.y[mask], dtype=np.float64)

    yerr_lo: np.ndarray | None = None
    yerr_hi: np.ndarray | None = None
    if series.yerr_low is not None and series.yerr_high is not None:
        lo = np.where(np.isfinite(series.yerr_low[mask]), series.yerr_low[mask], 0.0)
        hi = np.where(np.isfinite(series.yerr_high[mask]), series.yerr_high[mask], 0.0)
        if log_y:
            # Clamp lower error so y - err_lo stays strictly positive on log10 plots
            lo = np.minimum(lo, ys * 0.999)
        yerr_lo = np.ascontiguousarray(np.maximum(lo, 0.0), dtype=np.float64)
        yerr_hi = np.ascontiguousarray(np.maximum(hi, 0.0), dtype=np.float64)

    return xs, ys, yerr_lo, yerr_hi


def _draw_toolbar(state: QDPViewerState) -> None:
    if imgui.button("Open QDP File..."):
        state.file_dialog = pfd.open_file(
            "Open XMM-Newton QDP File",
            str(Path(state.path_input).parent) if state.path_input else "",
            ["QDP Files (*.qdp *.QDP)", "*.qdp *.QDP", "All Files", "*"],
        )

    imgui.same_line()
    if imgui.button("Recent"):
        imgui.open_popup("##recent_files_popup")

    if imgui.begin_popup("##recent_files_popup"):
        if not state.recent_files:
            imgui.text_disabled("No recent files")
        else:
            for idx, recent_str in enumerate(list(state.recent_files)):
                rp = Path(recent_str)
                clicked, _ = imgui.menu_item(f"{rp.name}##recent_{idx}", "", False)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(recent_str)
                if clicked:
                    state.open_file(rp)
            imgui.separator()
            clear_clicked, _ = imgui.menu_item("Clear Recent", "", False)
            if clear_clicked:
                state.clear_recent()
        imgui.end_popup()

    imgui.same_line()
    imgui.set_next_item_width(480.0)
    changed, new_path = imgui.input_text("##path_input", state.path_input)
    if changed:
        state.path_input = new_path

    imgui.same_line()
    if imgui.button("Load") and state.path_input.strip():
        state.open_file(state.path_input.strip())

    if state.data is not None:
        imgui.same_line()
        if imgui.button("Reset Axes"):
            state.limit_reset_epoch += 1

    if state.file_dialog is not None and state.file_dialog.ready():
        selection = state.file_dialog.result()
        state.file_dialog = None
        if selection:
            state.open_file(selection[0])

    if state.error_message:
        imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.35, 1.0), state.error_message)


def _draw_sidebar(state: QDPViewerState) -> None:
    data = state.data
    if data is None:
        imgui.text_wrapped("Open a .qdp file to inspect its series, windows, and data.")
        return

    file_label = data.path.name if data.path else "<memory>"
    imgui.text(f"File: {file_label}")
    nrows = data.raw_tables[0].shape[0] if data.raw_tables else 0
    ncols = data.raw_tables[0].shape[1] if data.raw_tables else 0
    imgui.text(f"Rows: {nrows}  |  Cols: {ncols}  |  Windows: {len(data.windows)}")

    _, state.hide_minus_one = imgui.checkbox(
        "Hide -1.0 sentinel values", state.hide_minus_one
    )
    _, state.show_annotations = imgui.checkbox(
        "Show in-plot annotations", state.show_annotations
    )
    imgui.separator()

    series_by_y = {s.y_index: s for s in data.series}
    for win in data.windows:
        wstate = state.window_states[win.window_id]
        header = win.title or f"Window {win.window_id}"
        if imgui.collapsing_header(
            f"{header}##win_{win.window_id}", imgui.TreeNodeFlags_.default_open
        ):
            _, wstate.log_x = imgui.checkbox(
                f"Log X##wx_{win.window_id}", wstate.log_x
            )
            imgui.same_line()
            _, wstate.log_y = imgui.checkbox(
                f"Log Y##wy_{win.window_id}", wstate.log_y
            )

            for y_idx in win.y_vectors:
                s = series_by_y.get(y_idx)
                if s is None:
                    continue
                sstate = state.series_states[s.y_index]
                _, sstate.visible = imgui.checkbox(
                    f"{s.name}##vis_{win.window_id}_{y_idx}", sstate.visible
                )
                if sstate.visible:
                    imgui.same_line()
                    _, sstate.step = imgui.checkbox(
                        f"Step##stp_{win.window_id}_{y_idx}", sstate.step
                    )
                    if s.yerr_low is not None:
                        imgui.same_line()
                        _, sstate.show_errors = imgui.checkbox(
                            f"Err##err_{win.window_id}_{y_idx}", sstate.show_errors
                        )

    if any(w.labels for w in data.windows):
        if imgui.collapsing_header("Annotations"):
            for win in data.windows:
                for lbl in win.labels:
                    imgui.text_wrapped(f"[W{win.window_id}] {lbl.text}")

    if data.commands and imgui.collapsing_header(
        f"QDP Commands ({len(data.commands)})"
    ):
        for cmd in data.commands:
            imgui.text_unformatted(cmd)

    if data.raw_tables and imgui.collapsing_header("Data Preview (first 50 rows)"):
        arr = data.raw_tables[0]
        show_cols = min(arr.shape[1], 8)
        show_rows = min(arr.shape[0], 50)
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.scroll_x
        if imgui.begin_table("##qdp_preview", show_cols, flags, imgui.ImVec2(0, 220)):
            for c in range(show_cols):
                imgui.table_setup_column(f"C{c + 1}")
            imgui.table_headers_row()
            for r in range(show_rows):
                imgui.table_next_row()
                for c in range(show_cols):
                    imgui.table_set_column_index(c)
                    imgui.text(f"{arr[r, c]:.4g}")
            imgui.end_table()


_PGPLOT_COLORS: dict[int, imgui.ImVec4] = {
    1: imgui.ImVec4(0.92, 0.92, 0.95, 1.0),  # foreground / white
    2: imgui.ImVec4(0.95, 0.28, 0.28, 1.0),  # red
    3: imgui.ImVec4(0.25, 0.90, 0.40, 1.0),  # green
    4: imgui.ImVec4(0.32, 0.62, 1.00, 1.0),  # blue
    5: imgui.ImVec4(0.20, 0.90, 0.95, 1.0),  # cyan
    6: imgui.ImVec4(0.95, 0.35, 0.90, 1.0),  # magenta
    7: imgui.ImVec4(0.98, 0.85, 0.22, 1.0),  # yellow
    8: imgui.ImVec4(1.00, 0.55, 0.15, 1.0),  # orange
}


def _series_spec(series: QDPSeries) -> implot.Spec:
    """Build an ImPlot Spec using PGPLOT color indices when available."""
    spec = implot.Spec(line_weight=1.6)
    if series.color_index is not None and series.color_index in _PGPLOT_COLORS:
        spec.line_color = _PGPLOT_COLORS[series.color_index]
    return spec


def _draw_plot_annotations(win: QDPWindow, wstate: WindowViewState) -> None:
    """Render QDP text annotations with proper justification inside the plot rect."""
    draw_list = imgui.get_window_draw_list()
    plot_pos = implot.get_plot_pos()
    plot_size = implot.get_plot_size()
    x_min = plot_pos.x + 6.0
    x_max = plot_pos.x + plot_size.x - 6.0
    y_min = plot_pos.y + 4.0
    y_max = plot_pos.y + plot_size.y - 4.0

    for lbl in win.labels:
        if lbl.label_id == win.window_id and lbl.text == win.title:
            continue
        if wstate.log_x and lbl.x <= 0:
            continue
        if wstate.log_y and lbl.y <= 0:
            continue

        px = implot.plot_to_pixels(lbl.x, lbl.y)
        text_size = imgui.calc_text_size(lbl.text)

        if lbl.justify == "left":
            tx = px.x
        elif lbl.justify == "right":
            tx = px.x - text_size.x
        else:
            tx = px.x - text_size.x * 0.5
        ty = px.y - text_size.y * 0.5

        tx = max(x_min, min(tx, x_max - text_size.x))
        ty = max(y_min, min(ty, y_max - text_size.y))

        bg_col = imgui.get_color_u32(imgui.ImVec4(0.08, 0.09, 0.11, 0.72))
        fg_col = imgui.get_color_u32(imgui.ImVec4(0.95, 0.95, 0.95, 1.0))
        lower = lbl.text.lower()
        if "blue" in lower:
            fg_col = imgui.get_color_u32(_PGPLOT_COLORS[4])
        elif "red" in lower:
            fg_col = imgui.get_color_u32(_PGPLOT_COLORS[2])

        draw_list.add_rect_filled(
            imgui.ImVec2(tx - 3.0, ty - 1.0),
            imgui.ImVec2(tx + text_size.x + 3.0, ty + text_size.y + 1.0),
            bg_col,
            3.0,
        )
        draw_list.add_text(imgui.ImVec2(tx, ty), fg_col, lbl.text)


def _draw_window_plot(
    state: QDPViewerState,
    win: QDPWindow,
    series_by_y: dict[int, QDPSeries],
    plot_height: float,
) -> None:
    wstate = state.window_states[win.window_id]
    title = win.title or f"Window {win.window_id}"
    plot_id = f"{title}##plot_{win.window_id}_{state.limit_reset_epoch}"

    if implot.begin_plot(plot_id, imgui.ImVec2(-1.0, plot_height)):
        implot.setup_axes(win.xlabel or "X", win.ylabel or "Y")
        implot.setup_axis_scale(
            implot.ImAxis_.x1,
            implot.Scale_.log10 if wstate.log_x else implot.Scale_.linear,
        )
        implot.setup_axis_scale(
            implot.ImAxis_.y1,
            implot.Scale_.log10 if wstate.log_y else implot.Scale_.linear,
        )

        if win.xlim is not None:
            xlo, xhi = win.xlim
            if wstate.log_x and xlo <= 0:
                xlo = 1e-4
            if xhi > xlo:
                implot.setup_axis_limits(
                    implot.ImAxis_.x1, xlo, xhi, implot.Cond_.once
                )
        if win.ylim is not None:
            ylo, yhi = win.ylim
            if wstate.log_y and ylo <= 0:
                ylo = 1e-4
            if yhi > ylo:
                implot.setup_axis_limits(
                    implot.ImAxis_.y1, ylo, yhi, implot.Cond_.once
                )

        for y_idx in win.y_vectors:
            s = series_by_y.get(y_idx)
            if s is None:
                continue
            sstate = state.series_states[s.y_index]
            if not sstate.visible:
                continue

            xs, ys, yerr_lo, yerr_hi = _filter_series_arrays(
                s,
                hide_minus_one=state.hide_minus_one,
                log_x=wstate.log_x,
                log_y=wstate.log_y,
            )
            if xs.size == 0:
                continue

            spec = _series_spec(s)
            if sstate.show_errors and yerr_lo is not None and yerr_hi is not None:
                implot.plot_error_bars(
                    f"{s.name} (err)##{y_idx}", xs, ys, yerr_lo, yerr_hi, spec
                )

            if sstate.step:
                implot.plot_stairs(s.name, xs, ys, spec)
            else:
                implot.plot_line(s.name, xs, ys, spec)

        if state.show_annotations:
            _draw_plot_annotations(win, wstate)

        implot.end_plot()


def _draw_plots(state: QDPViewerState) -> None:
    data = state.data
    if data is None:
        avail = imgui.get_content_region_avail()
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + max(20.0, avail.y * 0.35))
        imgui.text("No QDP file loaded. Click 'Open QDP File...' above to begin.")
        return

    series_by_y = {s.y_index: s for s in data.series}
    n_wins = max(len(data.windows), 1)
    avail = imgui.get_content_region_avail()
    spacing = imgui.get_style().item_spacing.y
    plot_height = max(220.0, (avail.y - spacing * (n_wins - 1)) / n_wins)

    for win in data.windows:
        _draw_window_plot(state, win, series_by_y, plot_height)


def run_gui(initial_file: str | Path | None = None) -> None:
    """Launch the Dear ImGui QDP viewer application."""
    state = QDPViewerState()
    if initial_file is not None:
        state.open_file(initial_file)

    def gui() -> None:
        _draw_toolbar(state)
        imgui.separator()

        avail = imgui.get_content_region_avail()
        sidebar_width = min(340.0, max(240.0, avail.x * 0.26))

        if imgui.begin_child("##sidebar", imgui.ImVec2(sidebar_width, 0.0), imgui.ChildFlags_.borders):
            _draw_sidebar(state)
        imgui.end_child()

        imgui.same_line()

        if imgui.begin_child("##plots", imgui.ImVec2(0.0, 0.0)):
            _draw_plots(state)
        imgui.end_child()

    immapp.run(
        gui,
        with_implot=True,
        window_title="XMM-Newton QDP Viewer",
        window_size=(1360, 860),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``xmm-qdp`` / ``python -m xmm_tools.qdp``."""
    parser = argparse.ArgumentParser(
        description="Interactive ImGui viewer for XMM-Newton QDP files."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Optional path to a .qdp file to open on startup.",
    )
    args = parser.parse_args(argv)
    run_gui(args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
