"""Dear ImGui + ImPlot viewer for FITS flux images and interactive region generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

from astropy.coordinates import SkyCoord
import astropy.units as u
from imgui_bundle import (
    immapp,
    imgui,
    implot,
    portable_file_dialogs as pfd,
)
import numpy as np

from xmm_tools.image.fits_io import (
    FitsImageData,
    load_fits_image,
    pixel_to_sky,
    sky_to_pixel,
    smooth_image,
    stretch_image,
)
from xmm_tools.image.regions import (
    BoxRegion,
    CircleRegion,
    EllipseRegion,
    PolygonRegion,
    RegionItem,
    StripGridParams,
    box_corners_pixels,
    ellipse_outline_pixels,
    export_ds9_regions,
    generate_strip_grid,
    parse_ds9_regions,
    polygon_pixels,
    resize_box_edge,
    strip_params_from_sky_points,
)


_DEFAULT_RECENT_FITS_PATH = Path.home() / ".config" / "xmm_tools" / "recent_fits.json"
_MAX_RECENT_FILES = 10

_COLORMAPS: list[tuple[str, int]] = [
    ("Viridis", implot.Colormap_.viridis),
    ("Plasma", implot.Colormap_.plasma),
    ("Inferno", implot.Colormap_.hot),
    ("Greys", implot.Colormap_.greys),
    ("Jet", implot.Colormap_.jet),
    ("Cool", implot.Colormap_.cool),
    ("Spectral", implot.Colormap_.spectral),
]
_STRETCH_MODES: list[str] = ["asinh", "log10", "sqrt", "linear"]


def _load_recent_fits(store_path: Path = _DEFAULT_RECENT_FITS_PATH) -> list[str]:
    try:
        raw = json.loads(store_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(p) for p in raw if isinstance(p, str)][:_MAX_RECENT_FILES]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save_recent_fits(
    recent: list[str], store_path: Path = _DEFAULT_RECENT_FITS_PATH
) -> None:
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(
            json.dumps(recent[:_MAX_RECENT_FILES], indent=2), encoding="utf-8"
        )
    except OSError:
        pass


@dataclass
class ImageViewerState:
    """Application state for the FITS Flux Image & Region GUI."""

    image: FitsImageData | None = None
    error_message: str = ""
    path_input: str = ""
    recent_store_path: Path = _DEFAULT_RECENT_FITS_PATH
    recent_files: list[str] = field(default_factory=list)

    # Display & smoothing cache
    colormap_idx: int = 0
    stretch_idx: int = 0
    smooth_arcsec: float = 0.0
    vmin: float = 0.0
    vmax: float = 1.0
    auto_limits: bool = True
    manual_limits: bool = False
    view_reset_epoch: int = 0

    _cached_smooth_arcsec: float = -1.0
    _smoothed_data: np.ndarray | None = None
    _cached_display_key: tuple | None = None
    _display_flipped: np.ndarray | None = None

    # Interactive Strip Grid & Regions
    preview_strip: bool = True
    strip_end_sky: tuple[float, float] | None = None
    selected_region_idx: int | None = None
    add_shape_idx: int = 0  # 0=Circle, 1=Box, 2=Ellipse, 3=Polygon
    add_name: str = ""
    add_include: bool = True
    add_radius_arcsec: float = 30.0
    add_width_arcsec: float = 40.0
    add_height_arcsec: float = 40.0
    add_semi_major_arcsec: float = 35.0
    add_semi_minor_arcsec: float = 20.0
    add_angle_deg: float = 0.0
    placing_region: bool = False
    poly_draft_vertices: list[tuple[float, float]] = field(default_factory=list)
    strip_params: StripGridParams = field(
        default_factory=lambda: StripGridParams(
            start_ra_deg=0.0,
            start_dec_deg=0.0,
            pa_deg=90.0,
            width_arcsec=24.0,
            height_arcsec=60.0,
            count=6,
            gap_arcsec=0.0,
            prefix="bin",
        )
    )
    regions: list[RegionItem] = field(default_factory=list)

    # File dialogs
    open_fits_dialog: pfd.open_file | None = None
    open_reg_dialog: pfd.open_file | None = None
    save_reg_dialog: pfd.save_file | None = None

    def __post_init__(self) -> None:
        if not self.recent_files:
            self.recent_files = _load_recent_fits(self.recent_store_path)

    def _record_recent(self, path: Path) -> None:
        resolved = str(path.resolve())
        updated = [resolved] + [p for p in self.recent_files if p != resolved]
        self.recent_files = updated[:_MAX_RECENT_FILES]
        _save_recent_fits(self.recent_files, self.recent_store_path)

    def open_file(self, path: str | Path) -> None:
        try:
            loaded = load_fits_image(path)
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"Failed to load {path}: {exc}"
            return

        self.image = loaded
        self.error_message = ""
        self.path_input = str(loaded.path) if loaded.path else ""
        if loaded.path is not None:
            self._record_recent(loaded.path)
        self._cached_smooth_arcsec = -1.0
        self._smoothed_data = None
        self._cached_display_key = None
        self._display_flipped = None
        self.auto_limits = True
        self.manual_limits = False
        self.view_reset_epoch += 1

        # Initialize strip start/end around image center so handles are immediately visible
        ny, nx = loaded.shape
        start_sky = pixel_to_sky(loaded, nx * 0.42, ny * 0.5)
        end_sky = pixel_to_sky(loaded, nx * 0.62, ny * 0.5)
        if start_sky is not None and end_sky is not None:
            self.strip_params = strip_params_from_sky_points(
                start_sky[0],
                start_sky[1],
                end_sky[0],
                end_sky[1],
                width_arcsec=self.strip_params.width_arcsec,
                height_arcsec=self.strip_params.height_arcsec,
                gap_arcsec=self.strip_params.gap_arcsec,
                prefix=self.strip_params.prefix,
            )
            self.strip_end_sky = end_sky

    def get_strip_end_sky(self) -> tuple[float, float]:
        if self.strip_end_sky is not None:
            return self.strip_end_sky
        sp = self.strip_params
        origin = SkyCoord(ra=sp.start_ra_deg * u.deg, dec=sp.start_dec_deg * u.deg, frame="fk5")
        total_len = sp.count * (sp.width_arcsec + sp.gap_arcsec)
        end = origin.directional_offset_by(sp.pa_deg * u.deg, total_len * u.arcsec)
        self.strip_end_sky = (float(end.ra.deg), float(end.dec.deg))
        return self.strip_end_sky

    def get_smoothed_data(self) -> np.ndarray | None:
        if self.image is None:
            return None
        if (
            self._smoothed_data is None
            or abs(self._cached_smooth_arcsec - self.smooth_arcsec) > 1e-6
        ):
            sigma_pix = self.smooth_arcsec / max(self.image.pixel_scale_arcsec, 1e-6)
            self._smoothed_data = smooth_image(self.image.data, sigma_pix)
            self._cached_smooth_arcsec = self.smooth_arcsec
            self._cached_display_key = None
            if not self.manual_limits:
                self.auto_limits = True
        return self._smoothed_data

    def get_display_flipped(self) -> np.ndarray | None:
        smoothed = self.get_smoothed_data()
        if smoothed is None:
            return None
        mode = _STRETCH_MODES[self.stretch_idx]
        key = (
            self._cached_smooth_arcsec,
            mode,
            self.auto_limits,
            None if self.auto_limits else self.vmin,
            None if self.auto_limits else self.vmax,
        )
        if self._display_flipped is None or self._cached_display_key != key:
            stretched, lo, hi = stretch_image(
                smoothed,
                mode=mode,
                vmin=None if self.auto_limits else self.vmin,
                vmax=None if self.auto_limits else self.vmax,
            )
            self.vmin = lo
            self.vmax = hi
            self.auto_limits = False
            self._display_flipped = np.ascontiguousarray(
                np.flipud(stretched), dtype=np.float64
            )
            self._cached_display_key = key
        return self._display_flipped

    def preview_boxes(self) -> list[BoxRegion]:
        if self.image is None or not self.preview_strip:
            return []
        return generate_strip_grid(self.strip_params)

    def apply_strip_grid(self, *, replace: bool = False) -> None:
        boxes = generate_strip_grid(self.strip_params)
        if replace:
            self.regions = list(boxes)
        else:
            self.regions.extend(boxes)

    def add_manual_region_at(self, ra_deg: float, dec_deg: float) -> None:
        """Create and append a Circle, Box, or Ellipse at ``(ra_deg, dec_deg)``."""
        idx = len(self.regions) + 1
        shape = self.add_shape_idx
        if shape == 0:
            name = self.add_name.strip() or f"c{idx}"
            self.regions.append(
                CircleRegion(
                    name=name,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    radius_arcsec=self.add_radius_arcsec,
                    include=self.add_include,
                )
            )
        elif shape == 1:
            name = self.add_name.strip() or f"b{idx}"
            self.regions.append(
                BoxRegion(
                    name=name,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    width_arcsec=self.add_width_arcsec,
                    height_arcsec=self.add_height_arcsec,
                    angle_deg=self.add_angle_deg,
                    include=self.add_include,
                )
            )
        elif shape == 2:
            name = self.add_name.strip() or f"e{idx}"
            self.regions.append(
                EllipseRegion(
                    name=name,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    semi_major_arcsec=self.add_semi_major_arcsec,
                    semi_minor_arcsec=self.add_semi_minor_arcsec,
                    angle_deg=self.add_angle_deg,
                    include=self.add_include,
                )
            )
        self.selected_region_idx = len(self.regions) - 1

    def finish_polygon_draft(self) -> None:
        """Commit collected polygon vertices as a ``PolygonRegion``."""
        if len(self.poly_draft_vertices) < 3:
            return
        idx = len(self.regions) + 1
        name = self.add_name.strip() or f"p{idx}"
        self.regions.append(
            PolygonRegion(
                name=name,
                vertices_deg=list(self.poly_draft_vertices),
                include=self.add_include,
            )
        )
        self.poly_draft_vertices.clear()
        self.placing_region = False
        self.selected_region_idx = len(self.regions) - 1


def _draw_toolbar(state: ImageViewerState) -> None:
    if imgui.button("Open FITS..."):
        state.open_fits_dialog = pfd.open_file(
            "Open FITS Image",
            str(Path(state.path_input).parent) if state.path_input else "",
            ["FITS Files (*.fits *.fit *.FITS *.FIT)", "*.fits *.fit *.FITS *.FIT", "All Files", "*"],
        )

    imgui.same_line()
    if imgui.button("Recent"):
        imgui.open_popup("##recent_fits_popup")
    if imgui.begin_popup("##recent_fits_popup"):
        if not state.recent_files:
            imgui.text_disabled("No recent FITS files")
        else:
            for idx, rstr in enumerate(list(state.recent_files)):
                rp = Path(rstr)
                clicked, _ = imgui.menu_item(f"{rp.name}##rf_{idx}", "", False)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(rstr)
                if clicked:
                    state.open_file(rp)
        imgui.end_popup()

    imgui.same_line()
    imgui.set_next_item_width(420.0)
    changed, new_path = imgui.input_text("##fits_path_input", state.path_input)
    if changed:
        state.path_input = new_path

    imgui.same_line()
    if imgui.button("Load") and state.path_input.strip():
        state.open_file(state.path_input.strip())

    if state.image is not None:
        imgui.same_line()
        if imgui.button("Reset View"):
            state.view_reset_epoch += 1

    # Poll native file dialogs
    if state.open_fits_dialog is not None and state.open_fits_dialog.ready():
        sel = state.open_fits_dialog.result()
        state.open_fits_dialog = None
        if sel:
            state.open_file(sel[0])

    if state.open_reg_dialog is not None and state.open_reg_dialog.ready():
        sel = state.open_reg_dialog.result()
        state.open_reg_dialog = None
        if sel:
            try:
                wcs = state.image.wcs if state.image is not None else None
                state.regions.extend(parse_ds9_regions(sel[0], wcs=wcs))
                state.error_message = ""
            except Exception as exc:  # noqa: BLE001
                state.error_message = f"Failed to import regions: {exc}"

    if state.save_reg_dialog is not None and state.save_reg_dialog.ready():
        sel = state.save_reg_dialog.result()
        state.save_reg_dialog = None
        if sel:
            try:
                export_ds9_regions(state.regions, sel)
                state.error_message = ""
            except Exception as exc:  # noqa: BLE001
                state.error_message = f"Failed to export regions: {exc}"

    if state.error_message:
        imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.35, 1.0), state.error_message)


def _draw_selected_region_editor(state: ImageViewerState) -> None:
    """Render live parameter editors for the currently selected region."""
    idx = state.selected_region_idx
    if idx is None or not (0 <= idx < len(state.regions)):
        return
    reg = state.regions[idx]

    imgui.separator_text(f"Edit Selected: {reg.name}")
    _, reg.name = imgui.input_text("Name##edit_name", reg.name)
    _, reg.include = imgui.checkbox("Include (off = Exclude -)##edit_inc", reg.include)

    if isinstance(reg, CircleRegion):
        _, reg.radius_arcsec = imgui.slider_float(
            "Radius (\")##edit_r", reg.radius_arcsec, 2.0, 400.0, "%.1f"
        )
        _, reg.ra_deg = imgui.input_double("RA (deg)##edit_cra", reg.ra_deg, 0.0, 0.0, "%.6f")
        _, reg.dec_deg = imgui.input_double("Dec (deg)##edit_cdec", reg.dec_deg, 0.0, 0.0, "%.6f")
    elif isinstance(reg, BoxRegion):
        _, reg.width_arcsec = imgui.slider_float(
            "Width (\")##edit_bw", reg.width_arcsec, 2.0, 400.0, "%.1f"
        )
        _, reg.height_arcsec = imgui.slider_float(
            "Height (\")##edit_bh", reg.height_arcsec, 2.0, 600.0, "%.1f"
        )
        _, reg.angle_deg = imgui.slider_float(
            "Angle (deg)##edit_ba", reg.angle_deg, 0.0, 360.0, "%.1f"
        )
        if imgui.tree_node("Center Coords##edit_bcoords"):
            _, reg.ra_deg = imgui.input_double("RA##edit_bra", reg.ra_deg, 0.0, 0.0, "%.6f")
            _, reg.dec_deg = imgui.input_double("Dec##edit_bdec", reg.dec_deg, 0.0, 0.0, "%.6f")
            imgui.tree_pop()
    elif isinstance(reg, EllipseRegion):
        _, reg.semi_major_arcsec = imgui.slider_float(
            "Semi-Major (\")##edit_ea", reg.semi_major_arcsec, 2.0, 400.0, "%.1f"
        )
        _, reg.semi_minor_arcsec = imgui.slider_float(
            "Semi-Minor (\")##edit_eb", reg.semi_minor_arcsec, 2.0, 400.0, "%.1f"
        )
        _, reg.angle_deg = imgui.slider_float(
            "Angle (deg)##edit_eang", reg.angle_deg, 0.0, 360.0, "%.1f"
        )
        if imgui.tree_node("Center Coords##edit_ecoords"):
            _, reg.ra_deg = imgui.input_double("RA##edit_era", reg.ra_deg, 0.0, 0.0, "%.6f")
            _, reg.dec_deg = imgui.input_double("Dec##edit_edec", reg.dec_deg, 0.0, 0.0, "%.6f")
            imgui.tree_pop()
    elif isinstance(reg, PolygonRegion):
        imgui.text_disabled(f"Vertices: {len(reg.vertices_deg)}")


def _draw_sidebar(state: ImageViewerState) -> None:
    if state.image is None:
        imgui.text_wrapped("Open a FITS flux/count image to adjust display & generate regions.")
        return

    img = state.image
    ny, nx = img.shape
    fname = img.path.name if img.path else "<memory>"
    imgui.text(f"File: {fname}")
    imgui.text_disabled(
        f"{nx}x{ny} px  |  {img.pixel_scale_arcsec:.2f}\"/px  |  "
        f"{'WCS: FK5' if img.wcs is not None else 'No WCS'}"
    )

    if imgui.collapsing_header("Display & Colorbar", imgui.TreeNodeFlags_.default_open):
        cmap_names = [name for name, _ in _COLORMAPS]
        changed, new_c = imgui.combo("Colormap", state.colormap_idx, cmap_names)
        if changed:
            state.colormap_idx = new_c

        changed, new_s = imgui.combo("Stretch", state.stretch_idx, _STRETCH_MODES)
        if changed:
            state.stretch_idx = new_s

        _, state.smooth_arcsec = imgui.slider_float(
            "Smooth (\")", state.smooth_arcsec, 0.0, 30.0, "%.1f"
        )

        if imgui.button("Auto Scale"):
            state.auto_limits = True
            state.manual_limits = False
        imgui.same_line()
        imgui.text_disabled(f"[{state.vmin:.3g}, {state.vmax:.3g}]")

        ch_min, new_min = imgui.drag_float(
            "vmin", state.vmin, max(abs(state.vmax - state.vmin) * 0.005, 1e-6), format="%.4g"
        )
        ch_max, new_max = imgui.drag_float(
            "vmax", state.vmax, max(abs(state.vmax - state.vmin) * 0.005, 1e-6), format="%.4g"
        )
        if ch_min:
            state.vmin = new_min
            state.manual_limits = True
        if ch_max:
            state.vmax = max(new_max, state.vmin + 1e-12)
            state.manual_limits = True

    if imgui.collapsing_header("Strip Grid Generator", imgui.TreeNodeFlags_.default_open):
        sp = state.strip_params
        _, state.preview_strip = imgui.checkbox("Show Strip Handles", state.preview_strip)
        imgui.same_line()
        imgui.text_disabled("(drag dots on image)")

        ch_w, sp.width_arcsec = imgui.slider_float("Bin Width (\")", sp.width_arcsec, 4.0, 200.0, "%.1f")
        _, sp.height_arcsec = imgui.slider_float("Bin Height (\")", sp.height_arcsec, 4.0, 400.0, "%.1f")
        ch_g, sp.gap_arcsec = imgui.slider_float("Gap (\")", sp.gap_arcsec, 0.0, 100.0, "%.1f")
        ch_c, sp.count = imgui.slider_int("Count", sp.count, 1, 40)
        if ch_w or ch_g or ch_c:
            state.strip_end_sky = None
        _, sp.prefix = imgui.input_text("Prefix", sp.prefix)

        if imgui.tree_node("Axis Coordinates"):
            ch_ra, sp.start_ra_deg = imgui.input_double("Start RA", sp.start_ra_deg, 0.0, 0.0, "%.6f")
            ch_dec, sp.start_dec_deg = imgui.input_double("Start Dec", sp.start_dec_deg, 0.0, 0.0, "%.6f")
            ch_pa, sp.pa_deg = imgui.slider_float("PA (deg)", sp.pa_deg, 0.0, 360.0, "%.1f")
            if ch_ra or ch_dec or ch_pa:
                state.strip_end_sky = None
            imgui.tree_pop()

        if imgui.button("Append Strip"):
            state.apply_strip_grid(replace=False)
        imgui.same_line()
        if imgui.button("Replace All"):
            state.apply_strip_grid(replace=True)

    if imgui.collapsing_header(
        f"Regions ({len(state.regions)})", imgui.TreeNodeFlags_.default_open
    ):
        if imgui.button("Import .reg..."):
            state.open_reg_dialog = pfd.open_file(
                "Import DS9 Region File",
                "",
                ["DS9 Regions (*.reg)", "*.reg", "All Files", "*"],
            )
        imgui.same_line()
        if imgui.button("Export .reg..."):
            state.save_reg_dialog = pfd.save_file(
                "Export DS9 Region File",
                "regions.reg",
                ["DS9 Regions (*.reg)", "*.reg"],
            )
        imgui.same_line()
        if imgui.button("Clear"):
            state.regions.clear()
            state.selected_region_idx = None

        imgui.separator_text("Add Region")
        shapes = ["Circle", "Box", "Ellipse", "Polygon"]
        imgui.set_next_item_width(120.0)
        ch_sh, new_sh = imgui.combo("Shape", state.add_shape_idx, shapes)
        if ch_sh:
            state.add_shape_idx = new_sh
            state.poly_draft_vertices.clear()
        imgui.same_line()
        _, state.add_include = imgui.checkbox("Include", state.add_include)
        _, state.add_name = imgui.input_text("Label", state.add_name)

        if state.add_shape_idx == 0:
            _, state.add_radius_arcsec = imgui.slider_float(
                "Radius (\")", state.add_radius_arcsec, 2.0, 300.0, "%.1f"
            )
        elif state.add_shape_idx == 1:
            _, state.add_width_arcsec = imgui.slider_float(
                "Box W (\")", state.add_width_arcsec, 2.0, 400.0, "%.1f"
            )
            _, state.add_height_arcsec = imgui.slider_float(
                "Box H (\")", state.add_height_arcsec, 2.0, 400.0, "%.1f"
            )
            _, state.add_angle_deg = imgui.slider_float(
                "Angle (deg)", state.add_angle_deg, 0.0, 360.0, "%.1f"
            )
        elif state.add_shape_idx == 2:
            _, state.add_semi_major_arcsec = imgui.slider_float(
                "Semi-Major (\")", state.add_semi_major_arcsec, 2.0, 300.0, "%.1f"
            )
            _, state.add_semi_minor_arcsec = imgui.slider_float(
                "Semi-Minor (\")", state.add_semi_minor_arcsec, 2.0, 300.0, "%.1f"
            )
            _, state.add_angle_deg = imgui.slider_float(
                "Angle (deg)", state.add_angle_deg, 0.0, 360.0, "%.1f"
            )

        if state.add_shape_idx < 3:
            if state.placing_region:
                if imgui.button("Cancel Place"):
                    state.placing_region = False
                imgui.same_line()
                imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.2, 1.0), "Click on image")
            else:
                if imgui.button("Click Image to Add"):
                    state.placing_region = True
                imgui.same_line()
                if imgui.button("Add at Center"):
                    ny_c, nx_c = img.shape
                    c_sky = pixel_to_sky(img, nx_c * 0.5, ny_c * 0.5)
                    if c_sky is not None:
                        state.add_manual_region_at(c_sky[0], c_sky[1])
        else:
            if state.placing_region:
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.85, 0.2, 1.0),
                    f"Click vertices ({len(state.poly_draft_vertices)})",
                )
                if len(state.poly_draft_vertices) >= 3 and imgui.button("Finish Polygon"):
                    state.finish_polygon_draft()
                imgui.same_line()
                if imgui.button("Cancel##poly"):
                    state.poly_draft_vertices.clear()
                    state.placing_region = False
            else:
                if imgui.button("Pick Polygon Vertices"):
                    state.poly_draft_vertices.clear()
                    state.placing_region = True

        if state.regions:
            imgui.separator_text("Region List")
            list_h = min(160.0, max(52.0, len(state.regions) * 24.0))
            if imgui.begin_child("##reg_list", imgui.ImVec2(0.0, list_h), imgui.ChildFlags_.borders):
                to_remove: int | None = None
                for idx, reg in enumerate(state.regions):
                    is_sel = state.selected_region_idx == idx
                    prefix = "" if reg.include else "[-] "
                    if isinstance(reg, BoxRegion):
                        desc = f"{prefix}{reg.name}  ({reg.width_arcsec:.0f}\"x{reg.height_arcsec:.0f}\")"
                    elif isinstance(reg, EllipseRegion):
                        desc = f"{prefix}{reg.name}  (ell {reg.semi_major_arcsec:.0f}\"x{reg.semi_minor_arcsec:.0f}\")"
                    elif isinstance(reg, PolygonRegion):
                        desc = f"{prefix}{reg.name}  (poly {len(reg.vertices_deg)}v)"
                    else:
                        desc = f"{prefix}{reg.name}  (r={reg.radius_arcsec:.0f}\")"
                    clicked, _ = imgui.selectable(
                        f"{desc}##sel_{idx}", is_sel, imgui.SelectableFlags_.allow_overlap
                    )
                    if clicked:
                        state.selected_region_idx = None if is_sel else idx
                    imgui.same_line()
                    if imgui.small_button(f"X##del_{idx}"):
                        to_remove = idx
                if to_remove is not None:
                    state.regions.pop(to_remove)
                    if state.selected_region_idx == to_remove:
                        state.selected_region_idx = None
                    elif (
                        state.selected_region_idx is not None
                        and state.selected_region_idx > to_remove
                    ):
                        state.selected_region_idx -= 1
            imgui.end_child()

        _draw_selected_region_editor(state)

    imgui.spacing()
    imgui.separator()
    imgui.text_disabled("Tips: Drag cyan/yellow dots to adjust strip.")
    imgui.text_disabled("Click a region in list to edit or stretch box edge.")


def _draw_region_overlays(state: ImageViewerState) -> None:
    if state.image is None:
        return
    img = state.image
    draw_list = imgui.get_window_draw_list()
    plot_pos = implot.get_plot_pos()
    plot_size = implot.get_plot_size()
    draw_list.push_clip_rect(
        plot_pos,
        imgui.ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y),
        True,
    )

    label_bg = imgui.get_color_u32(imgui.ImVec4(0.05, 0.06, 0.08, 0.82))

    def _draw_label(pos: imgui.ImVec2, text: str, fg_u32: int) -> None:
        ts = imgui.calc_text_size(text)
        tx = pos.x - ts.x * 0.5
        ty = pos.y - ts.y * 0.5
        draw_list.add_rect_filled(
            imgui.ImVec2(tx - 3.0, ty - 1.0),
            imgui.ImVec2(tx + ts.x + 3.0, ty + ts.y + 1.0),
            label_bg,
            3.0,
        )
        draw_list.add_text(imgui.ImVec2(tx, ty), fg_u32, text)

    def _draw_box(
        box: BoxRegion,
        fg_u32: int,
        fill_u32: int,
        thickness: float,
        *,
        draw_label: bool = True,
        label_above: bool = True,
    ) -> None:
        corners = box_corners_pixels(box, img)
        if corners is None:
            return
        pts = [implot.plot_to_pixels(x, y) for x, y in corners]
        draw_list.add_quad_filled(pts[0], pts[1], pts[2], pts[3], fill_u32)
        for i in range(4):
            draw_list.add_line(pts[i], pts[(i + 1) % 4], fg_u32, thickness)

        if not draw_label:
            return
        top_mid = imgui.ImVec2((pts[2].x + pts[3].x) * 0.5, (pts[2].y + pts[3].y) * 0.5)
        bot_mid = imgui.ImVec2((pts[0].x + pts[1].x) * 0.5, (pts[0].y + pts[1].y) * 0.5)
        if top_mid.y > bot_mid.y:
            top_mid, bot_mid = bot_mid, top_mid  # screen y grows downward
        lbl_pos = (
            imgui.ImVec2(top_mid.x, top_mid.y - 12.0)
            if label_above
            else imgui.ImVec2(bot_mid.x, bot_mid.y + 12.0)
        )
        _draw_label(lbl_pos, box.name, fg_u32)

    def _box_screen_step(boxes: list[BoxRegion]) -> float:
        if len(boxes) < 2:
            return 1e6
        c0 = sky_to_pixel(img, boxes[0].ra_deg, boxes[0].dec_deg)
        c1 = sky_to_pixel(img, boxes[1].ra_deg, boxes[1].dec_deg)
        if c0 is None or c1 is None:
            return 1e6
        p0 = implot.plot_to_pixels(c0[0], c0[1])
        p1 = implot.plot_to_pixels(c1[0], c1[1])
        return ((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5

    green_fg = imgui.get_color_u32(imgui.ImVec4(0.30, 0.95, 0.45, 0.95))
    green_fill = imgui.get_color_u32(imgui.ImVec4(0.30, 0.95, 0.45, 0.08))
    hi_fg = imgui.get_color_u32(imgui.ImVec4(1.00, 0.40, 0.80, 1.00))
    hi_fill = imgui.get_color_u32(imgui.ImVec4(1.00, 0.40, 0.80, 0.22))
    cyan_fg = imgui.get_color_u32(imgui.ImVec4(0.25, 0.92, 1.00, 0.95))
    cyan_fill = imgui.get_color_u32(imgui.ImVec4(0.25, 0.92, 1.00, 0.09))

    saved_boxes = [r for r in state.regions if isinstance(r, BoxRegion)]
    saved_step = _box_screen_step(saved_boxes)
    # Two rows (above + below) double the horizontal budget per label
    saved_stride = max(1, int((52.0 / max(saved_step, 1.0)) // 2) * 2 or 1)
    if saved_step < 26.0:
        saved_stride = max(saved_stride, int(52.0 / max(saved_step, 1.0)) + 1)

    for idx, reg in enumerate(state.regions):
        is_sel = state.selected_region_idx == idx
        fg = hi_fg if is_sel else green_fg
        fill = hi_fill if is_sel else green_fill
        lw = 2.2 if is_sel else 1.2
        if isinstance(reg, BoxRegion):
            show_lbl = is_sel or idx == 0 or idx == len(state.regions) - 1 or (idx % saved_stride == 0)
            _draw_box(reg, fg, fill, lw, draw_label=show_lbl, label_above=(idx % 2 == 0))
        elif isinstance(reg, CircleRegion):
            c_px = sky_to_pixel(img, reg.ra_deg, reg.dec_deg)
            if c_px is not None:
                c_scr = implot.plot_to_pixels(c_px[0], c_px[1])
                r_px = reg.radius_arcsec / max(img.pixel_scale_arcsec, 1e-6)
                edge_scr = implot.plot_to_pixels(c_px[0] + r_px, c_px[1])
                r_scr = abs(edge_scr.x - c_scr.x)
                draw_list.add_circle_filled(c_scr, r_scr, fill, 36)
                draw_list.add_circle(c_scr, r_scr, fg, 36, lw)
                _draw_label(imgui.ImVec2(c_scr.x, c_scr.y - r_scr - 10.0), reg.name, fg)
        elif isinstance(reg, EllipseRegion):
            outline = ellipse_outline_pixels(reg, img)
            if outline:
                pts = [implot.plot_to_pixels(x, y) for x, y in outline]
                draw_list.add_convex_poly_filled(pts, fill)
                draw_list.add_polyline(pts, fg, lw, imgui.ImDrawFlags_.closed)
                cx = sum(p.x for p in pts) / len(pts)
                cy = min(p.y for p in pts) - 10.0
                _draw_label(imgui.ImVec2(cx, cy), reg.name, fg)
        elif isinstance(reg, PolygonRegion):
            verts = polygon_pixels(reg, img)
            if verts and len(verts) >= 3:
                pts = [implot.plot_to_pixels(x, y) for x, y in verts]
                draw_list.add_polyline(pts, fg, lw, imgui.ImDrawFlags_.closed)
                cx = sum(p.x for p in pts) / len(pts)
                cy = sum(p.y for p in pts) / len(pts)
                _draw_label(imgui.ImVec2(cx, cy), reg.name, fg)

    prev_boxes = state.preview_boxes()
    prev_step = _box_screen_step(prev_boxes)
    prev_stride = 1 if prev_step >= 28.0 else max(2, int(56.0 / max(prev_step, 1.0)) + 1)
    for idx, box in enumerate(prev_boxes):
        show_lbl = idx == 0 or idx == len(prev_boxes) - 1 or (idx % prev_stride == 0)
        _draw_box(box, cyan_fg, cyan_fill, 1.2, draw_label=show_lbl, label_above=(idx % 2 == 0))

    # Draw in-progress polygon vertices
    if state.poly_draft_vertices and img.wcs is not None:
        draft_pts: list[imgui.ImVec2] = []
        for ra, dec in state.poly_draft_vertices:
            px = sky_to_pixel(img, ra, dec)
            if px is not None:
                draft_pts.append(implot.plot_to_pixels(px[0], px[1]))
        if draft_pts:
            poly_col = imgui.get_color_u32(imgui.ImVec4(1.0, 0.85, 0.2, 0.95))
            if len(draft_pts) >= 2:
                draw_list.add_polyline(draft_pts, poly_col, 1.5, 0)
            for pt in draft_pts:
                draw_list.add_circle_filled(pt, 4.0, poly_col, 12)

    # Subtle axis guide line between start and end handles
    if state.preview_strip and img.wcs is not None:
        s_px = sky_to_pixel(img, state.strip_params.start_ra_deg, state.strip_params.start_dec_deg)
        end_sky = state.get_strip_end_sky()
        e_px = sky_to_pixel(img, end_sky[0], end_sky[1])
        if s_px is not None and e_px is not None:
            p_start = implot.plot_to_pixels(s_px[0], s_px[1])
            p_end = implot.plot_to_pixels(e_px[0], e_px[1])
            axis_col = imgui.get_color_u32(imgui.ImVec4(1.0, 0.85, 0.2, 0.55))
            draw_list.add_line(p_start, p_end, axis_col, 1.2)

    draw_list.pop_clip_rect()

    # Interactive draggable handles (must be called inside begin_plot/end_plot)
    if state.preview_strip and img.wcs is not None:
        s_px = sky_to_pixel(img, state.strip_params.start_ra_deg, state.strip_params.start_dec_deg)
        end_sky = state.get_strip_end_sky()
        e_px = sky_to_pixel(img, end_sky[0], end_sky[1])
        if s_px is not None and e_px is not None:
            ch_s, sx_new, sy_new, _, hov_s, _ = implot.drag_point(
                101, s_px[0], s_px[1], imgui.ImVec4(0.20, 0.95, 1.00, 1.0), 5.0
            )
            ch_e, ex_new, ey_new, _, hov_e, _ = implot.drag_point(
                102, e_px[0], e_px[1], imgui.ImVec4(1.00, 0.85, 0.20, 1.0), 5.0
            )
            if hov_s:
                imgui.set_tooltip("Drag: Move strip START")
            elif hov_e:
                imgui.set_tooltip(
                    f"Drag: Rotate PA={state.strip_params.pa_deg:.1f} deg, N={state.strip_params.count}"
                )
            if ch_s or ch_e:
                new_start = pixel_to_sky(img, sx_new, sy_new)
                new_end = pixel_to_sky(img, ex_new, ey_new)
                if new_start is not None and new_end is not None:
                    state.strip_params = strip_params_from_sky_points(
                        new_start[0],
                        new_start[1],
                        new_end[0],
                        new_end[1],
                        width_arcsec=state.strip_params.width_arcsec,
                        height_arcsec=state.strip_params.height_arcsec,
                        gap_arcsec=state.strip_params.gap_arcsec,
                        prefix=state.strip_params.prefix,
                    )
                    state.strip_end_sky = new_end

    # Center drag handle + single-edge height handles for the currently selected region
    if (
        state.selected_region_idx is not None
        and 0 <= state.selected_region_idx < len(state.regions)
        and img.wcs is not None
    ):
        sel_reg = state.regions[state.selected_region_idx]
        handle_col = imgui.ImVec4(1.00, 0.40, 0.80, 1.0)
        if isinstance(sel_reg, (CircleRegion, BoxRegion, EllipseRegion)):
            c_px = sky_to_pixel(img, sel_reg.ra_deg, sel_reg.dec_deg)
            if c_px is not None:
                ch_c, cx_new, cy_new, _, hov_c, _ = implot.drag_point(
                    200, c_px[0], c_px[1], handle_col, 5.5
                )
                if hov_c:
                    imgui.set_tooltip(f"{sel_reg.name}: drag to move center")
                if ch_c:
                    new_sky = pixel_to_sky(img, cx_new, cy_new)
                    if new_sky is not None:
                        sel_reg.ra_deg, sel_reg.dec_deg = new_sky

        if isinstance(sel_reg, BoxRegion):
            corners = box_corners_pixels(sel_reg, img)
            if corners is not None:
                bot_mid_px = (
                    (corners[0][0] + corners[1][0]) * 0.5,
                    (corners[0][1] + corners[1][1]) * 0.5,
                )
                top_mid_px = (
                    (corners[2][0] + corners[3][0]) * 0.5,
                    (corners[2][1] + corners[3][1]) * 0.5,
                )
                ch_top, tx, ty, _, hov_top, _ = implot.drag_point(
                    201, top_mid_px[0], top_mid_px[1], handle_col, 5.5
                )
                ch_bot, bx, by, _, hov_bot, _ = implot.drag_point(
                    202, bot_mid_px[0], bot_mid_px[1], handle_col, 5.5
                )
                if hov_top or hov_bot:
                    imgui.set_tooltip(
                        f"{sel_reg.name}: drag edge (height={sel_reg.height_arcsec:.1f}\")"
                    )
                if ch_top:
                    sky_pt = pixel_to_sky(img, tx, ty)
                    if sky_pt is not None:
                        resize_box_edge(sel_reg, sky_pt[0], sky_pt[1], edge="top")
                elif ch_bot:
                    sky_pt = pixel_to_sky(img, bx, by)
                    if sky_pt is not None:
                        resize_box_edge(sel_reg, sky_pt[0], sky_pt[1], edge="bottom")


def _draw_image_canvas(state: ImageViewerState) -> None:
    if state.image is None:
        avail = imgui.get_content_region_avail()
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + max(20.0, avail.y * 0.35))
        imgui.text("No FITS image loaded. Click 'Open FITS...' above to begin.")
        return

    img = state.image
    display = state.get_display_flipped()
    if display is None:
        return

    ny, nx = img.shape
    _, cmap = _COLORMAPS[state.colormap_idx]
    avail = imgui.get_content_region_avail()
    cbar_width = 90.0
    plot_width = max(200.0, avail.x - cbar_width - 12.0)
    plot_height = max(200.0, avail.y - 28.0)

    status_text = "Hover over image for pixel/WCS coordinates"
    implot.push_colormap(cmap)
    plot_id = f"##fits_heatmap_{state.view_reset_epoch}"
    if implot.begin_plot(
        plot_id,
        imgui.ImVec2(plot_width, plot_height),
        implot.Flags_.equal | implot.Flags_.no_legend,
    ):
        implot.setup_axes("X (pix)", "Y (pix)")
        implot.setup_axis_limits(
            implot.ImAxis_.x1, -0.5, float(nx) - 0.5, implot.Cond_.once
        )
        implot.setup_axis_limits(
            implot.ImAxis_.y1, -0.5, float(ny) - 0.5, implot.Cond_.once
        )
        implot.plot_heatmap(
            "##heatmap",
            display,
            0.0,
            1.0,
            "",
            implot.Point(-0.5, -0.5),
            implot.Point(float(nx) - 0.5, float(ny) - 0.5),
        )

        _draw_region_overlays(state)

        if implot.is_plot_hovered():
            mouse = implot.get_plot_mouse_pos()
            ix = int(np.floor(mouse.x + 0.5))
            iy = int(np.floor(mouse.y + 0.5))
            if 0 <= ix < nx and 0 <= iy < ny:
                smoothed = state.get_smoothed_data()
                val = float(smoothed[iy, ix]) if smoothed is not None else float("nan")
                sky = pixel_to_sky(img, mouse.x, mouse.y)
                if sky is not None:
                    status_text = (
                        f"x={mouse.x:.1f}, y={mouse.y:.1f}  |  "
                        f"RA={sky[0]:.5f}, Dec={sky[1]:.5f}  |  val={val:.4g}"
                    )
                else:
                    status_text = f"x={mouse.x:.1f}, y={mouse.y:.1f}  |  val={val:.4g}"

                if state.placing_region and imgui.is_mouse_clicked(0) and sky is not None:
                    if state.add_shape_idx == 3:
                        state.poly_draft_vertices.append(sky)
                    else:
                        state.add_manual_region_at(sky[0], sky[1])
                        state.placing_region = False

        implot.end_plot()
    implot.pop_colormap()

    imgui.same_line()
    implot.colormap_scale(
        "##colorbar",
        state.vmin,
        state.vmax,
        imgui.ImVec2(cbar_width - 10.0, plot_height),
        "%.3g",
        0,
        cmap,
    )
    imgui.text(status_text)


def run_image_gui(initial_file: str | Path | None = None) -> None:
    """Launch the Dear ImGui FITS Flux Image & Region Generator GUI."""
    state = ImageViewerState()
    if initial_file is not None:
        state.open_file(initial_file)

    def gui() -> None:
        _draw_toolbar(state)
        imgui.separator()

        avail = imgui.get_content_region_avail()
        sidebar_width = min(360.0, max(260.0, avail.x * 0.28))

        if imgui.begin_child(
            "##img_sidebar", imgui.ImVec2(sidebar_width, 0.0), imgui.ChildFlags_.borders
        ):
            _draw_sidebar(state)
        imgui.end_child()

        imgui.same_line()

        if imgui.begin_child("##img_canvas", imgui.ImVec2(0.0, 0.0)):
            _draw_image_canvas(state)
        imgui.end_child()

    immapp.run(
        gui,
        with_implot=True,
        window_title="XMM-Newton Flux Image & Region Tool",
        window_size=(1400, 880),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``xmm-img`` / ``python -m xmm_tools.image``."""
    parser = argparse.ArgumentParser(
        description="Interactive ImGui FITS flux image viewer & DS9 region generator."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Optional path to a .fits image to open on startup.",
    )
    args = parser.parse_args(argv)
    run_image_gui(args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
