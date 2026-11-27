"""Standalone Dear ImGui + ImPlot GUI and CLI for XMM-Newton cheese source merging."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from xmm_tools.image.fits_io import (
    FitsImageData,
    load_fits_image,
    smooth_image,
    stretch_image,
)
from xmm_tools.srclist.io import (
    export_merged_ds9_reg,
    export_merged_emllist,
    load_cheese_epoch,
)
from xmm_tools.srclist.merge import (
    angular_separation_arcsec,
    filter_detections,
    merge_catalogs,
)
from xmm_tools.srclist.models import (
    EpochCatalog,
    FilterParams,
    MergedSource,
    OverlayEllipse,
)

_STRETCH_MODES: list[str] = ["asinh", "log10", "sqrt", "linear"]
_ELLIPSE_INST_OPTIONS: list[str] = ["All", "MOS1", "MOS2", "PN"]

# Distinct RGBA colors per epoch (cycling when > 6 epochs).
_EPOCH_PALETTE: list[tuple[float, float, float, float]] = [
    (0.25, 0.88, 1.00, 0.95),  # cyan
    (1.00, 0.78, 0.20, 0.95),  # gold
    (0.82, 0.45, 1.00, 0.95),  # violet
    (1.00, 0.48, 0.35, 0.95),  # coral
    (0.45, 0.95, 0.65, 0.95),  # mint
    (0.55, 0.75, 1.00, 0.95),  # sky blue
]


def _sky_offsets_to_pixels_wcs(
    ra_deg: float,
    dec_deg: float,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    angle_deg: float,
    wcs: WCS,
) -> list[tuple[float, float]] | None:
    center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5")
    alpha = math.radians(angle_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    d_east = -x_arcsec * ca + y_arcsec * sa
    d_north = x_arcsec * sa + y_arcsec * ca
    sep = np.hypot(d_east, d_north) * u.arcsec
    pa = np.arctan2(d_east, d_north) * u.rad
    pts = center.directional_offset_by(pa, sep)
    xs, ys = wcs.wcs_world2pix(pts.ra.deg, pts.dec.deg, 0)
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _ellipse_outline_pixels_wcs(
    ell: OverlayEllipse, wcs: WCS, *, n_pts: int = 40
) -> list[tuple[float, float]] | None:
    theta = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    return _sky_offsets_to_pixels_wcs(
        ell.ra_deg,
        ell.dec_deg,
        ell.semi_major_arcsec * np.cos(theta),
        ell.semi_minor_arcsec * np.sin(theta),
        ell.angle_deg,
        wcs,
    )


def _wcs_pixel_to_sky(wcs: WCS, x_pix: float, y_pix: float) -> tuple[float, float]:
    sky = wcs.pixel_to_world(x_pix, y_pix)
    return float(sky.ra.deg), float(sky.dec.deg)


def _wcs_sky_to_pixel(
    wcs: WCS, ra_deg: float, dec_deg: float
) -> tuple[float, float] | None:
    xs, ys = wcs.wcs_world2pix([ra_deg], [dec_deg], 0)
    x_val = float(xs[0])
    y_val = float(ys[0])
    if not (math.isfinite(x_val) and math.isfinite(y_val)):
        return None
    return x_val, y_val


@dataclass
class SrclistViewerState:
    """Testable, headless-safe application state for ``xmm-srclist``."""

    epochs: list[EpochCatalog] = field(default_factory=list)
    filter_params: FilterParams = field(default_factory=FilterParams)
    match_radius_arcsec: float = 5.0
    merged_sources: list[MergedSource] = field(default_factory=list)
    manual_sources: list[MergedSource] = field(default_factory=list)
    excluded_detection_keys: set[tuple[str, int]] = field(default_factory=set)
    _next_manual_id: int = 1
    selected_source_idx: int | None = None
    placing_manual_source: bool = False

    bg_image: FitsImageData | None = None
    bg_path_input: str = ""
    epoch_path_input: str = ""
    error_message: str = ""
    status_message: str = ""

    # Display & overlay options
    colormap_idx: int = 0
    stretch_idx: int = 0
    smooth_arcsec: float = 0.0
    vmin: float = 0.0
    vmax: float = 1.0
    auto_limits: bool = True
    manual_limits: bool = False
    view_reset_epoch: int = 0

    show_epoch_detections: bool = True
    show_merged_sources: bool = True
    show_sky_ellipses: bool = False
    ellipse_instrument_idx: int = 0
    manual_radius_arcsec: float = 15.0
    export_radius_arcsec: float = 15.0

    # Smoothing & display cache for background FITS
    _cached_smooth_arcsec: float = -1.0
    _smoothed_data: np.ndarray | None = None
    _cached_display_key: tuple[Any, ...] | None = None
    _display_flipped: np.ndarray | None = None

    # Synthetic WCS cache when no background image is loaded
    _synthetic_wcs: WCS | None = None
    _synthetic_dims: tuple[int, int] = (1800, 1800)

    # Native file dialogs (created lazily in GUI)
    add_epoch_dir_dialog: Any = None
    add_epoch_file_dialog: Any = None
    open_bg_dialog: Any = None
    save_fits_dialog: Any = None
    save_reg_dialog: Any = None

    def add_epoch(
        self, path: str | Path, *, label: str | None = None
    ) -> EpochCatalog | None:
        """Load and append an epoch from a cheese dir, ``ana.toml``, or FITS."""
        try:
            catalog = load_cheese_epoch(path, label=label)
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"Failed to load {path}: {exc}"
            return None

        # Ensure unique epoch_id if the same path is loaded twice.
        existing_ids = {ep.epoch_id for ep in self.epochs}
        if catalog.epoch_id in existing_ids:
            suffix = 2
            while f"{catalog.epoch_id}#{suffix}" in existing_ids:
                suffix += 1
            catalog.epoch_id = f"{catalog.epoch_id}#{suffix}"
            for det in catalog.detections:
                det.epoch_id = catalog.epoch_id

        self.epochs.append(catalog)
        self.error_message = ""
        self.status_message = (
            f"Loaded {catalog.label}: {len(catalog.detections)} detections"
        )
        self._invalidate_synthetic_wcs()
        self.recompute_merged()
        if len(self.epochs) == 1 and self.bg_image is None:
            self.view_reset_epoch += 1
        return catalog

    def remove_epoch(self, index: int) -> None:
        """Remove an epoch by index and recompute merged sources."""
        if not (0 <= index < len(self.epochs)):
            return
        removed = self.epochs.pop(index)
        self.excluded_detection_keys = {
            key for key in self.excluded_detection_keys if key[0] != removed.epoch_id
        }
        self.status_message = f"Removed epoch {removed.label}"
        self._invalidate_synthetic_wcs()
        self.recompute_merged()

    def set_epoch_enabled(self, index: int, enabled: bool) -> None:
        """Enable or disable an epoch and refresh the merged source list."""
        if not (0 <= index < len(self.epochs)):
            return
        self.epochs[index].enabled = enabled
        self.recompute_merged()

    def recompute_merged(self) -> list[MergedSource]:
        """Re-run filtering and cross-epoch clustering, keeping selection if valid."""
        prev_manual_id: int | None = None
        prev_member_keys: set[tuple[str, int]] = set()
        prev_coord: tuple[float, float] | None = None
        if self.selected_source_idx is not None and 0 <= self.selected_source_idx < len(
            self.merged_sources
        ):
            sel = self.merged_sources[self.selected_source_idx]
            prev_coord = (sel.ra_deg, sel.dec_deg)
            if sel.is_manual:
                prev_manual_id = sel.manual_id
            else:
                prev_member_keys = {(m.epoch_id, m.ml_id_src) for m in sel.members}

        self.merged_sources = merge_catalogs(
            self.epochs,
            filter_params=self.filter_params,
            match_radius_arcsec=self.match_radius_arcsec,
            excluded_detection_keys=self.excluded_detection_keys,
            manual_sources=self.manual_sources,
        )

        if prev_manual_id is not None:
            self.selected_source_idx = next(
                (
                    idx
                    for idx, src in enumerate(self.merged_sources)
                    if src.is_manual and src.manual_id == prev_manual_id
                ),
                None,
            )
        elif prev_member_keys and self.merged_sources:
            best_idx: int | None = None
            best_overlap = 0
            for idx, src in enumerate(self.merged_sources):
                if src.is_manual:
                    continue
                overlap = sum(
                    1
                    for m in src.members
                    if (m.epoch_id, m.ml_id_src) in prev_member_keys
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            self.selected_source_idx = best_idx
        elif prev_coord is not None and self.merged_sources:
            tol = max(self.match_radius_arcsec, 2.0)
            best_idx = None
            best_sep = tol
            for idx, src in enumerate(self.merged_sources):
                sep = angular_separation_arcsec(
                    prev_coord[0], prev_coord[1], src.ra_deg, src.dec_deg
                )
                if sep <= best_sep:
                    best_sep = sep
                    best_idx = idx
            self.selected_source_idx = best_idx
        elif (
            self.selected_source_idx is not None
            and self.selected_source_idx >= len(self.merged_sources)
        ):
            self.selected_source_idx = None

        return self.merged_sources

    def toggle_source_include(self, index: int) -> None:
        """Toggle ``include`` on ``merged_sources[index]`` and persist exclusion."""
        if not (0 <= index < len(self.merged_sources)):
            return
        src = self.merged_sources[index]
        new_state = not src.include
        src.include = new_state

        if src.is_manual:
            for msrc in self.manual_sources:
                if (
                    src.manual_id is not None and msrc.manual_id == src.manual_id
                ) or (
                    src.manual_id is None
                    and angular_separation_arcsec(
                        src.ra_deg, src.dec_deg, msrc.ra_deg, msrc.dec_deg
                    )
                    <= 1e-3
                ):
                    msrc.include = new_state
            return

        if not new_state:
            for m in src.members:
                self.excluded_detection_keys.add((m.epoch_id, m.ml_id_src))
        else:
            for m in src.members:
                self.excluded_detection_keys.discard((m.epoch_id, m.ml_id_src))

    def set_all_included(self, include: bool) -> None:
        """Include or exclude all current merged sources."""
        if include:
            self.excluded_detection_keys.clear()
        for src in self.merged_sources:
            src.include = include
            if not include and not src.is_manual:
                for m in src.members:
                    self.excluded_detection_keys.add((m.epoch_id, m.ml_id_src))
        for msrc in self.manual_sources:
            msrc.include = include

    def add_manual_source(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        det_ml: float = 50.0,
        scts: float = 100.0,
        bg_map: float = 0.5,
    ) -> MergedSource:
        """Add a user-defined manual source and re-merge."""
        if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
            raise ValueError("Manual source RA and Dec must be finite")
        scts_val = float(scts) if (math.isfinite(scts) and scts > 0.0) else 1.0
        bg_val = float(bg_map) if (math.isfinite(bg_map) and bg_map > 0.0) else 0.1
        manual_id = self._next_manual_id
        self._next_manual_id += 1
        manual = MergedSource(
            merged_id=len(self.merged_sources) + 1,
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            radec_err_arcsec=1.0,
            det_ml=float(det_ml),
            flux=1.0e-14,
            scts=scts_val,
            bg_map=bg_val,
            ext=0.0,
            ext_ml=0.0,
            dist_nn_arcsec=999.0,
            include=True,
            is_manual=True,
            manual_id=manual_id,
            members=[],
            representative_row={},
        )
        self.manual_sources.append(manual)
        self._invalidate_synthetic_wcs()
        self.recompute_merged()
        for idx, src in enumerate(self.merged_sources):
            if src.is_manual and src.manual_id == manual_id:
                self.selected_source_idx = idx
                break
        self.status_message = (
            f"Added manual source at ({ra_deg:.5f}, {dec_deg:.5f})"
        )
        return manual

    def remove_manual_source(self, merged_index: int) -> None:
        """Remove a manual source corresponding to ``merged_sources[merged_index]``."""
        if not (0 <= merged_index < len(self.merged_sources)):
            return
        target = self.merged_sources[merged_index]
        if not target.is_manual:
            return
        for i, msrc in enumerate(self.manual_sources):
            if (
                target.manual_id is not None and msrc.manual_id == target.manual_id
            ) or (
                target.manual_id is None
                and angular_separation_arcsec(
                    target.ra_deg, target.dec_deg, msrc.ra_deg, msrc.dec_deg
                )
                <= 1e-4
            ):
                self.manual_sources.pop(i)
                break
        self.selected_source_idx = None
        self._invalidate_synthetic_wcs()
        self.recompute_merged()

    def open_background_fits(self, path: str | Path) -> None:
        """Load an optional background FITS image with WCS."""
        try:
            loaded = load_fits_image(path)
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"Failed to load background FITS {path}: {exc}"
            return
        if loaded.wcs is None:
            self.error_message = f"Background FITS has no celestial WCS: {path}"
            return
        self.bg_image = loaded
        self.bg_path_input = str(loaded.path) if loaded.path else ""
        self.error_message = ""
        self._cached_smooth_arcsec = -1.0
        self._smoothed_data = None
        self._cached_display_key = None
        self._display_flipped = None
        self.auto_limits = True
        self.manual_limits = False
        self.view_reset_epoch += 1
        self.status_message = f"Loaded background image {loaded.path}"

    def clear_background_fits(self) -> None:
        """Unload background FITS image and revert to synthetic catalog WCS."""
        self.bg_image = None
        self.bg_path_input = ""
        self._cached_smooth_arcsec = -1.0
        self._smoothed_data = None
        self._cached_display_key = None
        self._display_flipped = None
        self._invalidate_synthetic_wcs()
        self.view_reset_epoch += 1

    def candidate_background_images(self) -> list[Path]:
        """Collect deduplicated candidate background FITS files across epochs."""
        found: list[Path] = []
        seen: set[Path] = set()
        for ep in self.epochs:
            for cand in ep.candidate_images:
                resolved = cand.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(cand)
        return found

    def apply_cheese_defaults(self) -> None:
        """Reset filter thresholds to ESAS ``cheese`` defaults."""
        self.filter_params.min_det_ml = 15.0
        self.filter_params.min_flux = 0.2e-14
        self.filter_params.min_dist_nn_arcsec = 0.0
        self.filter_params.exclude_extended = False
        self.recompute_merged()

    def apply_show_all(self) -> None:
        """Relax filter thresholds to display all catalog entries."""
        self.filter_params.min_det_ml = 0.0
        self.filter_params.min_flux = 0.0
        self.filter_params.min_dist_nn_arcsec = 0.0
        self.filter_params.exclude_extended = False
        self.recompute_merged()

    def template_catalog(self) -> EpochCatalog | None:
        """Return first enabled epoch (or first loaded epoch) for export metadata."""
        for ep in self.epochs:
            if ep.enabled:
                return ep
        return self.epochs[0] if self.epochs else None

    def export_fits(self, out_path: str | Path) -> Path:
        """Export included merged sources as a SAS-compatible ``merged_emllist.fits``."""
        written = export_merged_emllist(
            self.merged_sources,
            out_path,
            template_catalog=self.template_catalog(),
            include_only=True,
        )
        self.error_message = ""
        self.status_message = f"Exported FITS: {written}"
        return written

    def export_reg(self, out_path: str | Path) -> Path:
        """Export included merged sources as a DS9 ``.reg`` file."""
        written = export_merged_ds9_reg(
            self.merged_sources,
            out_path,
            default_radius_arcsec=self.export_radius_arcsec,
            include_excluded=False,
        )
        self.error_message = ""
        self.status_message = f"Exported DS9 region: {written}"
        return written

    def _invalidate_synthetic_wcs(self) -> None:
        self._synthetic_wcs = None

    def get_active_wcs(self) -> tuple[WCS, int, int, float] | None:
        """Return ``(wcs, nx, ny, pixel_scale_arcsec)`` for canvas rendering.

        Uses ``bg_image.wcs`` when a background FITS image is loaded, or a
        synthetic North-up, East-left ``TAN`` WCS (``1.0"/pix``) centered on
        loaded sources when in pure catalog mode.
        """
        if self.bg_image is not None and self.bg_image.wcs is not None:
            ny, nx = self.bg_image.shape
            return (
                self.bg_image.wcs,
                nx,
                ny,
                max(self.bg_image.pixel_scale_arcsec, 1e-6),
            )

        if self._synthetic_wcs is not None:
            nx, ny = self._synthetic_dims
            return self._synthetic_wcs, nx, ny, 1.0

        coords: list[tuple[float, float]] = []
        for ep in self.epochs:
            for det in ep.detections:
                coords.append((det.ra_deg, det.dec_deg))
        for msrc in self.manual_sources:
            coords.append((msrc.ra_deg, msrc.dec_deg))

        if not coords:
            return None

        ra_arr = np.array([c[0] for c in coords], dtype=np.float64)
        dec_arr = np.array([c[1] for c in coords], dtype=np.float64)
        # Unit-vector mean avoids 0/360 RA wraparound issues.
        ra_rad = np.radians(ra_arr)
        dec_rad = np.radians(dec_arr)
        mx = float(np.mean(np.cos(dec_rad) * np.cos(ra_rad)))
        my = float(np.mean(np.cos(dec_rad) * np.sin(ra_rad)))
        mz = float(np.mean(np.sin(dec_rad)))
        norm = math.sqrt(mx * mx + my * my + mz * mz)
        if norm > 0.0:
            center_ra = math.degrees(math.atan2(my, mx)) % 360.0
            center_dec = math.degrees(math.asin(max(-1.0, min(1.0, mz / norm))))
        else:
            center_ra, center_dec = float(ra_arr[0]), float(dec_arr[0])

        max_sep = max(
            (
                angular_separation_arcsec(center_ra, center_dec, r, d)
                for r, d in coords
            ),
            default=900.0,
        )
        span_pix = max(1800, math.ceil(max_sep * 2.4) + 120)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crpix = [span_pix * 0.5 + 0.5, span_pix * 0.5 + 0.5]
        wcs.wcs.crval = [center_ra, center_dec]
        wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]

        self._synthetic_wcs = wcs
        self._synthetic_dims = (span_pix, span_pix)
        return wcs, span_pix, span_pix, 1.0

    def get_smoothed_data(self) -> np.ndarray | None:
        """Return smoothed background image array (cached)."""
        if self.bg_image is None:
            return None
        if (
            self._smoothed_data is None
            or abs(self._cached_smooth_arcsec - self.smooth_arcsec) > 1e-6
        ):
            sigma_pix = self.smooth_arcsec / max(self.bg_image.pixel_scale_arcsec, 1e-6)
            self._smoothed_data = smooth_image(self.bg_image.data, sigma_pix)
            self._cached_smooth_arcsec = self.smooth_arcsec
            self._cached_display_key = None
            if not self.manual_limits:
                self.auto_limits = True
        return self._smoothed_data

    def get_display_flipped(self) -> np.ndarray | None:
        """Return vertically flipped stretched array for ``implot.plot_heatmap``."""
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


def _epoch_kept_count(ep: EpochCatalog, params: FilterParams) -> int:
    return len(filter_detections(ep.detections, params))


def run_srclist_gui(
    initial_paths: Sequence[str | Path] = (),
    *,
    bg_fits: str | Path | None = None,
    match_radius_arcsec: float = 5.0,
    min_det_ml: float = 15.0,
    min_flux: float = 0.2e-14,
    min_dist_nn_arcsec: float = 0.0,
) -> None:
    """Launch the interactive Dear ImGui + ImPlot cheese source-list merger GUI."""
    from imgui_bundle import (
        imgui,
        immapp,
        implot,
    )
    from imgui_bundle import (
        portable_file_dialogs as pfd,
    )

    colormaps: list[tuple[str, int]] = [
        ("Viridis", implot.Colormap_.viridis),
        ("Plasma", implot.Colormap_.plasma),
        ("Inferno", implot.Colormap_.hot),
        ("Greys", implot.Colormap_.greys),
        ("Jet", implot.Colormap_.jet),
        ("Cool", implot.Colormap_.cool),
        ("Spectral", implot.Colormap_.spectral),
    ]

    state = SrclistViewerState(
        filter_params=FilterParams(
            min_det_ml=min_det_ml,
            min_flux=min_flux,
            min_dist_nn_arcsec=min_dist_nn_arcsec,
        ),
        match_radius_arcsec=match_radius_arcsec,
    )
    for p in initial_paths:
        state.add_epoch(p)
    if bg_fits is not None:
        state.open_background_fits(bg_fits)

    def _draw_toolbar() -> None:
        imgui.set_next_item_width(320.0)
        ch_ep, new_ep = imgui.input_text("##epoch_path", state.epoch_path_input)
        if ch_ep:
            state.epoch_path_input = new_ep
        imgui.same_line()
        if imgui.button("Add Cheese Dir / emllist") and state.epoch_path_input.strip():
            state.add_epoch(state.epoch_path_input.strip())
        imgui.same_line()
        if imgui.button("Browse Dir..."):
            state.add_epoch_dir_dialog = pfd.select_folder(
                "Select Cheese / Observation Directory",
                state.epoch_path_input or "",
            )
        imgui.same_line()
        if imgui.button("Browse File..."):
            state.add_epoch_file_dialog = pfd.open_file(
                "Select emllist.fits or ana.toml",
                state.epoch_path_input or "",
                [
                    "FITS or TOML (*.fits *.fit *.toml)",
                    "*.fits *.fit *.toml",
                    "All Files",
                    "*",
                ],
            )
        imgui.same_line()
        if imgui.button("Open Bg FITS..."):
            state.open_bg_dialog = pfd.open_file(
                "Open Background FITS Image",
                str(Path(state.bg_path_input).parent) if state.bg_path_input else "",
                ["FITS Files (*.fits *.fit)", "*.fits *.fit", "All Files", "*"],
            )
        cands = state.candidate_background_images()
        if cands:
            imgui.same_line()
            imgui.set_next_item_width(170.0)
            if imgui.begin_combo("##cand_bg", "Quick Bg FITS"):
                for idx, cand in enumerate(cands):
                    label = f"{cand.parent.name}/{cand.name}##cand_{idx}"
                    selected = (
                        state.bg_image is not None
                        and state.bg_image.path is not None
                        and state.bg_image.path.resolve() == cand.resolve()
                    )
                    clicked, _ = imgui.selectable(label, selected)
                    if clicked:
                        state.open_background_fits(cand)
                imgui.end_combo()
        if state.bg_image is not None:
            imgui.same_line()
            if imgui.button("Clear Bg"):
                state.clear_background_fits()

        imgui.same_line()
        if imgui.button("Export merged_emllist.fits..."):
            state.save_fits_dialog = pfd.save_file(
                "Export Merged emllist.fits",
                "merged_emllist.fits",
                ["FITS Files (*.fits)", "*.fits"],
            )
        imgui.same_line()
        if imgui.button("Export DS9 .reg..."):
            state.save_reg_dialog = pfd.save_file(
                "Export DS9 Region File",
                "merged_sources.reg",
                ["DS9 Regions (*.reg)", "*.reg"],
            )
        imgui.same_line()
        if imgui.button("Reset View"):
            state.view_reset_epoch += 1

        # Poll file dialogs
        if (
            state.add_epoch_dir_dialog is not None
            and state.add_epoch_dir_dialog.ready()
        ):
            sel = state.add_epoch_dir_dialog.result()
            state.add_epoch_dir_dialog = None
            if sel:
                state.epoch_path_input = sel
                state.add_epoch(sel)

        if (
            state.add_epoch_file_dialog is not None
            and state.add_epoch_file_dialog.ready()
        ):
            sel = state.add_epoch_file_dialog.result()
            state.add_epoch_file_dialog = None
            if sel:
                state.epoch_path_input = sel[0]
                state.add_epoch(sel[0])

        if state.open_bg_dialog is not None and state.open_bg_dialog.ready():
            sel = state.open_bg_dialog.result()
            state.open_bg_dialog = None
            if sel:
                state.open_background_fits(sel[0])

        if state.save_fits_dialog is not None and state.save_fits_dialog.ready():
            sel = state.save_fits_dialog.result()
            state.save_fits_dialog = None
            if sel:
                try:
                    state.export_fits(sel)
                except Exception as exc:  # noqa: BLE001
                    state.error_message = f"Failed to export FITS: {exc}"

        if state.save_reg_dialog is not None and state.save_reg_dialog.ready():
            sel = state.save_reg_dialog.result()
            state.save_reg_dialog = None
            if sel:
                try:
                    state.export_reg(sel)
                except Exception as exc:  # noqa: BLE001
                    state.error_message = f"Failed to export .reg: {exc}"

        if state.error_message:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.35, 1.0), state.error_message)
        elif state.status_message:
            imgui.text_disabled(state.status_message)

    def _draw_sidebar() -> None:
        # 1. Epochs / Catalogs
        if imgui.collapsing_header(
            f"Epochs / Catalogs ({len(state.epochs)})",
            imgui.TreeNodeFlags_.default_open,
        ):
            if not state.epochs:
                imgui.text_disabled("No epochs loaded. Add a cheese dir or emllist.")
            else:
                to_remove: int | None = None
                for idx, ep in enumerate(state.epochs):
                    col = _EPOCH_PALETTE[idx % len(_EPOCH_PALETTE)]
                    imgui.color_button(
                        f"##ep_col_{idx}",
                        imgui.ImVec4(*col),
                        imgui.ColorEditFlags_.no_tooltip,
                        imgui.ImVec2(12.0, 12.0),
                    )
                    imgui.same_line()
                    ch_en, new_en = imgui.checkbox(f"##ep_en_{idx}", ep.enabled)
                    if ch_en:
                        state.set_epoch_enabled(idx, new_en)
                    imgui.same_line()
                    kept = _epoch_kept_count(ep, state.filter_params)
                    imgui.text(f"{ep.label} ({kept}/{len(ep.detections)})")
                    imgui.same_line()
                    if imgui.small_button(f"X##ep_rm_{idx}"):
                        to_remove = idx
                if to_remove is not None:
                    state.remove_epoch(to_remove)

        # 2. Filter & Cross-Match Controls
        if imgui.collapsing_header(
            "Filter & Cross-Match", imgui.TreeNodeFlags_.default_open
        ):
            ch_mr, new_mr = imgui.slider_float(
                "Match Radius (\")",
                state.match_radius_arcsec,
                0.0,
                30.0,
                "%.1f (0=no merge)" if state.match_radius_arcsec <= 0.0 else "%.1f",
            )
            if ch_mr:
                state.match_radius_arcsec = new_mr
                state.recompute_merged()

            fp = state.filter_params
            ch_ml, new_ml = imgui.slider_float(
                "Min DET_ML", fp.min_det_ml, 0.0, 200.0, "%.1f"
            )
            if ch_ml:
                fp.min_det_ml = new_ml
                state.recompute_merged()

            flux_1e14 = fp.min_flux * 1e14
            ch_fl, new_fl = imgui.slider_float(
                "Min Flux (1e-14)", flux_1e14, 0.0, 10.0, "%.2f"
            )
            if ch_fl:
                fp.min_flux = new_fl * 1e-14
                state.recompute_merged()

            ch_dn, new_dn = imgui.slider_float(
                "Min DIST_NN (\")", fp.min_dist_nn_arcsec, 0.0, 100.0, "%.1f"
            )
            if ch_dn:
                fp.min_dist_nn_arcsec = new_dn
                state.recompute_merged()

            ch_ex, new_ex = imgui.checkbox(
                "Exclude Extended (EXT>0, EXT_ML>=6)", fp.exclude_extended
            )
            if ch_ex:
                fp.exclude_extended = new_ex
                state.recompute_merged()

            if imgui.button("Cheese Defaults"):
                state.apply_cheese_defaults()
            imgui.same_line()
            if imgui.button("Show All"):
                state.apply_show_all()

        # 3. Overlay & Background Controls
        if imgui.collapsing_header(
            "Overlays & Background", imgui.TreeNodeFlags_.default_open
        ):
            _, state.show_epoch_detections = imgui.checkbox(
                "Show Epoch Detections", state.show_epoch_detections
            )
            _, state.show_merged_sources = imgui.checkbox(
                "Show Merged Sources", state.show_merged_sources
            )
            _, state.show_sky_ellipses = imgui.checkbox(
                "Show Cheese Sky Ellipses", state.show_sky_ellipses
            )
            if state.show_sky_ellipses:
                imgui.set_next_item_width(140.0)
                _, state.ellipse_instrument_idx = imgui.combo(
                    "Ellipse Inst",
                    state.ellipse_instrument_idx,
                    _ELLIPSE_INST_OPTIONS,
                )

            if state.bg_image is not None:
                imgui.separator_text("Background Display")
                cmap_names = [name for name, _ in colormaps]
                _, state.colormap_idx = imgui.combo(
                    "Colormap", state.colormap_idx, cmap_names
                )
                _, state.stretch_idx = imgui.combo(
                    "Stretch", state.stretch_idx, _STRETCH_MODES
                )
                _, state.smooth_arcsec = imgui.slider_float(
                    "Smooth (\")", state.smooth_arcsec, 0.0, 30.0, "%.1f"
                )
                if imgui.button("Auto Scale"):
                    state.auto_limits = True
                    state.manual_limits = False
                imgui.same_line()
                imgui.text_disabled(f"[{state.vmin:.3g}, {state.vmax:.3g}]")
                step = max(abs(state.vmax - state.vmin) * 0.005, 1e-6)
                ch_min, new_min = imgui.drag_float(
                    "vmin", state.vmin, step, format="%.4g"
                )
                ch_max, new_max = imgui.drag_float(
                    "vmax", state.vmax, step, format="%.4g"
                )
                if ch_min:
                    state.vmin = new_min
                    state.manual_limits = True
                if ch_max:
                    state.vmax = max(new_max, state.vmin + 1e-12)
                    state.manual_limits = True

        # 4. Merged Sources List & Inspector
        n_inc = sum(1 for s in state.merged_sources if s.include)
        n_tot = len(state.merged_sources)
        if imgui.collapsing_header(
            f"Merged Sources ({n_inc}/{n_tot} included)",
            imgui.TreeNodeFlags_.default_open,
        ):
            if imgui.button("Include All"):
                state.set_all_included(True)
            imgui.same_line()
            if imgui.button("Exclude All"):
                state.set_all_included(False)

            if state.placing_manual_source:
                if imgui.button("Cancel Manual Add"):
                    state.placing_manual_source = False
                imgui.same_line()
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.85, 0.2, 1.0), "Click canvas to add"
                )
            else:
                if imgui.button("Click Image to Add Manual Source"):
                    state.placing_manual_source = True

            if state.merged_sources:
                tbl_flags = (
                    imgui.TableFlags_.borders
                    | imgui.TableFlags_.row_bg
                    | imgui.TableFlags_.scroll_y
                    | imgui.TableFlags_.sizing_fixed_fit
                )
                tbl_h = min(260.0, max(100.0, len(state.merged_sources) * 23.0 + 26.0))
                if imgui.begin_table("##merged_tbl", 6, tbl_flags, imgui.ImVec2(0, tbl_h)):
                    imgui.table_setup_column("Inc", imgui.TableColumnFlags_.width_fixed, 28)
                    imgui.table_setup_column("ID", imgui.TableColumnFlags_.width_fixed, 32)
                    imgui.table_setup_column("RA", imgui.TableColumnFlags_.width_fixed, 70)
                    imgui.table_setup_column("Dec", imgui.TableColumnFlags_.width_fixed, 70)
                    imgui.table_setup_column("ML", imgui.TableColumnFlags_.width_fixed, 48)
                    imgui.table_setup_column("N", imgui.TableColumnFlags_.width_fixed, 42)
                    imgui.table_headers_row()

                    for idx, src in enumerate(state.merged_sources):
                        imgui.table_next_row()
                        imgui.table_set_column_index(0)
                        ch_inc, _ = imgui.checkbox(f"##inc_{idx}", src.include)
                        if ch_inc:
                            state.toggle_source_include(idx)

                        imgui.table_set_column_index(1)
                        is_sel = state.selected_source_idx == idx
                        sel_flags = (
                            imgui.SelectableFlags_.span_all_columns
                            | imgui.SelectableFlags_.allow_overlap
                        )
                        clicked, _ = imgui.selectable(
                            f"S{src.merged_id}##row_{idx}", is_sel, sel_flags
                        )
                        if clicked:
                            state.selected_source_idx = idx

                        imgui.table_set_column_index(2)
                        imgui.text(f"{src.ra_deg:.4f}")
                        imgui.table_set_column_index(3)
                        imgui.text(f"{src.dec_deg:.4f}")
                        imgui.table_set_column_index(4)
                        imgui.text(f"{src.det_ml:.1f}")
                        imgui.table_set_column_index(5)
                        tag = "man" if src.is_manual else f"{len(src.members)}"
                        imgui.text(tag)
                    imgui.end_table()

            # Inspector for selected source
            sel_idx = state.selected_source_idx
            if sel_idx is not None and 0 <= sel_idx < len(state.merged_sources):
                sel = state.merged_sources[sel_idx]
                imgui.separator_text(f"Inspector: S{sel.merged_id}")
                imgui.text(
                    f"RA={sel.ra_deg:.6f}  Dec={sel.dec_deg:.6f}  "
                    f"err={sel.radec_err_arcsec:.2f}\""
                )
                imgui.text(
                    f"ML={sel.det_ml:.2f}  Flux={sel.flux:.3e}  "
                    f"DIST_NN={sel.dist_nn_arcsec:.1f}\""
                )
                imgui.text(f"SCTS={sel.scts:.1f}  BG_MAP={sel.bg_map:.4f}")
                if sel.is_manual:
                    if imgui.button("Delete Manual Source"):
                        state.remove_manual_source(sel_idx)
                elif sel.members:
                    imgui.text_disabled(f"Members ({len(sel.members)}):")
                    for m in sel.members:
                        imgui.text_disabled(
                            f"  [{m.epoch_label}] #{m.ml_id_src} "
                            f"ML={m.det_ml:.1f} Flux={m.flux:.2e} BG={m.bg_map:.3f}"
                        )

    def _draw_canvas() -> None:
        geom = state.get_active_wcs()
        if geom is None:
            avail = imgui.get_content_region_avail()
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + max(20.0, avail.y * 0.35))
            imgui.text(
                "Add a cheese directory or emllist.fits above to visualize & merge."
            )
            return

        wcs, nx, ny, pix_scale = geom
        has_bg = state.bg_image is not None
        display = state.get_display_flipped() if has_bg else None
        _, cmap = colormaps[state.colormap_idx]

        avail = imgui.get_content_region_avail()
        cbar_width = 90.0 if has_bg else 0.0
        plot_width = max(200.0, avail.x - cbar_width - (12.0 if has_bg else 4.0))
        plot_height = max(200.0, avail.y - 28.0)

        status_text = (
            "Left-click: select | Right-click / Double-click / Space: toggle mask"
        )
        if has_bg:
            implot.push_colormap(cmap)

        plot_flags = (
            implot.Flags_.equal | implot.Flags_.no_legend | implot.Flags_.no_menus
        )
        plot_id = f"##srclist_canvas_{state.view_reset_epoch}"
        if implot.begin_plot(plot_id, imgui.ImVec2(plot_width, plot_height), plot_flags):
            implot.setup_axes("X (pix, East <-)", "Y (pix, North ^)")
            implot.setup_axis_limits(
                implot.ImAxis_.x1, -0.5, float(nx) - 0.5, implot.Cond_.once
            )
            implot.setup_axis_limits(
                implot.ImAxis_.y1, -0.5, float(ny) - 0.5, implot.Cond_.once
            )

            if display is not None:
                implot.plot_heatmap(
                    "##bg",
                    display,
                    0.0,
                    1.0,
                    "",
                    implot.Point(-0.5, -0.5),
                    implot.Point(float(nx) - 0.5, float(ny) - 0.5),
                )

            draw_list = imgui.get_window_draw_list()
            plot_pos = implot.get_plot_pos()
            plot_size = implot.get_plot_size()
            draw_list.push_clip_rect(
                plot_pos,
                imgui.ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y),
                True,
            )

            # 1. Cheese sky ellipses
            if state.show_sky_ellipses:
                inst_filter = _ELLIPSE_INST_OPTIONS[state.ellipse_instrument_idx].lower()
                ell_col = imgui.get_color_u32(imgui.ImVec4(1.0, 0.72, 0.25, 0.55))
                for ep in state.epochs:
                    if not ep.enabled:
                        continue
                    for inst_name, ell_list in ep.sky_ellipses.items():
                        if inst_filter != "all" and inst_name.lower() != inst_filter:
                            continue
                        for ell in ell_list:
                            pts_px = _ellipse_outline_pixels_wcs(ell, wcs)
                            if pts_px:
                                scr = [
                                    implot.plot_to_pixels(px, py) for px, py in pts_px
                                ]
                                draw_list.add_polyline(
                                    scr, ell_col, 1.1, imgui.ImDrawFlags_.closed
                                )

            # 2. Per-epoch detections
            if state.show_epoch_detections:
                for ep_idx, ep in enumerate(state.epochs):
                    if not ep.enabled:
                        continue
                    col = _EPOCH_PALETTE[ep_idx % len(_EPOCH_PALETTE)]
                    col_u32 = imgui.get_color_u32(imgui.ImVec4(*col))
                    for det in filter_detections(ep.detections, state.filter_params):
                        px_pt = _wcs_sky_to_pixel(wcs, det.ra_deg, det.dec_deg)
                        if px_pt is None:
                            continue
                        scr = implot.plot_to_pixels(px_pt[0], px_pt[1])
                        arm = 4.0
                        if ep_idx % 2 == 0:
                            draw_list.add_line(
                                imgui.ImVec2(scr.x - arm, scr.y),
                                imgui.ImVec2(scr.x + arm, scr.y),
                                col_u32,
                                1.2,
                            )
                            draw_list.add_line(
                                imgui.ImVec2(scr.x, scr.y - arm),
                                imgui.ImVec2(scr.x, scr.y + arm),
                                col_u32,
                                1.2,
                            )
                        else:
                            draw_list.add_line(
                                imgui.ImVec2(scr.x - arm, scr.y - arm),
                                imgui.ImVec2(scr.x + arm, scr.y + arm),
                                col_u32,
                                1.2,
                            )
                            draw_list.add_line(
                                imgui.ImVec2(scr.x - arm, scr.y + arm),
                                imgui.ImVec2(scr.x + arm, scr.y - arm),
                                col_u32,
                                1.2,
                            )

            # 3. Merged sources
            inc_u32 = imgui.get_color_u32(imgui.ImVec4(0.25, 0.95, 0.45, 0.95))
            exc_u32 = imgui.get_color_u32(imgui.ImVec4(0.75, 0.30, 0.30, 0.60))
            sel_u32 = imgui.get_color_u32(imgui.ImVec4(1.00, 0.35, 0.85, 1.00))
            lbl_bg = imgui.get_color_u32(imgui.ImVec4(0.05, 0.06, 0.08, 0.78))

            merged_screen_pts: list[tuple[int, imgui.ImVec2, float]] = []
            if state.show_merged_sources:
                for idx, src in enumerate(state.merged_sources):
                    px_pt = _wcs_sky_to_pixel(wcs, src.ra_deg, src.dec_deg)
                    if px_pt is None:
                        continue
                    c_scr = implot.plot_to_pixels(px_pt[0], px_pt[1])
                    r_pix = max(3.0, state.export_radius_arcsec / pix_scale)
                    edge_scr = implot.plot_to_pixels(px_pt[0] + r_pix, px_pt[1])
                    r_scr = max(6.0, abs(edge_scr.x - c_scr.x))
                    merged_screen_pts.append((idx, c_scr, r_scr))

                    is_sel = state.selected_source_idx == idx
                    fg = inc_u32 if src.include else exc_u32
                    lw = 2.0 if src.include else 1.1
                    draw_list.add_circle(c_scr, r_scr, fg, 28, lw)
                    if not src.include:
                        d = r_scr * 0.65
                        draw_list.add_line(
                            imgui.ImVec2(c_scr.x - d, c_scr.y - d),
                            imgui.ImVec2(c_scr.x + d, c_scr.y + d),
                            exc_u32,
                            1.2,
                        )
                        draw_list.add_line(
                            imgui.ImVec2(c_scr.x - d, c_scr.y + d),
                            imgui.ImVec2(c_scr.x + d, c_scr.y - d),
                            exc_u32,
                            1.2,
                        )
                    if is_sel:
                        draw_list.add_circle(c_scr, r_scr + 4.0, sel_u32, 32, 2.2)

                    if is_sel or r_scr >= 8.0 or len(state.merged_sources) <= 80:
                        txt = f"S{src.merged_id}"
                        ts = imgui.calc_text_size(txt)
                        tx = c_scr.x - ts.x * 0.5
                        ty = c_scr.y - r_scr - ts.y - 3.0
                        draw_list.add_rect_filled(
                            imgui.ImVec2(tx - 2.0, ty - 1.0),
                            imgui.ImVec2(tx + ts.x + 2.0, ty + ts.y + 1.0),
                            lbl_bg,
                            2.0,
                        )
                        draw_list.add_text(imgui.ImVec2(tx, ty), fg, txt)

            draw_list.pop_clip_rect()

            # Canvas hover & click interactions
            if implot.is_plot_hovered():
                mouse_plot = implot.get_plot_mouse_pos()
                mouse_scr = imgui.get_mouse_pos()
                ra_h, dec_h = _wcs_pixel_to_sky(wcs, mouse_plot.x, mouse_plot.y)
                status_text = (
                    f"x={mouse_plot.x:.1f}, y={mouse_plot.y:.1f}  |  "
                    f"RA={ra_h:.5f}, Dec={dec_h:.5f}"
                )
                if has_bg and state.bg_image is not None:
                    ix = int(np.floor(mouse_plot.x + 0.5))
                    iy = int(np.floor(mouse_plot.y + 0.5))
                    if 0 <= ix < nx and 0 <= iy < ny:
                        smoothed = state.get_smoothed_data()
                        if smoothed is not None:
                            val = float(smoothed[iy, ix])
                            status_text += f"  |  val={val:.4g}"

                nearest_idx: int | None = None
                nearest_dist = float("inf")
                for idx, c_scr, r_scr in merged_screen_pts:
                    dist = math.hypot(mouse_scr.x - c_scr.x, mouse_scr.y - c_scr.y)
                    hit_r = max(14.0, r_scr + 4.0)
                    if dist <= hit_r and dist < nearest_dist:
                        nearest_dist = dist
                        nearest_idx = idx

                if nearest_idx is not None:
                    nsrc = state.merged_sources[nearest_idx]
                    tag = "manual" if nsrc.is_manual else f"N={len(nsrc.members)}"
                    inc_lbl = "ON" if nsrc.include else "OFF"
                    imgui.set_tooltip(
                        f"S{nsrc.merged_id} [{inc_lbl}] ML={nsrc.det_ml:.1f} "
                        f"Flux={nsrc.flux:.2e} ({tag})"
                    )

                left_delta = imgui.get_mouse_drag_delta(0)
                left_click = (
                    imgui.is_mouse_released(0)
                    and (left_delta.x * left_delta.x + left_delta.y * left_delta.y)
                    < 16.0
                )
                right_delta = imgui.get_mouse_drag_delta(1)
                right_click = (
                    imgui.is_mouse_released(1)
                    and (right_delta.x * right_delta.x + right_delta.y * right_delta.y)
                    < 16.0
                )

                if state.placing_manual_source and left_click:
                    state.add_manual_source(ra_h, dec_h)
                    state.placing_manual_source = False
                elif left_click and nearest_idx is not None:
                    state.selected_source_idx = nearest_idx

                if (right_click or imgui.is_mouse_double_clicked(0)) and (
                    nearest_idx is not None
                ):
                    state.selected_source_idx = nearest_idx
                    state.toggle_source_include(nearest_idx)

            if (
                not imgui.get_io().want_text_input
                and imgui.is_key_pressed(imgui.Key.space, False)
                and state.selected_source_idx is not None
            ):
                state.toggle_source_include(state.selected_source_idx)

            implot.end_plot()

        if has_bg:
            implot.pop_colormap()
            imgui.same_line()
            implot.colormap_scale(
                "##srclist_cbar",
                state.vmin,
                state.vmax,
                imgui.ImVec2(cbar_width - 10.0, plot_height),
                "%.3g",
                0,
                cmap,
            )

        imgui.text(status_text)

    def _gui() -> None:
        _draw_toolbar()
        imgui.separator()

        avail = imgui.get_content_region_avail()
        sidebar_width = min(400.0, max(280.0, avail.x * 0.30))

        if imgui.begin_child(
            "##srclist_sidebar",
            imgui.ImVec2(sidebar_width, 0.0),
            imgui.ChildFlags_.borders,
        ):
            _draw_sidebar()
        imgui.end_child()

        imgui.same_line()

        if imgui.begin_child("##srclist_canvas_pane", imgui.ImVec2(0.0, 0.0)):
            _draw_canvas()
        imgui.end_child()

    immapp.run(
        _gui,
        with_implot=True,
        window_title="XMM-Newton Cheese Source-List Merger",
        window_size=(1440, 900),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``xmm-srclist`` / ``python -m xmm_tools.srclist``."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge & curate multi-epoch XMM-Newton cheese emllist.fits catalogs."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Cheese directories, ana.toml files, or emllist.fits files to preload.",
    )
    parser.add_argument(
        "--bg",
        dest="bg_fits",
        default=None,
        help="Optional background FITS image with celestial WCS.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=5.0,
        help="Cross-epoch clustering radius in arcsec (0 = no merge, default 5.0).",
    )
    parser.add_argument(
        "--min-ml",
        type=float,
        default=15.0,
        help="Minimum DET_ML threshold (default 15.0).",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=0.2e-14,
        help="Minimum effective flux in erg cm^-2 s^-1 (default 0.2e-14).",
    )
    parser.add_argument(
        "--min-dist-nn",
        type=float,
        default=0.0,
        help="Minimum DIST_NN in arcsec (default 0.0).",
    )
    parser.add_argument(
        "--export-fits",
        default=None,
        help="Headlessly export merged_emllist.fits to this path and exit.",
    )
    parser.add_argument(
        "--export-reg",
        default=None,
        help="Headlessly export DS9 .reg overlay to this path and exit.",
    )
    args = parser.parse_args(argv)

    if args.export_fits is not None or args.export_reg is not None:
        state = SrclistViewerState(
            filter_params=FilterParams(
                min_det_ml=args.min_ml,
                min_flux=args.min_flux,
                min_dist_nn_arcsec=args.min_dist_nn,
            ),
            match_radius_arcsec=args.match_radius,
        )
        for p in args.paths:
            loaded = state.add_epoch(p)
            if loaded is None:
                print(state.error_message, file=sys.stderr)
                return 1
        if not state.epochs:
            print("No input epochs provided for headless export.", file=sys.stderr)
            return 1

        if args.export_fits is not None:
            fits_out = state.export_fits(args.export_fits)
            print(
                f"Exported {len(state.merged_sources)} sources to {fits_out}"
            )
        if args.export_reg is not None:
            reg_out = state.export_reg(args.export_reg)
            print(f"Exported DS9 regions to {reg_out}")
        return 0

    run_srclist_gui(
        args.paths,
        bg_fits=args.bg_fits,
        match_radius_arcsec=args.match_radius,
        min_det_ml=args.min_ml,
        min_flux=args.min_flux,
        min_dist_nn_arcsec=args.min_dist_nn,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
