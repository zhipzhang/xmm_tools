"""Data models for XMM-Newton ``emllist.fits`` source-list merging & curation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.io import fits


@dataclass
class OverlayEllipse:
    """Sky-coordinate cheese mask ellipse parsed from ``*-bkgregtsky.fits``."""

    ra_deg: float
    dec_deg: float
    semi_major_arcsec: float
    semi_minor_arcsec: float
    angle_deg: float  # DS9 FK5 convention (deg CCW from West toward North)
    instrument: str


@dataclass
class SourceDetection:
    """Single point-source entry extracted from an epoch's ``emllist.fits``."""

    epoch_id: str
    epoch_label: str
    ml_id_src: int
    ra_deg: float
    dec_deg: float
    radec_err_arcsec: float
    det_ml: float
    flux: float
    scts: float
    bg_map: float
    ext: float
    ext_ml: float
    dist_nn_arcsec: float
    rate: float
    exp_map: float
    raw_row: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochCatalog:
    """Loaded epoch ``emllist.fits`` catalog and associated cheese products."""

    epoch_id: str
    label: str
    source_path: Path
    cheese_dir: Path | None = None
    obs_id: str = ""
    enabled: bool = True
    detections: list[SourceDetection] = field(default_factory=list)
    sky_ellipses: dict[str, list[OverlayEllipse]] = field(default_factory=dict)
    candidate_images: list[Path] = field(default_factory=list)
    primary_header: fits.Header = field(default_factory=fits.Header)
    srclist_header: fits.Header = field(default_factory=fits.Header)
    column_defs: list[tuple[str, str, str | None]] = field(default_factory=list)


@dataclass
class FilterParams:
    """Detection filtering thresholds applied prior to cross-epoch merging."""

    min_det_ml: float = 15.0
    min_flux: float = 0.2e-14
    min_dist_nn_arcsec: float = 0.0
    exclude_extended: bool = False


@dataclass
class MergedSource:
    """Unified source entry ready for curation and SAS ``emllist.fits`` export."""

    merged_id: int
    ra_deg: float
    dec_deg: float
    radec_err_arcsec: float
    det_ml: float
    flux: float
    scts: float
    bg_map: float
    ext: float
    ext_ml: float
    dist_nn_arcsec: float
    include: bool = True
    is_manual: bool = False
    manual_id: int | None = None
    members: list[SourceDetection] = field(default_factory=list)
    representative_row: dict[str, Any] = field(default_factory=dict)
