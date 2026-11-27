"""FITS and DS9 I/O for XMM-Newton ``emllist.fits`` and cheese sky regions."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from xmm_tools.config import load_ana_config
from xmm_tools.esas.cheese import CheeseResult
from xmm_tools.srclist.models import (
    EpochCatalog,
    MergedSource,
    OverlayEllipse,
    SourceDetection,
)

logger = logging.getLogger(__name__)

# Standard 46-column emldetect SRCLIST schema validated against SAS 22.1.
STANDARD_SRCLIST_COLUMNS: list[tuple[str, str, str | None]] = [
    ("ML_ID_SRC", "J", None),
    ("BOX_ID_SRC", "J", None),
    ("ID_INST", "J", None),
    ("ID_BAND", "J", None),
    ("ID_CLUSTER", "J", None),
    ("SCTS", "E", "counts"),
    ("SCTS_ERR", "E", "counts"),
    ("X_IMA", "E", "image pixels"),
    ("X_IMA_ERR", "E", "image pixels"),
    ("Y_IMA", "E", "image pixels"),
    ("Y_IMA_ERR", "E", "image pixels"),
    ("EXT", "E", "image pixels"),
    ("EXT_ERR", "E", "image pixels"),
    ("DET_ML", "E", None),
    ("EXT_ML", "E", None),
    ("BG_MAP", "E", "counts pixel-1"),
    ("EXP_MAP", "E", "seconds"),
    ("FLUX", "E", "erg cm-2 s-1"),
    ("FLUX_ERR", "E", "erg cm-2 s-1"),
    ("RATE", "E", "counts s-1"),
    ("RATE_ERR", "E", "counts s-1"),
    ("RA", "D", "deg"),
    ("DEC", "D", "deg"),
    ("RADEC_ERR", "E", "arcsec"),
    ("LII", "D", "deg"),
    ("BII", "D", "deg"),
    ("RAWX", "J", "raw pixels"),
    ("RAWY", "J", "raw pixels"),
    ("OFFAX", "E", "arcmin"),
    ("CCDNR", "J", None),
    ("HR1", "E", None),
    ("HR1_ERR", "E", None),
    ("HR2", "E", None),
    ("HR2_ERR", "E", None),
    ("HR3", "E", None),
    ("HR3_ERR", "E", None),
    ("HR4", "E", None),
    ("HR4_ERR", "E", None),
    ("CUTRAD", "E", "pixels"),
    ("MASKFRAC", "E", None),
    ("EEF", "E", None),
    ("VIGNETTING", "E", None),
    ("ONTIME", "E", None),
    ("PILEUP", "E", None),
    ("DIST_NN", "E", "arcsec"),
    ("FLAG", "12A", None),
]

MANDATORY_SRCLIST_COLUMNS: tuple[str, ...] = (
    "ML_ID_SRC",
    "BOX_ID_SRC",
    "ID_INST",
    "ID_BAND",
    "ID_CLUSTER",
    "SCTS",
    "SCTS_ERR",
    "X_IMA",
    "X_IMA_ERR",
    "Y_IMA",
    "Y_IMA_ERR",
    "EXT",
    "EXT_ERR",
    "DET_ML",
    "EXT_ML",
    "BG_MAP",
    "EXP_MAP",
    "FLUX",
    "FLUX_ERR",
    "RATE",
    "RATE_ERR",
    "RA",
    "DEC",
    "RADEC_ERR",
    "DIST_NN",
    "FLAG",
)

DEFAULT_TCDELT1 = -0.000694444461259991  # -2.5 arcsec/pixel
DEFAULT_TCDELT2 = 0.000694444461259991  # +2.5 arcsec/pixel
_DEFAULT_BG_MAP = 0.1
_DEFAULT_SCTS = 1.0


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return default
    return fval if math.isfinite(fval) else default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if isinstance(val, (float, np.floating)) and not math.isfinite(float(val)):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _to_python_scalar(val: Any) -> Any:
    if isinstance(val, bytes):
        return val.decode("ascii", errors="replace").strip()
    if isinstance(val, np.bytes_):
        return bytes(val).decode("ascii", errors="replace").strip()
    if isinstance(val, (np.str_, str)):
        return str(val).strip()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        fval = float(val)
        return fval if math.isfinite(fval) else 0.0
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _first_scalar(val: Any, default: float = 0.0) -> float:
    arr = np.atleast_1d(val)
    if arr.size == 0:
        return default
    return _safe_float(arr.flat[0], default=default)


def _find_srclist_hdu(hdul: fits.HDUList) -> fits.BinTableHDU:
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.name.upper() == "SRCLIST":
            return hdu
    raise ValueError("No SRCLIST BinTableHDU found in FITS file")


def _extract_raw_row(row: Any, col_names: Sequence[str]) -> dict[str, Any]:
    return {name: _to_python_scalar(row[name]) for name in col_names}


def _pick_summary_row(rows: list[Any], col_set: set[str]) -> Any:
    if "ID_INST" in col_set and "ID_BAND" in col_set:
        for row in rows:
            if _safe_int(row["ID_INST"], -1) == 0 and _safe_int(row["ID_BAND"], -1) == 0:
                return row
    if "ID_INST" in col_set:
        for row in rows:
            if _safe_int(row["ID_INST"], -1) == 0:
                return row
    if "DET_ML" in col_set:
        return max(rows, key=lambda r: _safe_float(r["DET_ML"], -1.0))
    return rows[0]


def _compute_effective_flux(
    rows: list[Any], summary_row: Any, col_set: set[str]
) -> float:
    if "FLUX" not in col_set:
        return 0.0
    if "ID_INST" in col_set and "ID_BAND" in col_set:
        band1_fluxes = [
            _safe_float(r["FLUX"], 0.0)
            for r in rows
            if _safe_int(r["ID_INST"], -1) in (1, 2, 3)
            and _safe_int(r["ID_BAND"], -1) == 1
            and _safe_float(r["FLUX"], 0.0) > 0.0
        ]
        if band1_fluxes:
            return max(band1_fluxes)
    return max(0.0, _safe_float(summary_row["FLUX"], 0.0))


def _compute_effective_bg_map(
    rows: list[Any], summary_row: Any, col_set: set[str]
) -> float:
    if "BG_MAP" not in col_set:
        return 0.0
    summary_bg = _safe_float(summary_row["BG_MAP"], 0.0)
    if summary_bg > 0.0:
        return summary_bg
    pos_bgs = [
        _safe_float(r["BG_MAP"], 0.0)
        for r in rows
        if _safe_float(r["BG_MAP"], 0.0) > 0.0
    ]
    return max(pos_bgs) if pos_bgs else 0.0


def _compute_effective_scts(
    rows: list[Any], summary_row: Any, col_set: set[str]
) -> float:
    if "SCTS" not in col_set:
        return 0.0
    summary_scts = _safe_float(summary_row["SCTS"], 0.0)
    if summary_scts > 0.0:
        return summary_scts
    pos_scts = [
        _safe_float(r["SCTS"], 0.0)
        for r in rows
        if _safe_float(r["SCTS"], 0.0) > 0.0
    ]
    return max(pos_scts) if pos_scts else 0.0


def load_emllist(
    path: str | Path,
    *,
    label: str | None = None,
    epoch_id: str | None = None,
) -> EpochCatalog:
    """Load an ``emllist.fits`` source catalog into an :class:`EpochCatalog`."""
    file_path = Path(path)
    with fits.open(file_path) as hdul:
        primary_header = hdul[0].header.copy()
        srclist_hdu = _find_srclist_hdu(hdul)
        srclist_header = srclist_hdu.header.copy()
        column_defs = [
            (str(col.name), str(col.format), str(col.unit) if col.unit else None)
            for col in srclist_hdu.columns
        ]
        col_names = [name for name, _, _ in column_defs]
        col_set = set(col_names)
        data = srclist_hdu.data
        rows = list(data) if data is not None else []

    obs_id = str(
        primary_header.get("OBS_ID") or srclist_header.get("OBS_ID") or ""
    ).strip()
    resolved_label = label or (obs_id if obs_id else file_path.stem)
    resolved_epoch_id = epoch_id or str(file_path.resolve())

    grouped: dict[int, list[Any]] = {}
    has_ml_id = "ML_ID_SRC" in col_set
    for idx, row in enumerate(rows):
        if has_ml_id:
            raw_id = _safe_int(row["ML_ID_SRC"], 0)
            src_id = raw_id if raw_id > 0 else -(idx + 1)
        else:
            src_id = -(idx + 1)
        grouped.setdefault(src_id, []).append(row)

    detections: list[SourceDetection] = []
    for src_id, src_rows in grouped.items():
        summary = _pick_summary_row(src_rows, col_set)
        if "RA" not in col_set or "DEC" not in col_set:
            logger.warning(
                "Skipping source %s in %s: missing RA/DEC columns",
                src_id,
                file_path,
            )
            continue
        ra_val = _safe_float(summary["RA"], float("nan"))
        dec_val = _safe_float(summary["DEC"], float("nan"))
        if not (math.isfinite(ra_val) and math.isfinite(dec_val)):
            logger.warning(
                "Skipping source %s in %s: non-finite RA/DEC (%r, %r)",
                src_id,
                file_path,
                summary["RA"],
                summary["DEC"],
            )
            continue

        raw_row = _extract_raw_row(summary, col_names)
        eff_flux = _compute_effective_flux(src_rows, summary, col_set)
        eff_bg = _compute_effective_bg_map(src_rows, summary, col_set)
        eff_scts = _compute_effective_scts(src_rows, summary, col_set)
        raw_row["FLUX"] = eff_flux
        raw_row["BG_MAP"] = eff_bg
        raw_row["SCTS"] = eff_scts

        detections.append(
            SourceDetection(
                epoch_id=resolved_epoch_id,
                epoch_label=resolved_label,
                ml_id_src=src_id,
                ra_deg=ra_val,
                dec_deg=dec_val,
                radec_err_arcsec=(
                    _safe_float(summary["RADEC_ERR"], 1.0)
                    if "RADEC_ERR" in col_set
                    else 1.0
                ),
                det_ml=(
                    _safe_float(summary["DET_ML"], 0.0) if "DET_ML" in col_set else 0.0
                ),
                flux=eff_flux,
                scts=eff_scts,
                bg_map=eff_bg,
                ext=_safe_float(summary["EXT"], 0.0) if "EXT" in col_set else 0.0,
                ext_ml=(
                    _safe_float(summary["EXT_ML"], 0.0) if "EXT_ML" in col_set else 0.0
                ),
                dist_nn_arcsec=(
                    _safe_float(summary["DIST_NN"], 999.0)
                    if "DIST_NN" in col_set
                    else 999.0
                ),
                rate=_safe_float(summary["RATE"], 0.0) if "RATE" in col_set else 0.0,
                exp_map=(
                    _safe_float(summary["EXP_MAP"], 0.0)
                    if "EXP_MAP" in col_set
                    else 0.0
                ),
                raw_row=raw_row,
            )
        )

    return EpochCatalog(
        epoch_id=resolved_epoch_id,
        label=resolved_label,
        source_path=file_path,
        cheese_dir=None,
        obs_id=obs_id,
        enabled=True,
        detections=detections,
        sky_ellipses={},
        candidate_images=[],
        primary_header=primary_header,
        srclist_header=srclist_header,
        column_defs=column_defs,
    )


def load_sky_region_ellipses(
    fits_path: str | Path, *, instrument: str
) -> list[OverlayEllipse]:
    """Parse cheese sky-coordinate exclusion ellipses from ``*-bkgregtsky.fits``."""
    file_path = Path(fits_path)
    with fits.open(file_path) as hdul:
        reg_hdu: fits.BinTableHDU | None = None
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU) and hdu.name.upper() == "REGION":
                reg_hdu = hdu
                break
        if reg_hdu is None:
            for hdu in hdul:
                if isinstance(hdu, fits.BinTableHDU):
                    reg_hdu = hdu
                    break
        if reg_hdu is None:
            raise ValueError(f"No REGION BinTableHDU found in {file_path}")

        hdr = reg_hdu.header
        data = reg_hdu.data
        if data is None or len(data) == 0:
            return []

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [
            str(hdr.get("TCTYP2", "RA---TAN")),
            str(hdr.get("TCTYP3", "DEC--TAN")),
        ]
        wcs.wcs.crpix = [
            _safe_float(hdr.get("TCRPX2"), 1.0),
            _safe_float(hdr.get("TCRPX3"), 1.0),
        ]
        wcs.wcs.crval = [
            _safe_float(hdr.get("TCRVL2"), 0.0),
            _safe_float(hdr.get("TCRVL3"), 0.0),
        ]
        tcdlt2 = _safe_float(hdr.get("TCDLT2"), -1.38888888888889e-5)
        tcdlt3 = _safe_float(hdr.get("TCDLT3"), 1.38888888888889e-5)
        wcs.wcs.cdelt = [tcdlt2, tcdlt3]
        scale_arcsec = abs(tcdlt3 if tcdlt3 != 0.0 else tcdlt2) * 3600.0
        if scale_arcsec <= 0.0:
            scale_arcsec = 0.05

        col_names = set(reg_hdu.columns.names)
        if not {"X", "Y", "R"}.issubset(col_names):
            logger.warning("REGION table in %s missing X/Y/R columns", file_path)
            return []

        ellipses: list[OverlayEllipse] = []
        for row_idx, row in enumerate(data):
            shape_raw = _to_python_scalar(row["SHAPE"]) if "SHAPE" in col_names else ""
            shape = str(shape_raw).strip().upper()
            if "CIRCLE" not in shape and "ELLIPSE" not in shape:
                logger.warning(
                    "Skipping unsupported region shape %r at row %d in %s",
                    shape_raw,
                    row_idx,
                    file_path,
                )
                continue

            x_pix = _first_scalar(row["X"], float("nan"))
            y_pix = _first_scalar(row["Y"], float("nan"))
            if not (math.isfinite(x_pix) and math.isfinite(y_pix)):
                logger.warning(
                    "Skipping malformed region row %d in %s: non-finite X/Y",
                    row_idx,
                    file_path,
                )
                continue

            ra_arr, dec_arr = wcs.wcs_pix2world([x_pix], [y_pix], 1)
            ra_deg = float(ra_arr[0])
            dec_deg = float(dec_arr[0])
            if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
                logger.warning(
                    "Skipping malformed region row %d in %s: non-finite sky coords",
                    row_idx,
                    file_path,
                )
                continue

            try:
                r_arr = np.atleast_1d(np.asarray(row["R"], dtype=float))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping malformed region row %d in %s: invalid R column",
                    row_idx,
                    file_path,
                )
                continue

            rot_val = _first_scalar(row["ROTANG"], 0.0) if "ROTANG" in col_names else 0.0

            if "CIRCLE" in shape:
                if r_arr.size < 1:
                    logger.warning(
                        "Skipping malformed CIRCLE row %d in %s: empty R",
                        row_idx,
                        file_path,
                    )
                    continue
                r0 = float(r_arr[0])
                if not math.isfinite(r0) or r0 <= 0.0:
                    logger.warning(
                        "Skipping malformed CIRCLE row %d in %s: invalid radius %r",
                        row_idx,
                        file_path,
                        r0,
                    )
                    continue
                rad_arcsec = r0 * scale_arcsec
                ellipses.append(
                    OverlayEllipse(
                        ra_deg=ra_deg,
                        dec_deg=dec_deg,
                        semi_major_arcsec=rad_arcsec,
                        semi_minor_arcsec=rad_arcsec,
                        angle_deg=0.0,
                        instrument=instrument,
                    )
                )
            elif "ELLIPSE" in shape:
                if r_arr.size < 2:
                    logger.warning(
                        "Skipping malformed ELLIPSE row %d in %s: R has size %d",
                        row_idx,
                        file_path,
                        r_arr.size,
                    )
                    continue
                r0 = float(r_arr[0])
                r1 = float(r_arr[1])
                if (
                    not math.isfinite(r0)
                    or not math.isfinite(r1)
                    or r0 <= 0.0
                    or r1 <= 0.0
                ):
                    logger.warning(
                        "Skipping malformed ELLIPSE row %d in %s: invalid radii (%r, %r)",
                        row_idx,
                        file_path,
                        r0,
                        r1,
                    )
                    continue
                semi_a = r0 * scale_arcsec
                semi_b = r1 * scale_arcsec
                angle = float(rot_val) % 360.0
                if semi_a < semi_b:
                    semi_a, semi_b = semi_b, semi_a
                    angle = (angle + 90.0) % 360.0
                ellipses.append(
                    OverlayEllipse(
                        ra_deg=ra_deg,
                        dec_deg=dec_deg,
                        semi_major_arcsec=semi_a,
                        semi_minor_arcsec=semi_b,
                        angle_deg=angle,
                        instrument=instrument,
                    )
                )
    return ellipses


def _infer_instrument(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("mos1") or "m1" in stem:
        return "mos1"
    if stem.startswith("mos2") or "m2" in stem:
        return "mos2"
    if stem.startswith("pn"):
        return "pn"
    return stem.split("-")[0]


def _discover_candidate_images(cheese_dir: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        resolved = p.resolve()
        if resolved not in seen and p.is_file():
            seen.add(resolved)
            found.append(p)

    for img_path in sorted(cheese_dir.glob("*-cheeset.fits")):
        _add(img_path)

    qpb_dir = cheese_dir.parent / "qpb_image"
    if qpb_dir.is_dir():
        for img_path in sorted(qpb_dir.glob("*_flux.fits")):
            _add(img_path)
    return found


def load_cheese_epoch(
    path: str | Path, *, label: str | None = None
) -> EpochCatalog:
    """Load an epoch from a ``cheese`` directory, ``ana.toml``, or ``emllist.fits``."""
    target = Path(path)
    cheese_dir: Path | None = None
    emllist_path: Path | None = None
    region_sky_map: dict[str, Path] = {}

    if target.is_file() and target.suffix.lower() == ".toml":
        cfg = load_ana_config(target)
        cheese_dir = Path(cfg.paths.resolve(cfg.paths.cheese_dir))
    elif target.is_dir():
        if (target / "manifest.json").is_file() or (target / "emllist.fits").is_file():
            cheese_dir = target
        elif (target / "cheese").is_dir():
            cheese_dir = target / "cheese"
        else:
            cheese_dir = target
    elif target.is_file():
        emllist_path = target
        parent = target.parent
        if (parent / "manifest.json").is_file() or any(
            parent.glob("*-bkgregtsky.fits")
        ) or any(parent.glob("*-cheeset.fits")):
            cheese_dir = parent
    else:
        raise FileNotFoundError(f"Cheese directory or emllist file not found: {target}")

    if cheese_dir is not None:
        manifest_path = cheese_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                res = CheeseResult.from_manifest(
                    manifest_path, require_completed=False
                )
                if res.emllist is not None and res.emllist.is_file():
                    emllist_path = res.emllist
                region_sky_map = {
                    inst: p for inst, p in res.region_sky.items() if p.is_file()
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ignoring unreadable manifest %s: %s", manifest_path, exc)

        if emllist_path is None:
            candidate = cheese_dir / "emllist.fits"
            if candidate.is_file():
                emllist_path = candidate
            else:
                fits_matches = sorted(cheese_dir.glob("*emllist*.fits"))
                if fits_matches:
                    emllist_path = fits_matches[0]

        if not region_sky_map:
            for reg_file in sorted(cheese_dir.glob("*-bkgregtsky.fits")):
                inst = _infer_instrument(reg_file)
                region_sky_map[inst] = reg_file

    if emllist_path is None or not emllist_path.is_file():
        raise FileNotFoundError(f"Could not locate emllist.fits from {target}")

    catalog = load_emllist(emllist_path, label=label)
    catalog.cheese_dir = cheese_dir
    if not label and cheese_dir is not None and not catalog.obs_id:
        catalog.label = cheese_dir.parent.name or cheese_dir.name

    sky_ellipses: dict[str, list[OverlayEllipse]] = {}
    for inst, reg_path in sorted(region_sky_map.items()):
        try:
            sky_ellipses[inst] = load_sky_region_ellipses(reg_path, instrument=inst)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load sky region %s: %s", reg_path, exc)
    catalog.sky_ellipses = sky_ellipses

    if cheese_dir is not None:
        catalog.candidate_images = _discover_candidate_images(cheese_dir)

    return catalog


def _resolve_output_schema(
    template_catalog: EpochCatalog | None,
) -> list[tuple[str, str, str | None]]:
    std_by_name = {
        name: (name, fmt, unit) for name, fmt, unit in STANDARD_SRCLIST_COLUMNS
    }
    mandatory_set = set(MANDATORY_SRCLIST_COLUMNS)

    if template_catalog is not None and template_catalog.column_defs:
        cols: list[tuple[str, str, str | None]] = []
        for name, fmt, unit in template_catalog.column_defs:
            if name in mandatory_set and name in std_by_name:
                _, std_fmt, std_unit = std_by_name[name]
                cols.append((name, std_fmt, unit or std_unit))
            else:
                cols.append((name, fmt, unit))
    else:
        cols = list(STANDARD_SRCLIST_COLUMNS)

    existing = {name for name, _, _ in cols}
    for req in MANDATORY_SRCLIST_COLUMNS:
        if req not in existing and req in std_by_name:
            cols.append(std_by_name[req])
            existing.add(req)
    return cols


def _spherical_mean_sky(coords: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Compute spherical unit-vector mean ``(ra_deg, dec_deg)``, handling 0/360 wrap."""
    if not coords:
        return 0.0, 0.0
    ra_rad = np.radians([c[0] for c in coords])
    dec_rad = np.radians([c[1] for c in coords])
    mx = float(np.mean(np.cos(dec_rad) * np.cos(ra_rad)))
    my = float(np.mean(np.cos(dec_rad) * np.sin(ra_rad)))
    mz = float(np.mean(np.sin(dec_rad)))
    norm = math.sqrt(mx * mx + my * my + mz * mz)
    if norm <= 0.0:
        ra0 = float(coords[0][0]) % 360.0
        return (0.0 if ra0 >= 360.0 else ra0), float(coords[0][1])
    mean_ra = math.degrees(math.atan2(my, mx)) % 360.0
    if mean_ra >= 360.0:
        mean_ra = 0.0
    mean_dec = math.degrees(math.asin(max(-1.0, min(1.0, mz / norm))))
    return mean_ra, mean_dec


def _resolve_wcs_keywords(
    kept: Sequence[MergedSource], template_catalog: EpochCatalog | None
) -> dict[str, Any]:
    hdr_src = template_catalog.srclist_header if template_catalog is not None else None
    hdr_pri = template_catalog.primary_header if template_catalog is not None else None

    def _pick(key: str, default: Any) -> Any:
        if hdr_src is not None and key in hdr_src:
            return hdr_src[key]
        if hdr_pri is not None and key in hdr_pri:
            return hdr_pri[key]
        return default

    mean_ra, mean_dec = _spherical_mean_sky([(s.ra_deg, s.dec_deg) for s in kept])

    return {
        "TCTYPE1": str(_pick("TCTYPE1", "RA---TAN")),
        "TCTYPE2": str(_pick("TCTYPE2", "DEC--TAN")),
        "TCRPIX1": _safe_float(_pick("TCRPIX1", 450.5), 450.5),
        "TCRPIX2": _safe_float(_pick("TCRPIX2", 450.5), 450.5),
        "TCRVAL1": _safe_float(_pick("TCRVAL1", mean_ra), mean_ra),
        "TCRVAL2": _safe_float(_pick("TCRVAL2", mean_dec), mean_dec),
        "TCDELT1": _safe_float(_pick("TCDELT1", DEFAULT_TCDELT1), DEFAULT_TCDELT1),
        "TCDELT2": _safe_float(_pick("TCDELT2", DEFAULT_TCDELT2), DEFAULT_TCDELT2),
        "TCUNIT1": str(_pick("TCUNIT1", "deg")),
        "TCUNIT2": str(_pick("TCUNIT2", "deg")),
        "RADECSYS": str(_pick("RADECSYS", "FK5")),
        "EQUINOX": _safe_float(_pick("EQUINOX", 2000.0), 2000.0),
    }


def _build_table_wcs(wcs_kw: dict[str, Any]) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [wcs_kw["TCTYPE1"], wcs_kw["TCTYPE2"]]
    wcs.wcs.crpix = [wcs_kw["TCRPIX1"], wcs_kw["TCRPIX2"]]
    wcs.wcs.crval = [wcs_kw["TCRVAL1"], wcs_kw["TCRVAL2"]]
    wcs.wcs.cdelt = [wcs_kw["TCDELT1"], wcs_kw["TCDELT2"]]
    return wcs


def _default_for_format(fmt: str) -> Any:
    upper = fmt.upper()
    if "A" in upper:
        return ""
    if any(ch in upper for ch in ("J", "I", "K", "B")):
        return 0
    if "L" in upper:
        return False
    return 0.0


def export_merged_emllist(
    sources: Sequence[MergedSource],
    out_path: str | Path,
    *,
    template_catalog: EpochCatalog | None = None,
    include_only: bool = True,
) -> Path:
    """Write a SAS 22.1-compatible ``merged_emllist.fits`` with 1 row per source."""
    kept = [s for s in sources if (s.include or not include_only)]
    col_schema = _resolve_output_schema(template_catalog)
    wcs_kw = _resolve_wcs_keywords(kept, template_catalog)
    table_wcs = _build_table_wcs(wcs_kw)

    if kept:
        ra_list = [s.ra_deg for s in kept]
        dec_list = [s.dec_deg for s in kept]
        x_arr, y_arr = table_wcs.wcs_world2pix(ra_list, dec_list, 1)
        if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
            mean_ra, mean_dec = _spherical_mean_sky(list(zip(ra_list, dec_list)))
            wcs_kw["TCRVAL1"] = mean_ra
            wcs_kw["TCRVAL2"] = mean_dec
            table_wcs = _build_table_wcs(wcs_kw)
            x_arr, y_arr = table_wcs.wcs_world2pix(ra_list, dec_list, 1)
    else:
        x_arr = np.empty(0, dtype=float)
        y_arr = np.empty(0, dtype=float)

    col_arrays: dict[str, list[Any]] = {name: [] for name, _, _ in col_schema}
    for idx, src in enumerate(kept):
        row_idx = idx + 1
        base = dict(src.representative_row)
        x_raw = float(x_arr[idx])
        y_raw = float(y_arr[idx])
        x_val = x_raw if math.isfinite(x_raw) else float(wcs_kw["TCRPIX1"])
        y_val = y_raw if math.isfinite(y_raw) else float(wcs_kw["TCRPIX2"])
        scts_val = (
            float(src.scts)
            if (math.isfinite(src.scts) and src.scts > 0.0)
            else _DEFAULT_SCTS
        )
        bg_val = (
            float(src.bg_map)
            if (math.isfinite(src.bg_map) and src.bg_map > 0.0)
            else _DEFAULT_BG_MAP
        )

        overrides: dict[str, Any] = {
            "ML_ID_SRC": row_idx,
            "BOX_ID_SRC": row_idx,
            "ID_INST": 0,
            "ID_BAND": 0,
            "ID_CLUSTER": row_idx,
            "SCTS": scts_val,
            "X_IMA": x_val,
            "Y_IMA": y_val,
            "EXT": float(src.ext),
            "DET_ML": float(src.det_ml),
            "EXT_ML": float(src.ext_ml),
            "BG_MAP": bg_val,
            "FLUX": float(src.flux),
            "RA": float(src.ra_deg),
            "DEC": float(src.dec_deg),
            "RADEC_ERR": float(src.radec_err_arcsec),
            "DIST_NN": float(src.dist_nn_arcsec),
        }
        for name, fmt, _ in col_schema:
            if name in overrides:
                col_arrays[name].append(overrides[name])
            elif name in base:
                val = base[name]
                if "A" in fmt.upper():
                    col_arrays[name].append(str(val))
                elif any(ch in fmt.upper() for ch in ("J", "I", "K", "B")):
                    col_arrays[name].append(_safe_int(val, 0))
                else:
                    col_arrays[name].append(_safe_float(val, 0.0))
            else:
                col_arrays[name].append(_default_for_format(fmt))

    fits_cols: list[fits.Column] = []
    for name, fmt, unit in col_schema:
        upper = fmt.upper()
        if "A" in upper:
            dtype = str
        elif any(ch in upper for ch in ("J", "I", "K", "B")):
            dtype = np.int32
        elif "D" in upper:
            dtype = np.float64
        else:
            dtype = np.float32
        arr = np.asarray(col_arrays[name], dtype=dtype)
        fits_cols.append(fits.Column(name=name, format=fmt, unit=unit, array=arr))

    primary_hdr = (
        template_catalog.primary_header.copy()
        if template_catalog is not None
        else fits.Header()
    )
    srclist_hdr = (
        template_catalog.srclist_header.copy()
        if template_catalog is not None
        else fits.Header()
    )

    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    table_hdu = fits.BinTableHDU.from_columns(fits_cols, header=srclist_hdr, name="SRCLIST")

    for hdr in (primary_hdu.header, table_hdu.header):
        for k, v in wcs_kw.items():
            hdr[k] = v
    table_hdu.header["EXTNAME"] = "SRCLIST"
    table_hdu.header["NDETECT"] = len(kept)
    primary_hdu.header["NDETECT"] = len(kept)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary_hdu, table_hdu]).writeto(target, overwrite=True)
    return target


def export_merged_ds9_reg(
    sources: Sequence[MergedSource],
    out_path: str | Path,
    *,
    default_radius_arcsec: float = 15.0,
    include_excluded: bool = False,
) -> Path:
    """Export merged sources as a DS9 ``fk5`` ``.reg`` overlay file."""
    lines = [
        "# Region file format: DS9 version 4.1",
        (
            "global color=green dashlist=8 3 width=1 "
            'font="helvetica 10 normal roman" select=1 highlite=1 '
            "dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
        ),
        "fk5",
    ]
    for src in sources:
        if not src.include and not include_excluded:
            continue
        prefix = "" if src.include else "-"
        tag = "manual" if src.is_manual else f"N={len(src.members)}"
        lines.append(
            f"{prefix}circle({src.ra_deg:.7f},{src.dec_deg:.7f},"
            f'{default_radius_arcsec:.2f}") '
            f"# text={{S{src.merged_id} ML={src.det_ml:.1f} {tag}}}"
        )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
