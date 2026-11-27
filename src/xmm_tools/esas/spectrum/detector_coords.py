"""Sky-to-DETX/DETY coordinate conversion using ``evselect`` and ``ecoordconv``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from xmm_tools.esas.spectrum.regions import _import_regions
from xmm_tools.exec import (
    SASCommand,
    SASRunner,
    StageWorkspace,
    StepLedger,
    is_nonempty_file,
)

DET_SCALE = 20.0
_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_ECOORD_RE = re.compile(rf"DETX:\s*DETY:\s*({_NUMBER_RE})\s+({_NUMBER_RE})")


def ensure_ref_sky_images(
    active_insts: Sequence[str],
    clean_events: Mapping[str, Path],
    ws: StageWorkspace,
    ledger: StepLedger,
    runner: SASRunner,
    *,
    cal_inputs: Mapping[str, Path],
    dry_run: bool,
) -> dict[str, Path]:
    """Create shared 600x600 X/Y reference images per instrument via ``StepLedger``."""
    ref_images: dict[str, Path] = {}
    for inst in active_insts:
        evt_path = clean_events[inst]
        ref_dir = ws.step_dir("ref_sky", inst)
        img_name = f"ref_{inst}.fits"
        cmd = SASCommand(
            program="evselect",
            cwd=ref_dir,
            params={
                "table": "events.fits",
                "withimageset": "yes",
                "imageset": img_name,
                "xcolumn": "X",
                "ycolumn": "Y",
                "imagebinning": "imageSize",
                "ximagesize": 600,
                "yimagesize": 600,
            },
            symlinks={"events.fits": evt_path},
            expected_outputs=[img_name],
            log_file=f"ref_img_{inst}.log",
        )
        step_inputs = dict(cal_inputs)
        if not dry_run or is_nonempty_file(evt_path):
            step_inputs["events"] = evt_path
        ledger.run_step(f"ref_sky_image:{inst}", runner, cmd, inputs=step_inputs)
        ref_images[inst] = ref_dir / img_name
    return ref_images


def run_ecoordconv(
    fits_image: Path,
    ra_deg: float,
    dec_deg: float,
    runner: SASRunner,
    log_dir: Path,
    tag: str,
) -> tuple[float, float]:
    """Convert one RA/Dec coordinate to ``(DETX, DETY)`` via SAS ``ecoordconv``."""
    cmd = SASCommand(
        program="ecoordconv",
        cwd=log_dir,
        params={
            "imageset": fits_image.name,
            "x": f"{ra_deg:.8f}",
            "y": f"{dec_deg:.8f}",
            "coordtype": "eqpos",
        },
        symlinks={fits_image.name: fits_image},
        expected_outputs=[],
        log_file=f"ecoordconv_{tag}.log",
    )
    res = runner.run(cmd, force=True)
    if res.status == "dry_run" or res.log_path is None or not res.log_path.is_file():
        return 0.0, 0.0
    text = res.log_path.read_text(encoding="utf-8", errors="replace")
    match = _ECOORD_RE.search(text)
    if not match:
        raise RuntimeError(
            f"Could not parse DETX/DETY from ecoordconv log {res.log_path}:\n{text[-1000:]}"
        )
    return float(match.group(1)), float(match.group(2))


def sky_region_to_esas_expr(
    sky_reg: Any,
    fits_image: Path,
    wcs_ref: WCS | None,
    runner: SASRunner,
    log_dir: Path,
    tag: str,
) -> str:
    """Convert one ``astropy-regions`` SkyRegion into an ESAS ``DETX/DETY`` expression."""
    _, CircleSkyRegion, EllipseSkyRegion, RectangleSkyRegion, PolygonSkyRegion = (
        _import_regions()
    )
    if isinstance(sky_reg, CircleSkyRegion):
        detx, dety = run_ecoordconv(
            fits_image,
            sky_reg.center.ra.deg,
            sky_reg.center.dec.deg,
            runner,
            log_dir,
            tag,
        )
        radius = sky_reg.radius.to(u.arcsec).value * DET_SCALE
        return f"&&((DETX,DETY) IN circle({detx:.1f},{dety:.1f},{radius:.1f}))"

    if isinstance(sky_reg, EllipseSkyRegion):
        if wcs_ref is None:
            wcs_ref = WCS(fits.getheader(fits_image))
        detx0, dety0 = run_ecoordconv(
            fits_image,
            sky_reg.center.ra.deg,
            sky_reg.center.dec.deg,
            runner,
            log_dir,
            f"{tag}_c",
        )
        rx = 0.5 * sky_reg.width.to(u.arcsec).value * DET_SCALE
        ry = 0.5 * sky_reg.height.to(u.arcsec).value * DET_SCALE
        pix_reg = sky_reg.to_pixel(wcs_ref)
        theta = pix_reg.angle.to(u.rad).value
        end_x = pix_reg.center.x + 0.5 * pix_reg.width * np.cos(theta)
        end_y = pix_reg.center.y + 0.5 * pix_reg.width * np.sin(theta)
        end_sky = wcs_ref.pixel_to_world(end_x, end_y)
        detx1, dety1 = run_ecoordconv(
            fits_image,
            float(end_sky.ra.deg),
            float(end_sky.dec.deg),
            runner,
            log_dir,
            f"{tag}_a",
        )
        angle = float(np.degrees(np.arctan2(dety1 - dety0, detx1 - detx0)) % 360.0)
        return (
            f"&&((DETX,DETY) IN ellipse({detx0:.1f},{dety0:.1f},"
            f"{rx:.1f},{ry:.1f},{angle:.2f}))"
        )

    if isinstance(sky_reg, (RectangleSkyRegion, PolygonSkyRegion)):
        if isinstance(sky_reg, PolygonSkyRegion):
            vertices = sky_reg.vertices
        else:
            if wcs_ref is None:
                wcs_ref = WCS(fits.getheader(fits_image))
            pix_reg = sky_reg.to_pixel(wcs_ref)
            vertices = wcs_ref.pixel_to_world(
                pix_reg.corners[:, 0], pix_reg.corners[:, 1]
            )
        det_coords: list[str] = []
        for idx, vertex in enumerate(vertices):
            detx, dety = run_ecoordconv(
                fits_image,
                float(vertex.ra.deg),
                float(vertex.dec.deg),
                runner,
                log_dir,
                f"{tag}_v{idx}",
            )
            det_coords.extend([f"{detx:.1f}", f"{dety:.1f}"])
        return f"&&((DETX,DETY) IN polygon({','.join(det_coords)}))"

    raise ValueError(f"Unsupported sky region type: {type(sky_reg).__name__}")


def build_sky_region_expr(
    sky_regions: Sequence[Any],
    fits_image: Path,
    runner: SASRunner,
    log_dir: Path,
    inst: str,
) -> str:
    """Convert one or more ``SkyRegion`` shapes into a single ESAS filter string."""
    wcs_ref = WCS(fits.getheader(fits_image)) if is_nonempty_file(fits_image) else None
    parts = [
        sky_region_to_esas_expr(
            reg, fits_image, wcs_ref, runner, log_dir, f"{inst}_{idx}"
        )
        for idx, reg in enumerate(sky_regions)
    ]
    if len(parts) == 1:
        return parts[0]
    terms = [p.removeprefix("&&") for p in parts]
    return "&&(" + " || ".join(terms) + ")"
