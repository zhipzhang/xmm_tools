"""ESAS spectrum extraction (``mosspectra``/``pnspectra`` + ``mosback``/``pnback``).

Workflow per region job:

1. Expand ``SpectrumRegionSpec`` (``single`` / ``split`` / ``union``) from DS9 ``.reg``
   or pre-converted SAS ``DETX/DETY`` ``.txt`` files.
2. Convert DS9 sky regions to per-instrument ``DETX/DETY`` expressions using
   ``evselect`` reference images + ``ecoordconv`` (1 arcsec = 20.0 DET units).
3. Symlink cleaned event lists and cheese masks into an isolated region directory
   and pre-validate event counts with ``evselect`` (0-event instruments are skipped).
4. Run ``mosspectra`` + ``mosback`` and ``pnspectra`` + ``pnback`` via ``SASRunner``,
   concurrently across regions via ``ThreadPoolExecutor``.
5. Incrementally upsert all region results into ``spectrum/manifest.json``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from xmm_tools.esas.config import ESASConfig, SpectrumConfig, SpectrumRegionSpec
from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.esas.source_detect import SourceMaskResult, load_source_mask_result
from xmm_tools.sas.command import SASCommand
from xmm_tools.sas.runner import SASRunner

DET_SCALE = 20.0
_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_ECOORD_RE = re.compile(rf"DETX:\s*DETY:\s*({_NUMBER_RE})\s+({_NUMBER_RE})")
_PROTON_AREA_RE = re.compile(rf"Area\s*:\s*({_NUMBER_RE})")
_PROTON_FLUX_RE = re.compile(rf"Ave Flux\s*:\s*({_NUMBER_RE})")
_PROTON_SCALED_RE = re.compile(rf"Scaled Ave Flux\s*:\s*({_NUMBER_RE})")
_QDP_KEYS = ("bkgspec", "augspec", "ratehard")


@dataclass
class InstrumentProducts:
    """Extracted spectrum & background products for one instrument in one region."""

    fovt: Path | None = None
    fovtoot: Path | None = None
    fovtootsub: Path | None = None
    bkg: Path | None = None
    rmf: Path | None = None
    arf: Path | None = None
    qdp_plots: dict[str, Path] = field(default_factory=dict)
    region_file: Path | None = None
    event_count: int | None = None
    proton_area: float | None = None
    proton_ave_flux: float | None = None
    proton_scale: float | None = None
    skipped: bool = False
    skip_reason: str = ""

    def core_files_ready(self, inst: str) -> bool:
        """Return True when this instrument is skipped or all core files exist."""
        if self.skipped:
            return True
        required = [self.fovt, self.bkg, self.rmf, self.arf]
        if inst == "pn":
            required.extend([self.fovtoot, self.fovtootsub])
        return all(p is not None and p.exists() and p.stat().st_size > 0 for p in required)

    def to_manifest_dict(self, base_dir: Path) -> dict[str, Any]:
        def _rel(p: Path | None) -> str | None:
            if p is None:
                return None
            return str(p.relative_to(base_dir)) if p.is_relative_to(base_dir) else str(p)

        return {
            "fovt": _rel(self.fovt),
            "fovtoot": _rel(self.fovtoot),
            "fovtootsub": _rel(self.fovtootsub),
            "bkg": _rel(self.bkg),
            "rmf": _rel(self.rmf),
            "arf": _rel(self.arf),
            "qdp_plots": {k: _rel(v) for k, v in self.qdp_plots.items()},
            "region_file": _rel(self.region_file),
            "event_count": self.event_count,
            "proton_area": self.proton_area,
            "proton_ave_flux": self.proton_ave_flux,
            "proton_scale": self.proton_scale,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_manifest_dict(cls, raw: dict[str, Any], base_dir: Path) -> InstrumentProducts:
        def _abs(val: str | None) -> Path | None:
            if not val:
                return None
            p = Path(val)
            return p if p.is_absolute() else base_dir / p

        qdp_raw = raw.get("qdp_plots", {})
        return cls(
            fovt=_abs(raw.get("fovt")),
            fovtoot=_abs(raw.get("fovtoot")),
            fovtootsub=_abs(raw.get("fovtootsub")),
            bkg=_abs(raw.get("bkg")),
            rmf=_abs(raw.get("rmf")),
            arf=_abs(raw.get("arf")),
            qdp_plots={k: p for k, v in qdp_raw.items() if (p := _abs(v)) is not None},
            region_file=_abs(raw.get("region_file")),
            event_count=raw.get("event_count"),
            proton_area=raw.get("proton_area"),
            proton_ave_flux=raw.get("proton_ave_flux"),
            proton_scale=raw.get("proton_scale"),
            skipped=bool(raw.get("skipped", False)),
            skip_reason=str(raw.get("skip_reason", "")),
        )


@dataclass
class RegionSpectrumResult:
    """Spectrum extraction result for a single region directory."""

    key: str
    group: str
    label: str
    mode: str
    region_dir: Path
    source_file: Path
    elow: int
    ehigh: int
    instruments: dict[str, InstrumentProducts] = field(default_factory=dict)

    def all_ready(self) -> bool:
        return bool(self.instruments) and all(
            prod.core_files_ready(inst) for inst, prod in self.instruments.items()
        )

    def to_manifest_dict(self, spectrum_dir: Path) -> dict[str, Any]:
        rel_dir = (
            str(self.region_dir.relative_to(spectrum_dir))
            if self.region_dir.is_relative_to(spectrum_dir)
            else str(self.region_dir)
        )
        return {
            "group": self.group,
            "label": self.label,
            "mode": self.mode,
            "dir": rel_dir,
            "source_file": str(self.source_file),
            "elow": self.elow,
            "ehigh": self.ehigh,
            "instruments": {
                inst: prod.to_manifest_dict(self.region_dir)
                for inst, prod in self.instruments.items()
            },
        }

    @classmethod
    def from_manifest_dict(
        cls, key: str, raw: dict[str, Any], spectrum_dir: Path
    ) -> RegionSpectrumResult:
        rdir = Path(raw["dir"])
        if not rdir.is_absolute():
            rdir = spectrum_dir / rdir
        insts = {
            inst: InstrumentProducts.from_manifest_dict(pdata, rdir)
            for inst, pdata in raw.get("instruments", {}).items()
        }
        return cls(
            key=key,
            group=str(raw.get("group", key)),
            label=str(raw.get("label", key)),
            mode=str(raw.get("mode", "single")),
            region_dir=rdir,
            source_file=Path(raw.get("source_file", "")),
            elow=int(raw.get("elow", 400)),
            ehigh=int(raw.get("ehigh", 12500)),
            instruments=insts,
        )


@dataclass
class SpectrumResult:
    """Root manifest contract for ``spectrum/manifest.json``."""

    spectrum_dir: Path
    manifest_path: Path
    regions: dict[str, RegionSpectrumResult] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "stage": "spectrum",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "regions": {
                key: reg.to_manifest_dict(self.spectrum_dir)
                for key, reg in sorted(self.regions.items())
            },
        }

    def save_manifest(self) -> Path:
        self.spectrum_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.to_manifest_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> SpectrumResult:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        sdir = manifest_path.parent
        regions = {
            key: RegionSpectrumResult.from_manifest_dict(key, rdata, sdir)
            for key, rdata in raw.get("regions", {}).items()
        }
        return cls(spectrum_dir=sdir, manifest_path=manifest_path, regions=regions)


def load_spectrum_result(cfg: ESASConfig) -> SpectrumResult:
    """Load ``spectrum`` outputs from ``spectrum/manifest.json``."""
    sdir = cfg.paths.resolve(cfg.paths.spectrum_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Spectrum manifest not found at {manifest_path}; run 'xmm-esas spectrum' first."
        )
    return SpectrumResult.from_manifest(manifest_path)


@dataclass
class _ExpandedRegionJob:
    key: str
    group: str
    label: str
    mode: str
    region_dir: Path
    source_file: Path
    overwrite: bool
    elow: int
    ehigh: int
    sky_regions: list[Any] | None = None
    det_expr_text: str | None = None


def _sanitize_label(raw_label: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label.strip()).strip("_")
    return cleaned or fallback


def _import_regions(regions_path: Path | str | None = None) -> tuple[Any, ...]:
    """Import ``astropy-regions``, falling back to ``regions_path`` if needed."""
    def _load() -> tuple[Any, ...]:
        from regions import (
            CircleSkyRegion,
            EllipseSkyRegion,
            PolygonSkyRegion,
            RectangleSkyRegion,
            Regions,
        )
        return (
            Regions,
            CircleSkyRegion,
            EllipseSkyRegion,
            RectangleSkyRegion,
            PolygonSkyRegion,
        )

    try:
        return _load()
    except ImportError as exc:
        fallback = regions_path or os.environ.get("XMM_REGIONS_PATH")
        if fallback and str(fallback).strip():
            rpath = str(Path(str(fallback).strip()))
            if rpath not in sys.path:
                sys.path.insert(0, rpath)
            try:
                return _load()
            except ImportError as inner_exc:
                raise ImportError(
                    f"Could not import 'regions' even after adding regions_path={rpath!r} "
                    "to sys.path. Install 'regions' or verify [env].regions_path in esas.toml."
                ) from inner_exc
        raise ImportError(
            "DS9 sky region conversion requires 'regions'. "
            "Install 'regions' or set [env].regions_path in esas.toml."
        ) from exc


def _expand_region_spec(
    spec: SpectrumRegionSpec,
    cfg: ESASConfig,
    spectrum_dir: Path,
) -> list[_ExpandedRegionJob]:
    src_path = cfg.paths.resolve(spec.file)
    if not src_path.exists():
        raise FileNotFoundError(f"Region file not found for {spec.name!r}: {src_path}")

    elow = spec.elow if spec.elow is not None else cfg.spectrum.elow
    ehigh = spec.ehigh if spec.ehigh is not None else cfg.spectrum.ehigh

    # Allow pre-converted SAS DETX/DETY expression files (.txt) directly
    if src_path.suffix.lower() == ".txt":
        expr = src_path.read_text(encoding="utf-8").strip()
        if not expr:
            raise ValueError(f"Empty SAS detector expression file: {src_path}")
        return [
            _ExpandedRegionJob(
                key=spec.name,
                group=spec.name,
                label=spec.name,
                mode=spec.mode,
                region_dir=spectrum_dir / spec.name,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                det_expr_text=expr,
            )
        ]

    Regions, *_ = _import_regions(cfg.env.regions_path)
    parsed = list(Regions.read(str(src_path), format="ds9"))
    if not parsed:
        raise ValueError(f"No DS9 regions parsed from {src_path}")

    if spec.mode == "single":
        if len(parsed) > 1:
            raise ValueError(
                f"Region {spec.name!r} ({src_path.name}) contains {len(parsed)} shapes, "
                "but mode='single'. Use mode='union' to merge them or mode='split' to batch."
            )
        return [
            _ExpandedRegionJob(
                key=spec.name,
                group=spec.name,
                label=spec.name,
                mode="single",
                region_dir=spectrum_dir / spec.name,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                sky_regions=parsed,
            )
        ]

    if spec.mode == "union":
        return [
            _ExpandedRegionJob(
                key=spec.name,
                group=spec.name,
                label=spec.name,
                mode="union",
                region_dir=spectrum_dir / spec.name,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                sky_regions=parsed,
            )
        ]

    # mode == "split"
    jobs: list[_ExpandedRegionJob] = []
    seen_labels: set[str] = set()
    for idx, reg in enumerate(parsed, start=1):
        meta_text = str(reg.meta.get("text", "") or "") if hasattr(reg, "meta") else ""
        label = _sanitize_label(meta_text, f"r{idx}")
        if label in seen_labels:
            label = f"{label}_{idx}"
        seen_labels.add(label)
        jobs.append(
            _ExpandedRegionJob(
                key=f"{spec.name}/{label}",
                group=spec.name,
                label=label,
                mode="split",
                region_dir=spectrum_dir / spec.name / label,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                sky_regions=[reg],
            )
        )
    return jobs


def _ensure_ref_sky_image(
    inst: str,
    event_file: Path,
    inter_dir: Path,
    runner: SASRunner,
    *,
    force: bool,
) -> Path:
    """Create one shared 600x600 X/Y image per instrument for ``ecoordconv``."""
    img_path = inter_dir / f"ref_{inst}.fits"
    cmd = SASCommand(
        program="evselect",
        cwd=inter_dir,
        params={
            "table": str(event_file),
            "withimageset": "yes",
            "imageset": img_path.name,
            "xcolumn": "X",
            "ycolumn": "Y",
            "imagebinning": "imageSize",
            "ximagesize": 600,
            "yimagesize": 600,
        },
        core_outputs=[img_path.name],
        intermediate_dir=None,
        log_file=f"ref_img_{inst}.log",
    )
    runner.run(cmd, force=force)
    return img_path


def _run_ecoordconv(
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
            "imageset": str(fits_image),
            "x": f"{ra_deg:.8f}",
            "y": f"{dec_deg:.8f}",
            "coordtype": "eqpos",
        },
        core_outputs=[],
        intermediate_dir=None,
        log_file=f"ecoordconv_{tag}.log",
    )
    res = runner.run(cmd, force=True)
    if res.status == "dry_run" or res.log_path is None or not res.log_path.exists():
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
        detx, dety = _run_ecoordconv(
            fits_image, sky_reg.center.ra.deg, sky_reg.center.dec.deg, runner, log_dir, tag
        )
        radius = sky_reg.radius.to(u.arcsec).value * DET_SCALE
        return f"&&((DETX,DETY) IN circle({detx:.1f},{dety:.1f},{radius:.1f}))"

    if isinstance(sky_reg, EllipseSkyRegion):
        if wcs_ref is None:
            wcs_ref = WCS(fits.getheader(fits_image))
        detx0, dety0 = _run_ecoordconv(
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
        detx1, dety1 = _run_ecoordconv(
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
            vertices = wcs_ref.pixel_to_world(pix_reg.corners[:, 0], pix_reg.corners[:, 1])
        det_coords: list[str] = []
        for idx, vertex in enumerate(vertices):
            detx, dety = _run_ecoordconv(
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


def _symlink_relative(source: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), link_path.parent.resolve())
    if link_path.is_symlink():
        if link_path.resolve() == source.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        return
    link_path.symlink_to(target)


def _count_region_events(
    event_file: Path,
    expr_line: str,
    runner: SASRunner,
    work_dir: Path,
    inst: str,
) -> int:
    """Count events inside ``expr_line`` using ``evselect``."""
    filtered = work_dir / f"_val_{inst}.fits"
    bare_expr = expr_line[2:] if expr_line.startswith("&&") else expr_line
    cmd = SASCommand(
        program="evselect",
        cwd=work_dir,
        params={
            "table": event_file.name,
            "expression": bare_expr,
            "withfilteredset": "yes",
            "keepfilteroutput": "yes",
            "filteredset": filtered.name,
        },
        core_outputs=[filtered.name],
        intermediate_dir=None,
        log_file=f"validate_{inst}.log",
    )
    res = runner.run(cmd, force=True)
    if res.status == "dry_run" or not filtered.exists():
        return 1
    try:
        with fits.open(filtered) as hdul:
            return len(hdul["EVENTS"].data)
    finally:
        if filtered.exists():
            filtered.unlink()


def _find_prefix(region_dir: Path, inst: str) -> str:
    """Discover ESAS exposure prefix (e.g. ``mos1S001``) from generated ``*-fovt.pi``."""
    matches = sorted(region_dir.glob(f"{inst}*-fovt.pi"))
    if matches:
        return matches[0].name[: -len("-fovt.pi")]
    defaults = {"mos1": "mos1S001", "mos2": "mos2S002", "pn": "pnS003"}
    return defaults[inst]


def _run_protonscale(
    region_dir: Path,
    prefix: str,
    inst: str,
    runner: SASRunner,
    *,
    force: bool,
) -> tuple[float | None, float | None, float | None]:
    """Run ``protonscale`` on ``<prefix>-fovimspdet.fits`` and ``<prefix>-fovt.pi``."""
    mask_name = f"{prefix}-fovimspdet.fits"
    spec_name = f"{prefix}-fovt.pi"
    if not (region_dir / mask_name).exists() or not (region_dir / spec_name).exists():
        return None, None, None
    res = runner.run(
        SASCommand(
            program="protonscale",
            cwd=region_dir,
            params={"mode": 1, "maskfile": mask_name, "specfile": spec_name},
            core_outputs=[],
            intermediate_dir=None,
            log_file=f"protonscale_{inst}.log",
        ),
        force=force,
    )
    if res.status == "dry_run" or res.log_path is None or not res.log_path.exists():
        return None, None, None
    text = res.log_path.read_text(encoding="utf-8", errors="replace")
    m_area = _PROTON_AREA_RE.search(text)
    m_flux = _PROTON_FLUX_RE.search(text)
    m_scaled = _PROTON_SCALED_RE.search(text)
    return (
        float(m_area.group(1)) if m_area else None,
        float(m_flux.group(1)) if m_flux else None,
        float(m_scaled.group(1)) if m_scaled else None,
    )


def _collect_instrument_products(
    region_dir: Path,
    inst: str,
    reg_txt: Path,
    count: int,
    *,
    proton_area: float | None = None,
    proton_ave_flux: float | None = None,
    proton_scale: float | None = None,
) -> InstrumentProducts:
    prefix = _find_prefix(region_dir, inst)

    def _opt(suffix: str) -> Path | None:
        p = region_dir / f"{prefix}{suffix}"
        return p if p.exists() and p.stat().st_size > 0 else None

    qdp_plots = {
        k: p for k in _QDP_KEYS if (p := _opt(f"-{k}.qdp")) is not None
    }
    return InstrumentProducts(
        fovt=_opt("-fovt.pi"),
        fovtoot=_opt("-fovtoot.pi") if inst == "pn" else None,
        fovtootsub=_opt("-fovtootsub.pi") if inst == "pn" else None,
        bkg=_opt("-bkg.pi"),
        rmf=_opt(".rmf"),
        arf=_opt(".arf"),
        qdp_plots=qdp_plots,
        region_file=reg_txt,
        event_count=count,
        proton_area=proton_area,
        proton_ave_flux=proton_ave_flux,
        proton_scale=proton_scale,
        skipped=False,
    )


def _run_single_region_job(
    job: _ExpandedRegionJob,
    *,
    spec_cfg: SpectrumConfig,
    clean_events: dict[str, Path],
    mask_res: SourceMaskResult | None,
    ref_images: dict[str, Path],
    runner: SASRunner,
    dry_run: bool,
    force: bool,
) -> RegionSpectrumResult:
    rdir = job.region_dir
    active_insts = [i for i in ("mos1", "mos2", "pn") if i in clean_events]
    if dry_run:
        return RegionSpectrumResult(
            key=job.key,
            group=job.group,
            label=job.label,
            mode=job.mode,
            region_dir=rdir,
            source_file=job.source_file,
            elow=job.elow,
            ehigh=job.ehigh,
            instruments={
                inst: InstrumentProducts(region_file=rdir / f"reg_{inst}.txt")
                for inst in active_insts
            },
        )

    rdir.mkdir(parents=True, exist_ok=True)
    log_dir = rdir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save provenance region.reg when split from a multi-region file
    if job.sky_regions and job.mode == "split":
        Regions, *_ = _import_regions()
        Regions(job.sky_regions).write(str(rdir / "region.reg"), format="ds9", overwrite=True)

    # Symlink event lists and optional cheese masks into region_dir
    local_events: dict[str, Path] = {}
    for inst in active_insts:
        src_evt = clean_events[inst]
        dst_evt = rdir / src_evt.name
        _symlink_relative(src_evt, dst_evt)
        local_events[inst] = dst_evt
    if "pn_oot" in clean_events:
        _symlink_relative(clean_events["pn_oot"], rdir / clean_events["pn_oot"].name)

    local_masks_det: dict[str, Path] = {}
    local_masks_sky: dict[str, Path] = {}
    if spec_cfg.withsrcrem and mask_res is not None:
        for inst in active_insts:
            for suffix, reg_map, store in (
                ("det", mask_res.region_det, local_masks_det),
                ("sky", mask_res.region_sky, local_masks_sky),
            ):
                src_m = reg_map.get(inst) or mask_res.cheese_masks.get(f"{inst}_{suffix}")
                if src_m is not None:
                    dst_m = rdir / src_m.name
                    _symlink_relative(src_m, dst_m)
                    store[inst] = dst_m

    # Convert sky regions -> DETX/DETY expressions per instrument
    expr_files: dict[str, Path] = {}
    expr_texts: dict[str, str] = {}
    for inst in active_insts:
        reg_txt = rdir / f"reg_{inst}.txt"
        if job.det_expr_text is not None:
            expr = job.det_expr_text
        else:
            ref_img = ref_images[inst]
            wcs_ref = None if dry_run or not ref_img.exists() else WCS(fits.getheader(ref_img))
            parts = [
                sky_region_to_esas_expr(
                    reg, ref_img, wcs_ref, runner, log_dir, f"{inst}_{idx}"
                )
                for idx, reg in enumerate(job.sky_regions or [])
            ]
            if len(parts) == 1:
                expr = parts[0]
            else:
                terms = [p[2:] if p.startswith("&&") else p for p in parts]
                expr = "&&(" + " || ".join(terms) + ")"
        if not dry_run:
            reg_txt.write_text(expr + "\n", encoding="utf-8")
        expr_files[inst] = reg_txt
        expr_texts[inst] = expr

    # Pre-validate event counts & run spectra/back per instrument
    inst_results: dict[str, InstrumentProducts] = {}
    val_lines: list[str] = []
    for inst in active_insts:
        count = _count_region_events(
            local_events[inst], expr_texts[inst], runner, rdir, inst
        )
        val_lines.append(f"{inst}\t{count}")
        if count == 0:
            inst_results[inst] = InstrumentProducts(
                region_file=expr_files[inst],
                event_count=0,
                skipped=True,
                skip_reason="zero events in detector region",
            )
            continue

        if inst in ("mos1", "mos2"):
            ccds = spec_cfg.mos1_ccds if inst == "mos1" else spec_cfg.mos2_ccds
            spec_params: dict[str, Any] = {
                "eventfile": local_events[inst].name,
                "withsrcrem": "yes" if spec_cfg.withsrcrem else "no",
                "withregion": "yes",
                "regionfile": expr_files[inst].name,
                "keepinterfiles": "yes" if spec_cfg.keepinterfiles else "no",
                "pattern": spec_cfg.pattern_mos,
                "elow": job.elow,
                "ehigh": job.ehigh,
                "ccds": ccds,
                **spec_cfg.spectra_extra_params,
            }
            if spec_cfg.withsrcrem and inst in local_masks_det and inst in local_masks_sky:
                spec_params["maskdet"] = local_masks_det[inst].name
                spec_params["masksky"] = local_masks_sky[inst].name

            runner.run(
                SASCommand(
                    program="mosspectra",
                    cwd=rdir,
                    params=spec_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file=f"mosspectra_{inst}.log",
                ),
                force=force,
            )
            prefix = _find_prefix(rdir, inst)
            back_params: dict[str, Any] = {
                "inspecfile": f"{prefix}-fovt.pi",
                "withplotfiles": "yes" if spec_cfg.withplotfiles else "no",
                "elow": job.elow,
                "ehigh": job.ehigh,
                "ccds": ccds,
                **spec_cfg.back_extra_params,
            }
            runner.run(
                SASCommand(
                    program="mosback",
                    cwd=rdir,
                    params=back_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file=f"mosback_{inst}.log",
                ),
                force=force,
            )
        else:
            oot_name = clean_events["pn_oot"].name if "pn_oot" in clean_events else ""
            spec_params = {
                "eventfile": local_events["pn"].name,
                "ootevtfile": oot_name,
                "withsrcrem": "yes" if spec_cfg.withsrcrem else "no",
                "withregion": "yes",
                "regionfile": expr_files["pn"].name,
                "keepinterfiles": "yes" if spec_cfg.keepinterfiles else "no",
                "pattern": spec_cfg.pattern_pn,
                "elow": job.elow,
                "ehigh": job.ehigh,
                "quads": spec_cfg.pn_quads,
                **spec_cfg.spectra_extra_params,
            }
            if spec_cfg.withsrcrem and "pn" in local_masks_det and "pn" in local_masks_sky:
                spec_params["maskdet"] = local_masks_det["pn"].name
                spec_params["masksky"] = local_masks_sky["pn"].name

            runner.run(
                SASCommand(
                    program="pnspectra",
                    cwd=rdir,
                    params=spec_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file="pnspectra_pn.log",
                ),
                force=force,
            )
            prefix = _find_prefix(rdir, "pn")
            back_params = {
                "inspecfile": f"{prefix}-fovt.pi",
                "inspecoot": f"{prefix}-fovtoot.pi",
                "withplotfiles": "yes" if spec_cfg.withplotfiles else "no",
                "elow": job.elow,
                "ehigh": job.ehigh,
                "quads": spec_cfg.pn_quads,
                **spec_cfg.back_extra_params,
            }
            runner.run(
                SASCommand(
                    program="pnback",
                    cwd=rdir,
                    params=back_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file="pnback_pn.log",
                ),
                force=force,
            )

        p_area, p_flux, p_scaled = _run_protonscale(
            rdir, prefix, inst, runner, force=force
        )
        inst_results[inst] = _collect_instrument_products(
            rdir,
            inst,
            expr_files[inst],
            count,
            proton_area=p_area,
            proton_ave_flux=p_flux,
            proton_scale=p_scaled,
        )

    if not dry_run:
        (rdir / "validation.txt").write_text(
            "\n".join(val_lines) + "\n", encoding="utf-8"
        )

    return RegionSpectrumResult(
        key=job.key,
        group=job.group,
        label=job.label,
        mode=job.mode,
        region_dir=rdir,
        source_file=job.source_file,
        elow=job.elow,
        ehigh=job.ehigh,
        instruments=inst_results,
    )


def run_spectrum(
    cfg: ESASConfig,
    *,
    regions: list[SpectrumRegionSpec] | None = None,
    only: list[str] | None = None,
    flare: FlareResult | None = None,
    mask_res: SourceMaskResult | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> SpectrumResult:
    """Run ``mosspectra``/``pnspectra`` + ``mosback``/``pnback`` and update ``spectrum/manifest.json``.

    Supports incremental updates: existing entries in ``spectrum/manifest.json``
    are preserved and skipped unless ``force=True`` or ``region.overwrite=True``.
    """
    spec_cfg = cfg.spectrum
    target_specs = regions if regions is not None else spec_cfg.regions
    if not target_specs:
        raise ValueError(
            "No spectrum regions configured. Add [[spectrum.regions]] to esas.toml "
            "or pass --reg on the command line."
        )

    sdir = cfg.paths.resolve(cfg.paths.spectrum_dir)
    manifest_path = sdir / "manifest.json"
    result = (
        SpectrumResult.from_manifest(manifest_path)
        if manifest_path.exists()
        else SpectrumResult(spectrum_dir=sdir, manifest_path=manifest_path)
    )

    expanded_jobs: list[_ExpandedRegionJob] = []
    for spec in target_specs:
        expanded_jobs.extend(_expand_region_spec(spec, cfg, sdir))

    if only:
        only_set = set(only)
        expanded_jobs = [
            j for j in expanded_jobs if j.group in only_set or j.key in only_set
        ]

    # Determine which jobs actually need execution
    pending_jobs: list[_ExpandedRegionJob] = []
    for job in expanded_jobs:
        existing = result.regions.get(job.key)
        if (
            not force
            and not job.overwrite
            and existing is not None
            and existing.all_ready()
        ):
            continue
        pending_jobs.append(job)

    if not pending_jobs:
        return result

    flare_res = flare if flare is not None else load_flare_result(cfg)
    if mask_res is None and spec_cfg.withsrcrem:
        mask_res = load_source_mask_result(cfg)

    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=not force,
        docker_container=docker_container,
    )

    inter_dir = sdir / "_intermediate"
    if not dry_run:
        inter_dir.mkdir(parents=True, exist_ok=True)

    # Create one shared sky reference image per instrument if any job needs DS9 conversion
    ref_images: dict[str, Path] = {}
    if any(j.sky_regions is not None for j in pending_jobs):
        for inst in ("mos1", "mos2", "pn"):
            if inst in flare_res.clean_events:
                ref_images[inst] = _ensure_ref_sky_image(
                    inst,
                    flare_res.clean_events[inst],
                    inter_dir,
                    runner,
                    force=force,
                )

    manifest_lock = threading.Lock()
    workers = max(1, min(spec_cfg.max_workers, len(pending_jobs)))

    def _execute_and_record(job: _ExpandedRegionJob) -> RegionSpectrumResult:
        reg_res = _run_single_region_job(
            job,
            spec_cfg=spec_cfg,
            clean_events=flare_res.clean_events,
            mask_res=mask_res,
            ref_images=ref_images,
            runner=runner,
            dry_run=dry_run,
            force=force or job.overwrite,
        )
        if not dry_run:
            with manifest_lock:
                result.regions[reg_res.key] = reg_res
                result.save_manifest()
        return reg_res

    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_and_record, job): job.key for job in pending_jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                reg_res = future.result()
                if dry_run:
                    result.regions[reg_res.key] = reg_res
            except Exception as exc:
                errors.append((key, str(exc)))

    if errors:
        raise RuntimeError(f"Spectrum extraction failed for {errors}")

    return result
