"""SAS QPB-subtracted flux imaging via ``evqpb``, ``eimageget``, ``eimagecombine``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

from astropy.io import fits
import numpy as np

from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.exec.command import SASCommand
from xmm_tools.exec.runner import SASCommandError, SASRunner
from xmm_tools.pipeline.config import AnaConfig, BandSpec
from xmm_tools.pipeline.stage import (
    StageManifest,
    all_nonempty,
    read_log_tail,
    read_manifest_json,
    relativize_manifest_paths,
    resolve_manifest_paths,
)

_INST_PRODUCT_TOKENS: dict[str, str] = {
    "mos1": "M1",
    "mos2": "M2",
    "pn": "PN",
}


@dataclass
class QPBImageResult(StageManifest):
    """Core artifacts produced by ``run_qpb_image``."""

    qpb_image_dir: Path
    manifest_path: Path
    qpb_files: dict[str, Path] = field(default_factory=dict)
    band_flux_images: dict[str, Path] = field(default_factory=dict)
    subband_count_images: dict[str, list[Path]] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "stage": "qpb_image",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "qpb_files": {
                k: str(p.relative_to(self.qpb_image_dir))
                if p.is_relative_to(self.qpb_image_dir)
                else str(p)
                for k, p in self.qpb_files.items()
            },
            "band_flux_images": relativize_manifest_paths(self.band_flux_images),
        }

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> QPBImageResult:
        raw = read_manifest_json(manifest_path)
        qdir = manifest_path.parent
        return cls(
            qpb_image_dir=qdir,
            manifest_path=manifest_path,
            qpb_files=resolve_manifest_paths(qdir, raw, "qpb_files"),
            band_flux_images=resolve_manifest_paths(qdir, raw, "band_flux_images"),
        )


def resolve_subbands(band: BandSpec, default_edges: list[int]) -> list[tuple[int, int]]:
    """Return narrow sub-bands for ``band``, splitting along ``default_edges`` if needed."""
    if band.subbands:
        return [(int(lo), int(hi)) for lo, hi in band.subbands]

    inner = sorted(e for e in default_edges if band.emin < e < band.emax)
    pts = [band.emin, *inner, band.emax]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _resolve_attfile(
    cfg: AnaConfig,
    work_dir: Path,
    runner: SASRunner,
    *,
    force: bool,
) -> Path:
    """Locate an existing attitude FITS file or generate one via ``atthkgen``."""
    if cfg.qpb_image.attfile:
        return cfg.paths.resolve(cfg.qpb_image.attfile)

    cheese_dir = cfg.paths.resolve(cfg.paths.cheese_dir)
    candidates = [
        cheese_dir / "_intermediate" / "atthk.fits",
        cheese_dir / "atthk.fits",
        cfg.paths.workdir / "atthk.fits",
        work_dir / "atthk.fits",
    ]
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand

    cmd = SASCommand(
        program="atthkgen",
        cwd=work_dir,
        params={"atthkset": "atthk.fits", "timestep": 1.0},
        core_outputs=["atthk.fits"],
        intermediate_dir=None,
        log_file="atthkgen.log",
    )
    runner.run(cmd, force=force)
    return work_dir / "atthk.fits"


def _resolve_raw_events(
    cfg: AnaConfig, *, dry_run: bool
) -> tuple[dict[str, Path], Path | None]:
    """Resolve raw (un-flag-filtered) event lists from ``[instruments]``."""
    raw_events: dict[str, Path] = {}
    for inst, evt_name in (
        ("mos1", cfg.instruments.mos1),
        ("mos2", cfg.instruments.mos2),
        ("pn", cfg.instruments.pn),
    ):
        if not evt_name:
            continue
        p = cfg.paths.resolve(evt_name)
        if not dry_run and (not p.exists() or p.stat().st_size == 0):
            raise FileNotFoundError(
                f"Raw event file for {inst!r} not found or empty: {p}"
            )
        raw_events[inst] = p

    if not raw_events:
        raise ValueError("No active instruments configured in [instruments]")

    pn_oot: Path | None = None
    if cfg.instruments.pn and cfg.instruments.pn_oot:
        p_oot = cfg.paths.resolve(cfg.instruments.pn_oot)
        if not dry_run and (not p_oot.exists() or p_oot.stat().st_size == 0):
            raise FileNotFoundError(
                f"Raw PN OOT event file not found or empty: {p_oot}"
            )
        pn_oot = p_oot

    return raw_events, pn_oot


def _resolve_gti_files(
    cfg: AnaConfig,
    active_insts: list[str],
    flare_res: FlareResult,
    *,
    dry_run: bool,
) -> dict[str, Path]:
    """Resolve required GTI files from ``[sas.qpb_image.gti_files]`` or ``flare``."""
    gti_files: dict[str, Path] = {}
    for inst in active_insts:
        override = cfg.qpb_image.gti_files.get(inst)
        if override:
            gti_path = cfg.paths.resolve(override)
        else:
            flare_gti = flare_res.gti_files.get(inst)
            if flare_gti is None:
                raise FileNotFoundError(
                    f"Missing GTI file for {inst!r}: configure "
                    f"[sas.qpb_image.gti_files].{inst} in ana.toml or run "
                    "'xmm-ana flare' first."
                )
            gti_path = flare_gti

        if not dry_run and (not gti_path.exists() or gti_path.stat().st_size == 0):
            raise FileNotFoundError(
                f"GTI file for {inst!r} does not exist or is empty: {gti_path}"
            )
        gti_files[inst] = gti_path
    return gti_files


def _discover_eimageget_outputs(
    band_dir: Path,
    inst: str,
    *,
    pre_existing: set[str] | None = None,
) -> list[Path]:
    """Discover ``eimageget`` FITS outputs in ``band_dir`` for ``inst``."""
    if not band_dir.exists():
        return []

    ima_files = sorted(
        p for p in band_dir.glob("*_ima_*.fits") if p.is_file() and p.stat().st_size > 0
    )
    if pre_existing is not None:
        newly_created = [p for p in ima_files if p.name not in pre_existing]
        if newly_created:
            return newly_created

    token = _INST_PRODUCT_TOKENS.get(inst, inst.upper())
    return [p for p in ima_files if token in p.name]


def _discover_combined_subband_pairs(
    band_dir: Path, prefix: str
) -> tuple[list[tuple[Path, Path]], Path | None]:
    """Discover ``eimagecombine`` ``<prefix>ima_<i>_sub.fits`` + ``exp.fits`` pairs."""
    if not band_dir.exists():
        return [], None

    pat = re.compile(rf"^{re.escape(prefix)}ima_(\d+)_sub\.fits$")
    indexed: list[tuple[int, Path, Path]] = []
    for p in band_dir.iterdir():
        if not p.is_file() or p.stat().st_size == 0:
            continue
        m = pat.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        exp_path = band_dir / f"{prefix}ima_{idx}exp.fits"
        if exp_path.is_file() and exp_path.stat().st_size > 0:
            indexed.append((idx, p, exp_path))

    indexed.sort(key=lambda item: item[0])
    pairs = [(sub_p, exp_p) for _, sub_p, exp_p in indexed]
    mask_cand = band_dir / f"{prefix}ima_mask.fits"
    mask_path = (
        mask_cand if mask_cand.is_file() and mask_cand.stat().st_size > 0 else None
    )
    return pairs, mask_path


def _merge_subband_sub_exp_images(
    subband_pairs: list[tuple[Path, Path]],
    mask_path: Path | None,
    out_path: Path,
) -> Path:
    """Compute exposure-corrected rate ``sum_i (C_i / E_i)`` across sub-bands."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_valid: np.ndarray | None = None
    if mask_path is not None:
        with fits.open(mask_path) as mhdul:
            mdata = np.asarray(mhdul[0].data, dtype=np.float64)
            mask_valid = np.isfinite(mdata) & (mdata > 0)

    total: np.ndarray | None = None
    header: fits.Header | None = None

    for sub_path, exp_path in subband_pairs:
        with fits.open(sub_path) as shdul, fits.open(exp_path) as ehdul:
            counts = np.asarray(shdul[0].data, dtype=np.float64)
            exp = np.asarray(ehdul[0].data, dtype=np.float64)
            if header is None:
                header = shdul[0].header.copy()
                total = np.zeros_like(counts, dtype=np.float64)

            valid = np.isfinite(counts) & np.isfinite(exp) & (exp > 0)
            if mask_valid is not None and mask_valid.shape == valid.shape:
                valid &= mask_valid
            assert total is not None
            total[valid] += counts[valid] / exp[valid]

    assert total is not None and header is not None
    header["BUNIT"] = ("counts/s", "Exposure-corrected QPB-subtracted rate")
    header.add_history(
        "Computed sum_i(sub_i / exp_i) from eimagecombine outputs: "
        + ", ".join(sub_p.name for sub_p, _ in subband_pairs)
    )
    fits.PrimaryHDU(data=total.astype(np.float32), header=header).writeto(
        out_path, overwrite=True
    )
    return out_path


def run_qpb_image(
    cfg: AnaConfig,
    *,
    flare: FlareResult | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> QPBImageResult:
    """Generate QPB-subtracted, exposure-corrected flux images via ``evqpb`` + ``eimageget`` + ``eimagecombine``."""
    qi = cfg.qpb_image
    qdir = cfg.paths.resolve(qi.dir)
    work_dir = qdir / "_work"
    manifest_path = qdir / "manifest.json"

    if not force and manifest_path.exists():
        existing = QPBImageResult.from_manifest(manifest_path)
        if all(b in existing.band_flux_images for b in qi.bands) and all_nonempty(
            existing.band_flux_images
        ):
            return existing

    raw_events, pn_oot_path = _resolve_raw_events(cfg, dry_run=dry_run)
    active_insts = [k for k in ("mos1", "mos2", "pn") if k in raw_events]
    flare_res = flare if flare is not None else load_flare_result(cfg)
    gti_files = _resolve_gti_files(cfg, active_insts, flare_res, dry_run=dry_run)

    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=not force,
        docker_container=docker_container,
    )

    attfile = _resolve_attfile(cfg, work_dir, runner, force=force)

    # 1. Generate QPB/FWC event lists once per active instrument from raw events
    qpb_files: dict[str, Path] = {}
    for inst in active_insts:
        qpb_name = f"{inst}_qpb.fits"
        qpb_params: dict[str, Any] = {
            "table": raw_events[inst],
            "attfile": attfile,
            "outset": qpb_name,
        }
        qpb_params.update(qi.evqpb_extra_params)

        runner.run(
            SASCommand(
                program="evqpb",
                cwd=work_dir,
                params=qpb_params,
                core_outputs=[qpb_name],
                intermediate_dir=None,
                log_file=f"{inst}_evqpb.log",
            ),
            force=force,
        )
        qpb_files[inst] = work_dir / qpb_name

    # 2. Run eimageget (all sub-bands per instrument) + eimagecombine per target band
    band_flux_images: dict[str, Path] = {}
    subband_count_images: dict[str, list[Path]] = {}

    for band_name, band in qi.bands.items():
        subbands = resolve_subbands(band, qi.default_subband_edges)
        pimin_list = [lo for lo, _ in subbands]
        pimax_list = [hi for _, hi in subbands]
        band_dir = work_dir / band_name
        if force and not dry_run and band_dir.exists():
            shutil.rmtree(band_dir)
        band_outputs: list[Path] = []

        for inst in active_insts:
            existing_ima = (
                [] if force or dry_run else _discover_eimageget_outputs(band_dir, inst)
            )
            if existing_ima:
                band_outputs.extend(existing_ima)
                continue

            pre_existing = (
                {p.name for p in band_dir.iterdir()} if band_dir.exists() else set()
            )
            eimg_params: dict[str, Any] = {
                "evtfile": raw_events[inst],
                "fwcfile": qpb_files[inst],
                "gtifile": gti_files[inst],
                "attfile": attfile,
                "pimin": pimin_list,
                "pimax": pimax_list,
                "ximagebinsize": qi.bin_size,
                "yimagebinsize": qi.bin_size,
                "withexposure": True,
                "withfwcimages": True,
                "withmask": True,
            }
            if inst == "pn" and pn_oot_path is not None:
                eimg_params["ootfile"] = pn_oot_path
            eimg_params.update(qi.eimageget_extra_params)

            log_name = f"{inst}_eimageget.log"
            runner.run(
                SASCommand(
                    program="eimageget",
                    cwd=band_dir,
                    params=eimg_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file=log_name,
                ),
                force=force,
            )

            if not dry_run:
                produced = _discover_eimageget_outputs(
                    band_dir, inst, pre_existing=pre_existing
                )
                if not produced:
                    tail = read_log_tail(band_dir / log_name)
                    raise SASCommandError(
                        f"eimageget finished for {inst} ({band_name}) but no "
                        f"'*_ima_*.fits' outputs were found in {band_dir}.\n"
                        f"Log: {band_dir / log_name}\n{tail}"
                    )
                band_outputs.extend(produced)

        subband_count_images[band_name] = band_outputs

        # 3. Combine instruments per sub-band in band_dir via eimagecombine
        out_prefix = f"{band_name}_{band.emin}_{band.emax}_"
        flux_name = f"{band_name}_{band.emin}_{band.emax}_flux.fits"
        flux_path = qdir / flux_name

        comb_params: dict[str, Any] = {
            "prefix": out_prefix,
            "withcheckinput": True,
            "withaddimages": True,
            "withcombineimages": True,
            "withasmooth": qi.with_asmooth,
            "keepinterstage": True,
        }
        comb_params.update(qi.eimagecombine_extra_params)

        combine_log = f"{band_name}_combine.log"
        existing_pairs, _ = (
            ([], None)
            if force or dry_run
            else _discover_combined_subband_pairs(band_dir, out_prefix)
        )
        if not existing_pairs:
            runner.run(
                SASCommand(
                    program="eimagecombine",
                    cwd=band_dir,
                    params=comb_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file=combine_log,
                ),
                force=force,
            )

        if not dry_run:
            pairs, mask_path = _discover_combined_subband_pairs(band_dir, out_prefix)
            if not pairs:
                tail = read_log_tail(band_dir / combine_log)
                raise SASCommandError(
                    f"eimagecombine finished for {band_name} but no "
                    f"'{out_prefix}ima_<i>_sub.fits' + 'exp.fits' pairs were "
                    f"found in {band_dir}.\nLog: {band_dir / combine_log}\n{tail}"
                )
            _merge_subband_sub_exp_images(pairs, mask_path, flux_path)

        band_flux_images[band_name] = flux_path

    result = QPBImageResult(
        qpb_image_dir=qdir,
        manifest_path=manifest_path,
        qpb_files=qpb_files,
        band_flux_images=band_flux_images,
        subband_count_images=subband_count_images,
    )
    if not dry_run:
        result.save_manifest()
    return result
