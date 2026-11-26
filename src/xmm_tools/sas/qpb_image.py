"""SAS QPB-subtracted flux imaging via ``evqpb``, ``eimageget``, ``eimagecombine``."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
class SubbandCombinedProducts:
    """Combined ``eimagecombine`` outputs for a single energy sub-band."""

    index: int
    emin: int
    emax: int
    counts_image: Path
    exposure_image: Path
    smoothed_flux_image: Path | None = None

    def to_manifest_dict(self, base_dir: Path) -> dict[str, object]:
        def _rel(p: Path) -> str:
            return (
                str(p.relative_to(base_dir)) if p.is_relative_to(base_dir) else str(p)
            )

        data: dict[str, object] = {
            "index": self.index,
            "emin": self.emin,
            "emax": self.emax,
            "counts_image": _rel(self.counts_image),
            "exposure_image": _rel(self.exposure_image),
        }
        if self.smoothed_flux_image is not None:
            data["smoothed_flux_image"] = _rel(self.smoothed_flux_image)
        return data

    @classmethod
    def from_manifest_dict(
        cls, base_dir: Path, raw: dict[str, Any]
    ) -> SubbandCombinedProducts:
        def _res(val: str) -> Path:
            p = Path(val)
            return p if p.is_absolute() else (base_dir / p)

        smooth_raw = raw.get("smoothed_flux_image")
        return cls(
            index=int(raw["index"]),
            emin=int(raw["emin"]),
            emax=int(raw["emax"]),
            counts_image=_res(str(raw["counts_image"])),
            exposure_image=_res(str(raw["exposure_image"])),
            smoothed_flux_image=_res(str(smooth_raw)) if smooth_raw else None,
        )


@dataclass
class BandCombinedProducts:
    """Structured combined outputs for one configured target energy band."""

    band_name: str
    emin: int
    emax: int
    flux_image: Path
    mask_image: Path | None = None
    subbands: list[SubbandCombinedProducts] = field(default_factory=list)

    def to_manifest_dict(self, base_dir: Path) -> dict[str, object]:
        def _rel(p: Path) -> str:
            return (
                str(p.relative_to(base_dir)) if p.is_relative_to(base_dir) else str(p)
            )

        data: dict[str, object] = {
            "emin": self.emin,
            "emax": self.emax,
            "flux_image": _rel(self.flux_image),
            "subbands": [sb.to_manifest_dict(base_dir) for sb in self.subbands],
        }
        if self.mask_image is not None:
            data["mask_image"] = _rel(self.mask_image)
        return data

    @classmethod
    def from_manifest_dict(
        cls, base_dir: Path, band_name: str, raw: dict[str, Any]
    ) -> BandCombinedProducts:
        def _res(val: str) -> Path:
            p = Path(val)
            return p if p.is_absolute() else (base_dir / p)

        mask_raw = raw.get("mask_image")
        return cls(
            band_name=band_name,
            emin=int(raw["emin"]),
            emax=int(raw["emax"]),
            flux_image=_res(str(raw["flux_image"])),
            mask_image=_res(str(mask_raw)) if mask_raw else None,
            subbands=[
                SubbandCombinedProducts.from_manifest_dict(base_dir, sb)
                for sb in raw.get("subbands", [])
            ],
        )


@dataclass
class QPBImageResult(StageManifest):
    """Core artifacts produced by ``run_qpb_image``."""

    qpb_image_dir: Path
    manifest_path: Path
    qpb_files: dict[str, Path] = field(default_factory=dict)
    band_flux_images: dict[str, Path] = field(default_factory=dict)
    subband_count_images: dict[str, list[Path]] = field(default_factory=dict)
    bands: dict[str, BandCombinedProducts] = field(default_factory=dict)

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
            "bands": {
                bname: bprod.to_manifest_dict(self.qpb_image_dir)
                for bname, bprod in self.bands.items()
            },
        }

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> QPBImageResult:
        raw = read_manifest_json(manifest_path)
        qdir = manifest_path.parent
        raw_bands = raw.get("bands", {})
        bands = (
            {
                str(bname): BandCombinedProducts.from_manifest_dict(
                    qdir, str(bname), bdict
                )
                for bname, bdict in raw_bands.items()
                if isinstance(bdict, dict)
            }
            if isinstance(raw_bands, dict)
            else {}
        )
        return cls(
            qpb_image_dir=qdir,
            manifest_path=manifest_path,
            qpb_files=resolve_manifest_paths(qdir, raw, "qpb_files"),
            band_flux_images=resolve_manifest_paths(qdir, raw, "band_flux_images"),
            bands=bands,
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


def _run_eimagecombine_isolated(
    runner: SASRunner,
    band_dir: Path,
    out_prefix: str,
    comb_params: dict[str, Any],
    combine_log: str,
    *,
    force: bool,
    dry_run: bool,
    docker_container: str | None,
) -> None:
    """Run ``eimagecombine`` in a unique local scratch dir to avoid VirtioFS races."""
    if dry_run or docker_container is not None:
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
        return

    band_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xmm_eimagecombine_") as tmp_dir:
        scratch_dir = Path(tmp_dir)
        copied_inputs: set[str] = set()
        for src in band_dir.iterdir():
            if not src.is_file() or src.name.startswith(out_prefix):
                continue
            if "_ima_" in src.name:
                shutil.copy2(src, scratch_dir / src.name)
                copied_inputs.add(src.name)

        try:
            runner.run(
                SASCommand(
                    program="eimagecombine",
                    cwd=scratch_dir,
                    params=comb_params,
                    core_outputs=[],
                    intermediate_dir=None,
                    log_file=combine_log,
                ),
                force=force,
            )
        finally:
            log_src = scratch_dir / combine_log
            if log_src.is_file():
                shutil.copy2(log_src, band_dir / combine_log)
            for out_file in scratch_dir.iterdir():
                if not out_file.is_file() or out_file.name == combine_log:
                    continue
                if (
                    out_file.name.startswith(out_prefix)
                    or out_file.name not in copied_inputs
                ):
                    shutil.copy2(out_file, band_dir / out_file.name)


def _validate_eimagecombine_outputs(
    band_dir: Path,
    band_name: str,
    out_prefix: str,
    combine_log: str,
    expected_subbands: int,
) -> tuple[list[tuple[Path, Path]], Path | None]:
    """Verify ``eimagecombine`` log and sub-band pair completeness."""
    log_path = band_dir / combine_log
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for marker in ("ERROR:", "skipping file", "Error Status Returned"):
            if marker in log_text:
                tail = read_log_tail(log_path)
                raise SASCommandError(
                    f"eimagecombine reported an error ({marker!r}) for {band_name}.\n"
                    f"Log: {log_path}\n{tail}"
                )

    pairs, mask_path = _discover_combined_subband_pairs(band_dir, out_prefix)
    if not pairs:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine finished for {band_name} but no "
            f"'{out_prefix}ima_<i>_sub.fits' + 'exp.fits' pairs were "
            f"found in {band_dir}.\nLog: {log_path}\n{tail}"
        )
    if len(pairs) != expected_subbands:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine finished for {band_name} with incomplete sub-bands: "
            f"expected {expected_subbands}, found {len(pairs)}.\n"
            f"Log: {log_path}\n{tail}"
        )
    missing_smooth = [
        f"{out_prefix}ima_{idx}_subdiv_smooth.fits"
        for idx in range(expected_subbands)
        if not (band_dir / f"{out_prefix}ima_{idx}_subdiv_smooth.fits").is_file()
    ]
    if missing_smooth:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine finished for {band_name} without smoothed flux outputs: "
            f"{', '.join(missing_smooth)}.\nLog: {log_path}\n{tail}"
        )
    return pairs, mask_path


def _export_band_products(
    qdir: Path,
    band_dir: Path,
    band_name: str,
    band: BandSpec,
    subbands: list[tuple[int, int]],
    pairs: list[tuple[Path, Path]],
    mask_path: Path | None,
    flux_path: Path,
    out_prefix: str,
) -> BandCombinedProducts:
    """Copy combined per-subband outputs into ``qpb_image/<band_name>/``."""
    export_dir = qdir / band_name
    export_dir.mkdir(parents=True, exist_ok=True)

    exported_mask: Path | None = None
    if mask_path is not None and mask_path.is_file():
        exported_mask = export_dir / f"{band_name}_{band.emin}_{band.emax}_mask.fits"
        shutil.copy2(mask_path, exported_mask)

    exported_subbands: list[SubbandCombinedProducts] = []
    for idx, ((lo, hi), (sub_src, exp_src)) in enumerate(zip(subbands, pairs)):
        counts_dst = export_dir / f"{band_name}_{lo}_{hi}_counts.fits"
        exp_dst = export_dir / f"{band_name}_{lo}_{hi}_exp.fits"
        shutil.copy2(sub_src, counts_dst)
        shutil.copy2(exp_src, exp_dst)

        smooth_src = band_dir / f"{out_prefix}ima_{idx}_subdiv_smooth.fits"
        smooth_dst: Path | None = None
        if smooth_src.is_file() and smooth_src.stat().st_size > 0:
            smooth_dst = export_dir / f"{band_name}_{lo}_{hi}_smooth_flux.fits"
            shutil.copy2(smooth_src, smooth_dst)

        exported_subbands.append(
            SubbandCombinedProducts(
                index=idx,
                emin=lo,
                emax=hi,
                counts_image=counts_dst,
                exposure_image=exp_dst,
                smoothed_flux_image=smooth_dst,
            )
        )

    return BandCombinedProducts(
        band_name=band_name,
        emin=band.emin,
        emax=band.emax,
        flux_image=flux_path,
        mask_image=exported_mask,
        subbands=exported_subbands,
    )


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
    bands_products: dict[str, BandCombinedProducts] = {}

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

        n_sub = len(subbands)
        comb_params: dict[str, Any] = {
            "prefix": out_prefix,
            "withcheckinput": True,
            "withaddimages": True,
            "withcombineimages": True,
            "withasmooth": True,
            "keepinterstage": True,
            "ecut": qi.ecut,
            "epn_weight": [qi.epn_weight] * n_sub,
            "em1_weight": [qi.em1_weight] * n_sub,
            "em2_weight": [qi.em2_weight] * n_sub,
            "smoothstyle": qi.smoothstyle,
            "convolverstyle": qi.convolverstyle,
            "width": qi.width,
            "minwidth": qi.minwidth,
            "maxwidth": qi.maxwidth,
            "desiredsnr": qi.desiredsnr,
            "nconvolvers": qi.nconvolvers,
        }
        comb_params.update(qi.eimagecombine_extra_params)

        combine_log = f"{band_name}_combine.log"
        existing_pairs, _ = (
            ([], None)
            if force or dry_run
            else _discover_combined_subband_pairs(band_dir, out_prefix)
        )
        has_all_smooth = (
            not force
            and not dry_run
            and all(
                (band_dir / f"{out_prefix}ima_{idx}_subdiv_smooth.fits").is_file()
                for idx in range(n_sub)
            )
        )
        if len(existing_pairs) < n_sub or not has_all_smooth:
            if not dry_run and band_dir.exists():
                for stale in band_dir.glob(f"{out_prefix}*"):
                    if stale.is_file():
                        stale.unlink()
            _run_eimagecombine_isolated(
                runner,
                band_dir,
                out_prefix,
                comb_params,
                combine_log,
                force=force,
                dry_run=dry_run,
                docker_container=docker_container,
            )

        if not dry_run:
            from xmm_tools.sas.qpb_products import BandImages

            pairs, mask_path = _validate_eimagecombine_outputs(
                band_dir,
                band_name,
                out_prefix,
                combine_log,
                n_sub,
            )
            band_prod = _export_band_products(
                qdir,
                band_dir,
                band_name,
                band,
                subbands,
                pairs,
                mask_path,
                flux_path,
                out_prefix,
            )
            BandImages.from_band_products(band_prod).sum_smoothed_flux(
                apply_mask=True
            ).writeto(flux_path)
            bands_products[band_name] = band_prod

        band_flux_images[band_name] = flux_path

    result = QPBImageResult(
        qpb_image_dir=qdir,
        manifest_path=manifest_path,
        qpb_files=qpb_files,
        band_flux_images=band_flux_images,
        subband_count_images=subband_count_images,
        bands=bands_products,
    )
    if not dry_run:
        result.save_manifest()
    return result
