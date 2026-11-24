"""Quick-look QPB-subtracted flux imaging via ``evqpb``, ``eimageget``, ``eimagecombine``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from xmm_tools.esas.config import BandSpec, ESASConfig
from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.esas.source_detect import load_source_mask_result
from xmm_tools.sas.command import SASCommand
from xmm_tools.sas.runner import SASRunner


@dataclass
class QuickImageResult:
    """Core artifacts produced by ``run_quick_image``."""

    quick_image_dir: Path
    manifest_path: Path
    qpb_files: dict[str, Path] = field(default_factory=dict)
    band_flux_images: dict[str, Path] = field(default_factory=dict)
    subband_count_images: dict[str, list[Path]] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "stage": "quick_image",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "qpb_files": {
                k: str(p.relative_to(self.quick_image_dir))
                if p.is_relative_to(self.quick_image_dir)
                else str(p)
                for k, p in self.qpb_files.items()
            },
            "band_flux_images": {
                k: p.name for k, p in self.band_flux_images.items()
            },
        }

    def save_manifest(self) -> Path:
        self.quick_image_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.to_manifest_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> QuickImageResult:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        qdir = manifest_path.parent
        qpb = {k: qdir / v for k, v in raw.get("qpb_files", {}).items()}
        flux = {k: qdir / v for k, v in raw.get("band_flux_images", {}).items()}
        return cls(
            quick_image_dir=qdir,
            manifest_path=manifest_path,
            qpb_files=qpb,
            band_flux_images=flux,
        )


def resolve_subbands(band: BandSpec, default_edges: list[int]) -> list[tuple[int, int]]:
    """Return narrow sub-bands for ``band``, splitting along ``default_edges`` if needed."""
    if band.subbands:
        return [(int(lo), int(hi)) for lo, hi in band.subbands]

    inner = sorted(e for e in default_edges if band.emin < e < band.emax)
    pts = [band.emin, *inner, band.emax]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _resolve_attfile(
    cfg: ESASConfig,
    work_dir: Path,
    runner: SASRunner,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    """Locate an existing attitude FITS file or generate one via ``atthkgen``."""
    if cfg.quick_image.attfile:
        return cfg.paths.resolve(cfg.quick_image.attfile)

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


def _all_nonempty(paths: dict[str, Path]) -> bool:
    return bool(paths) and all(
        p.exists() and p.stat().st_size > 0 for p in paths.values()
    )


def run_quick_image(
    cfg: ESASConfig,
    *,
    flare: FlareResult | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> QuickImageResult:
    """Generate QPB-subtracted, exposure-corrected flux images via ``evqpb`` + ``eimageget``."""
    qi = cfg.quick_image
    qdir = cfg.paths.resolve(qi.dir)
    work_dir = qdir / "_work"
    manifest_path = qdir / "manifest.json"

    if not force and manifest_path.exists():
        existing = QuickImageResult.from_manifest(manifest_path)
        if _all_nonempty(existing.band_flux_images):
            return existing

    flare_res = flare if flare is not None else load_flare_result(cfg)
    mask_res = load_source_mask_result(cfg) if qi.use_cheese_mask else None

    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=not force,
        docker_container=docker_container,
    )

    attfile = _resolve_attfile(cfg, work_dir, runner, dry_run=dry_run, force=force)

    # 1. Generate QPB event lists once per active instrument
    qpb_files: dict[str, Path] = {}
    active_insts = [k for k in ("mos1", "mos2", "pn") if k in flare_res.clean_events]
    for inst in active_insts:
        evt_path = flare_res.clean_events[inst]
        qpb_name = f"{inst}_qpb.fits"
        qpb_params: dict[str, Any] = {
            "table": evt_path,
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

    # 2. Run eimageget per narrow sub-band and combine into target band flux images
    band_flux_images: dict[str, Path] = {}
    subband_count_images: dict[str, list[Path]] = {}

    for band_name, band in qi.bands.items():
        subbands = resolve_subbands(band, qi.default_subband_edges)
        sub_count_imgs: list[Path] = []
        sub_exp_imgs: list[Path] = []
        sub_bkg_imgs: list[Path] = []

        for lo, hi in subbands:
            sub_dir = work_dir / f"{band_name}_{lo}_{hi}"
            for inst in active_insts:
                evt_path = flare_res.clean_events[inst]
                qpb_path = qpb_files[inst]
                prefix = f"{inst}_{lo}_{hi}"
                cnt_name = f"{prefix}_image.fits"
                exp_name = f"{prefix}_exp.fits"
                bkg_name = f"{prefix}_bkg.fits"

                eimg_params: dict[str, Any] = {
                    "evtfile": evt_path,
                    "qpbfile": qpb_path,
                    "attfile": attfile,
                    "pimin": lo,
                    "pimax": hi,
                    "ximagebinsize": qi.bin_size,
                    "yimagebinsize": qi.bin_size,
                    "withvignetting": qi.with_vignetting,
                    "withexposure": True,
                    "outimage": cnt_name,
                    "outexp": exp_name,
                    "outqpb": bkg_name,
                }
                if inst == "pn" and "pn_oot" in flare_res.clean_events:
                    eimg_params["ootfile"] = flare_res.clean_events["pn_oot"]
                if mask_res is not None and inst in mask_res.cheese_masks:
                    eimg_params["withmask"] = True
                    eimg_params["maskfile"] = mask_res.cheese_masks[inst]
                eimg_params.update(qi.eimageget_extra_params)

                runner.run(
                    SASCommand(
                        program="eimageget",
                        cwd=sub_dir,
                        params=eimg_params,
                        core_outputs=[cnt_name, exp_name, bkg_name],
                        intermediate_dir=None,
                        log_file=f"{inst}_eimageget.log",
                    ),
                    force=force,
                )
                sub_count_imgs.append(sub_dir / cnt_name)
                sub_exp_imgs.append(sub_dir / exp_name)
                sub_bkg_imgs.append(sub_dir / bkg_name)

        subband_count_images[band_name] = sub_count_imgs

        # Combine sub-bands (and instruments) into final target band flux image
        out_prefix = f"{band_name}_{band.emin}_{band.emax}"
        flux_name = f"{out_prefix}_flux.fits"
        comb_cnt = f"{out_prefix}_counts.fits"
        comb_exp = f"{out_prefix}_exp.fits"
        comb_bkg = f"{out_prefix}_bkg.fits"

        comb_params: dict[str, Any] = {
            "images": sub_count_imgs,
            "exposures": sub_exp_imgs,
            "backgrounds": sub_bkg_imgs,
            "outimage": comb_cnt,
            "outexposure": comb_exp,
            "outbackground": comb_bkg,
            "outflux": flux_name,
        }
        comb_params.update(qi.eimagecombine_extra_params)

        runner.run(
            SASCommand(
                program="eimagecombine",
                cwd=qdir,
                params=comb_params,
                core_outputs=[flux_name],
                intermediate_dir="_intermediate",
                log_file=f"{out_prefix}_combine.log",
            ),
            force=force,
        )
        band_flux_images[band_name] = qdir / flux_name

    result = QuickImageResult(
        quick_image_dir=qdir,
        manifest_path=manifest_path,
        qpb_files=qpb_files,
        band_flux_images=band_flux_images,
        subband_count_images=subband_count_images,
    )
    if not dry_run:
        result.save_manifest()
    return result
