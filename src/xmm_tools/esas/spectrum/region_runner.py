"""Per-region ESAS spectrum extraction worker backed by ``StepLedger``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xmm_tools.config import SpectrumConfig
from xmm_tools.esas.cheese import CheeseResult
from xmm_tools.esas.naming import FOVT_SUFFIX, discover_esas_prefix, strip_esas_suffix
from xmm_tools.esas.spectrum.commands import (
    build_count_events_command,
    build_mosback_command,
    build_mosspectra_command,
    build_pnback_command,
    build_pnspectra_command,
    build_protonscale_command,
    parse_protonscale_log,
    read_filtered_event_count,
)
from xmm_tools.esas.spectrum.detector_coords import DET_SCALE, build_sky_region_expr
from xmm_tools.esas.spectrum.models import (
    InstrumentSpectrumProducts,
    RegionSpectrumResult,
)
from xmm_tools.esas.spectrum.regions import ExpandedRegionJob, _import_regions
from xmm_tools.exec import (
    SASCommandError,
    SASRunner,
    StageManifest,
    StepLedger,
    is_nonempty_file,
    publish_artifact,
    read_log_tail,
)

_QDP_KEYS = ("bkgspec", "augspec", "ratehard")
INSTRUMENTS = ("mos1", "mos2", "pn")


def _resolve_region_masks(
    mask_res: CheeseResult | None,
    inst: str,
    *,
    withsrcrem: bool,
    dry_run: bool = False,
) -> tuple[Path | None, Path | None]:
    if not withsrcrem:
        return None, None
    if mask_res is None:
        raise ValueError(
            "withsrcrem=True requires cheese masks; pass mask_res or run 'xmm-ana cheese' first."
        )
    det = mask_res.region_det.get(inst) or mask_res.cheese_masks.get(f"{inst}_det")
    sky = mask_res.region_sky.get(inst) or mask_res.cheese_masks.get(f"{inst}_sky")
    if det is None or (not dry_run and not is_nonempty_file(det)):
        raise FileNotFoundError(
            f"Missing or empty region_det mask for {inst!r} (withsrcrem=True): {det}"
        )
    if sky is None or (not dry_run and not is_nonempty_file(sky)):
        raise FileNotFoundError(
            f"Missing or empty region_sky mask for {inst!r} (withsrcrem=True): {sky}"
        )
    return det, sky


def _prepare_region_expr(
    job: ExpandedRegionJob,
    inst: str,
    *,
    work_dir: Path,
    ref_images: Mapping[str, Path],
    reg_ledger: StepLedger,
    runner: SASRunner,
    dry_run: bool,
) -> tuple[Path, str]:
    reg_txt = work_dir / f"reg_{inst}.txt"
    if job.det_expr_text is not None:
        if not dry_run:
            reg_txt.parent.mkdir(parents=True, exist_ok=True)
            content = job.det_expr_text + "\n"
            if not reg_txt.is_file() or reg_txt.read_text(encoding="utf-8") != content:
                reg_txt.write_text(content, encoding="utf-8")
        return reg_txt, job.det_expr_text

    ref_img = ref_images[inst]
    conv_inputs: dict[str, Path] = {}
    if not dry_run or is_nonempty_file(job.source_file):
        conv_inputs["source_file"] = job.source_file
    if not dry_run or is_nonempty_file(ref_img):
        conv_inputs["ref_image"] = ref_img

    def _compute_expr() -> dict[str, Path]:
        expr_val = build_sky_region_expr(
            job.sky_regions or [],
            ref_img,
            runner,
            work_dir / "_ecoord",
            inst,
        )
        reg_txt.parent.mkdir(parents=True, exist_ok=True)
        reg_txt.write_text(expr_val + "\n", encoding="utf-8")
        return {"reg_txt": reg_txt}

    reg_ledger.record_derived_step(
        f"convert_region:{inst}",
        description="sky_region_to_esas_det",
        params={
            "key": job.key,
            "mode": job.mode,
            "label": job.label,
            "instrument": inst,
            "det_scale": DET_SCALE,
        },
        inputs=conv_inputs,
        compute=_compute_expr,
        algorithm_version="1",
        cwd=work_dir,
    )
    expr = reg_txt.read_text(encoding="utf-8").strip() if not dry_run else ""
    return reg_txt, expr


def _count_events_step(
    *,
    inst: str,
    event_path: Path,
    reg_txt: Path,
    expr_text: str,
    work_dir: Path,
    reg_ledger: StepLedger,
    runner: SASRunner,
    cal_inputs: Mapping[str, Path],
    dry_run: bool,
) -> int:
    count_file = work_dir / f"_val_{inst}_count.txt"
    cmd = build_count_events_command(event_path, expr_text, work_dir, inst)
    step_inputs = dict(cal_inputs)
    if not dry_run or is_nonempty_file(event_path):
        step_inputs["clean_event"] = event_path
    if not dry_run or is_nonempty_file(reg_txt):
        step_inputs["reg_txt"] = reg_txt

    def _collect_count(d: Path) -> dict[str, Path]:
        filtered = d / f"_val_{inst}.fits"
        n_rows = read_filtered_event_count(filtered)
        if filtered.exists():
            filtered.unlink()
        count_file.write_text(f"{n_rows}\n", encoding="utf-8")
        return {"count_file": count_file}

    reg_ledger.run_step(
        f"count_events:{inst}",
        runner,
        cmd,
        inputs=step_inputs,
        collect_outputs=_collect_count,
    )
    if dry_run:
        return 1
    return int(count_file.read_text(encoding="utf-8").strip())


def _collect_spectra_outputs(work_dir: Path, inst: str) -> dict[str, Path]:
    prefix = discover_esas_prefix(work_dir, inst, FOVT_SUFFIX)
    outputs: dict[str, Path] = {
        "fovt": work_dir / f"{prefix}-fovt.pi",
        "rmf": work_dir / f"{prefix}.rmf",
        "arf": work_dir / f"{prefix}.arf",
    }
    if inst == "pn":
        outputs["fovtoot"] = work_dir / f"{prefix}-fovtoot.pi"
    fovimspdet = work_dir / f"{prefix}-fovimspdet.fits"
    if is_nonempty_file(fovimspdet):
        outputs["fovimspdet"] = fovimspdet
    return outputs


def _collect_back_outputs(work_dir: Path, prefix: str, inst: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {"bkg": work_dir / f"{prefix}-bkg.pi"}
    if inst == "pn":
        outputs["fovtootsub"] = work_dir / f"{prefix}-fovtootsub.pi"
    for qkey in _QDP_KEYS:
        qpath = work_dir / f"{prefix}-{qkey}.qdp"
        if is_nonempty_file(qpath):
            outputs[f"qdp_{qkey}"] = qpath
    return outputs


def _publish_instrument_files(
    *,
    inst: str,
    prefix: str,
    work_dir: Path,
    region_dir: Path,
    reg_txt_work: Path,
    proton_meta: tuple[float | None, float | None, float | None],
    count: int,
) -> InstrumentSpectrumProducts:
    pub_reg_txt = publish_artifact(
        reg_txt_work, region_dir / f"reg_{inst}.txt", mode="symlink"
    )

    def _pub_opt(suffix: str) -> Path | None:
        src = work_dir / f"{prefix}{suffix}"
        dst = region_dir / f"{prefix}{suffix}"
        if not is_nonempty_file(src):
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            return None
        return publish_artifact(src, dst, mode="symlink")

    qdp_plots: dict[str, Path] = {}
    for k in _QDP_KEYS:
        p = _pub_opt(f"-{k}.qdp")
        if p is not None:
            qdp_plots[k] = p
        elif region_dir.is_dir():
            for stale in region_dir.glob(f"{inst}*-{k}.qdp"):
                if stale.is_symlink() or stale.is_file():
                    stale.unlink()
    return InstrumentSpectrumProducts(
        fovt=_pub_opt("-fovt.pi"),
        fovtoot=_pub_opt("-fovtoot.pi") if inst == "pn" else None,
        fovtootsub=_pub_opt("-fovtootsub.pi") if inst == "pn" else None,
        bkg=_pub_opt("-bkg.pi"),
        rmf=_pub_opt(".rmf"),
        arf=_pub_opt(".arf"),
        qdp_plots=qdp_plots,
        region_file=pub_reg_txt,
        event_count=count,
        proton_area=proton_meta[0],
        proton_ave_flux=proton_meta[1],
        proton_scale=proton_meta[2],
        skipped=False,
    )


def _run_instrument_in_region(
    *,
    inst: str,
    job: ExpandedRegionJob,
    work_dir: Path,
    clean_events: Mapping[str, Path],
    mask_res: CheeseResult | None,
    ref_images: Mapping[str, Path],
    spec_cfg: SpectrumConfig,
    spectra_extra: Mapping[str, Any],
    back_extra: Mapping[str, Any],
    cal_inputs: Mapping[str, Path],
    reg_ledger: StepLedger,
    runner: SASRunner,
    dry_run: bool,
) -> tuple[InstrumentSpectrumProducts, int]:
    reg_txt_work, expr_text = _prepare_region_expr(
        job,
        inst,
        work_dir=work_dir,
        ref_images=ref_images,
        reg_ledger=reg_ledger,
        runner=runner,
        dry_run=dry_run,
    )
    event_path = clean_events[inst]
    count = _count_events_step(
        inst=inst,
        event_path=event_path,
        reg_txt=reg_txt_work,
        expr_text=expr_text,
        work_dir=work_dir,
        reg_ledger=reg_ledger,
        runner=runner,
        cal_inputs=cal_inputs,
        dry_run=dry_run,
    )
    if dry_run:
        return (
            InstrumentSpectrumProducts(region_file=job.region_dir / f"reg_{inst}.txt"),
            count,
        )

    if count == 0:
        pub_reg = publish_artifact(
            reg_txt_work, job.region_dir / f"reg_{inst}.txt", mode="symlink"
        )
        return (
            InstrumentSpectrumProducts(
                region_file=pub_reg,
                event_count=0,
                skipped=True,
                skip_reason="zero events in detector region",
            ),
            0,
        )

    mask_det, mask_sky = _resolve_region_masks(
        mask_res, inst, withsrcrem=spec_cfg.withsrcrem, dry_run=dry_run
    )
    spec_inputs: dict[str, Path] = {
        **cal_inputs,
        "clean_event": event_path,
        "reg_txt": reg_txt_work,
    }
    if mask_det is not None and mask_sky is not None:
        spec_inputs["maskdet"] = mask_det
        spec_inputs["masksky"] = mask_sky

    if inst == "pn":
        oot_path = clean_events.get("pn_oot")
        if oot_path is not None:
            spec_inputs["pn_oot"] = oot_path
        spec_cmd = build_pnspectra_command(
            event_file=event_path,
            oot_event_file=oot_path,
            region_file=reg_txt_work,
            work_dir=work_dir,
            spec_cfg=spec_cfg,
            elow=job.elow,
            ehigh=job.ehigh,
            mask_det=mask_det,
            mask_sky=mask_sky,
            extra_params=spectra_extra,
        )
    else:
        spec_cmd = build_mosspectra_command(
            inst=inst,
            event_file=event_path,
            region_file=reg_txt_work,
            work_dir=work_dir,
            spec_cfg=spec_cfg,
            elow=job.elow,
            ehigh=job.ehigh,
            mask_det=mask_det,
            mask_sky=mask_sky,
            extra_params=spectra_extra,
        )

    spec_step_id = f"spectra:{inst}"
    if (
        reg_ledger.can_reuse(spec_step_id, spec_cmd, inputs=spec_inputs) is None
        and work_dir.is_dir()
    ):
        for pat in (
            f"{inst}*-fovt*.pi",
            f"{inst}*-fovimspdet.fits",
            f"{inst}*.rmf",
            f"{inst}*.arf",
        ):
            for stale in work_dir.glob(pat):
                if stale.is_file() or stale.is_symlink():
                    stale.unlink()

    spec_rec = reg_ledger.run_step(
        spec_step_id,
        runner,
        spec_cmd,
        inputs=spec_inputs,
        collect_outputs=lambda d, i=inst: _collect_spectra_outputs(d, i),
    )
    prefix = strip_esas_suffix(spec_rec.outputs["fovt"].path)

    back_inputs: dict[str, Path] = {
        **cal_inputs,
        "fovt": spec_rec.outputs["fovt"].path,
    }
    if inst == "pn":
        back_inputs["fovtoot"] = spec_rec.outputs["fovtoot"].path
        back_cmd = build_pnback_command(
            prefix=prefix,
            work_dir=work_dir,
            spec_cfg=spec_cfg,
            elow=job.elow,
            ehigh=job.ehigh,
            extra_params=back_extra,
        )
    else:
        back_cmd = build_mosback_command(
            inst=inst,
            prefix=prefix,
            work_dir=work_dir,
            spec_cfg=spec_cfg,
            elow=job.elow,
            ehigh=job.ehigh,
            extra_params=back_extra,
        )

    back_step_id = f"back:{inst}"
    if (
        reg_ledger.can_reuse(back_step_id, back_cmd, inputs=back_inputs) is None
        and work_dir.is_dir()
    ):
        for qkey in _QDP_KEYS:
            for stale in work_dir.glob(f"{inst}*-{qkey}.qdp"):
                if stale.is_file() or stale.is_symlink():
                    stale.unlink()

    reg_ledger.run_step(
        back_step_id,
        runner,
        back_cmd,
        inputs=back_inputs,
        collect_outputs=lambda d, p=prefix, i=inst: _collect_back_outputs(d, p, i),
    )

    proton_meta: tuple[float | None, float | None, float | None] = (None, None, None)
    fovimspdet = work_dir / f"{prefix}-fovimspdet.fits"
    fovt_pi = work_dir / f"{prefix}-fovt.pi"
    if not is_nonempty_file(fovimspdet):
        if spec_cfg.keepinterfiles:
            spectra_prog = "pnspectra" if inst == "pn" else "mosspectra"
            raise SASCommandError(
                f"Expected {fovimspdet.name} for protonscale (keepinterfiles=True), "
                f"but it was not produced by {spectra_prog} in {work_dir}"
            )
    elif is_nonempty_file(fovt_pi):
        ps_cmd = build_protonscale_command(work_dir, prefix, inst)
        ps_log = f"protonscale_{inst}.log"
        ps_log_path = work_dir / ps_log
        reg_ledger.run_step(
            f"protonscale:{inst}",
            runner,
            ps_cmd,
            inputs={**cal_inputs, "maskfile": fovimspdet, "specfile": fovt_pi},
            require_outputs=False,
            collect_outputs=lambda d, lf=ps_log: (
                {"log": d / lf} if is_nonempty_file(d / lf) else {}
            ),
        )
        proton_meta = parse_protonscale_log(ps_log_path)
        if any(val is None for val in proton_meta):
            tail = read_log_tail(ps_log_path)
            raise SASCommandError(
                f"Failed to parse protonscale output from {ps_log_path} "
                f"(area={proton_meta[0]}, ave_flux={proton_meta[1]}, "
                f"scaled_ave_flux={proton_meta[2]}). Log tail:\n{tail}"
            )

    return (
        _publish_instrument_files(
            inst=inst,
            prefix=prefix,
            work_dir=work_dir,
            region_dir=job.region_dir,
            reg_txt_work=reg_txt_work,
            proton_meta=proton_meta,
            count=count,
        ),
        count,
    )


def run_single_region_job(
    job: ExpandedRegionJob,
    *,
    workspace_root: Path,
    spec_cfg: SpectrumConfig,
    spectra_extra: Mapping[str, Any],
    back_extra: Mapping[str, Any],
    clean_events: Mapping[str, Path],
    mask_res: CheeseResult | None,
    ref_images: Mapping[str, Path],
    cal_inputs: Mapping[str, Path],
    runner: SASRunner,
    dry_run: bool,
    force: bool,
) -> tuple[RegionSpectrumResult, StageManifest]:
    """Execute or reuse all ESAS spectrum extraction steps for one region."""
    rdir = job.region_dir
    work_dir = rdir / "_work"
    reg_manifest_path = rdir / "region_manifest.json"
    active_insts = [i for i in INSTRUMENTS if i in clean_events]
    if not dry_run:
        rdir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        if job.sky_regions and job.mode == "split":
            Regions, *_ = _import_regions()
            reg_prov = work_dir / "region.reg"
            Regions(job.sky_regions).write(str(reg_prov), format="ds9", overwrite=True)
            publish_artifact(reg_prov, rdir / "region.reg", mode="symlink")

    reg_ledger = StepLedger(
        reg_manifest_path,
        stage="spectrum_region",
        workspace_root=workspace_root,
        config={
            "key": job.key,
            "mode": job.mode,
            "elow": job.elow,
            "ehigh": job.ehigh,
            "withsrcrem": spec_cfg.withsrcrem,
            "withplotfiles": spec_cfg.withplotfiles,
            "pattern_mos": spec_cfg.pattern_mos,
            "pattern_pn": spec_cfg.pattern_pn,
            "mos1_ccds": spec_cfg.mos1_ccds,
            "mos2_ccds": spec_cfg.mos2_ccds,
            "pn_quads": spec_cfg.pn_quads,
        },
        dry_run=dry_run,
        force=force or job.overwrite,
    )

    inst_results: dict[str, InstrumentSpectrumProducts] = {}
    val_lines: list[str] = []
    for inst in active_insts:
        prod, count = _run_instrument_in_region(
            inst=inst,
            job=job,
            work_dir=work_dir,
            clean_events=clean_events,
            mask_res=mask_res,
            ref_images=ref_images,
            spec_cfg=spec_cfg,
            spectra_extra=spectra_extra,
            back_extra=back_extra,
            cal_inputs=cal_inputs,
            reg_ledger=reg_ledger,
            runner=runner,
            dry_run=dry_run,
        )
        inst_results[inst] = prod
        val_lines.append(f"{inst}\t{count}")

    if not dry_run:
        val_work = work_dir / "validation.txt"
        val_work.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
        publish_artifact(val_work, rdir / "validation.txt", mode="symlink")

    reg_res = RegionSpectrumResult(
        key=job.key,
        group=job.group,
        label=job.label,
        mode=job.mode,
        region_dir=rdir,
        source_file=job.source_file,
        elow=job.elow,
        ehigh=job.ehigh,
        active=True,
        instruments=inst_results,
    )
    reg_manifest = reg_ledger.commit(
        products={"region": reg_res.to_dict(rdir, workspace_root=workspace_root)},
        prune_unseen_steps=True,
    )
    return reg_res, reg_manifest
