"""Workflow orchestration for ESAS ``spectrum`` extraction."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig, SpectrumRegionSpec
from xmm_tools.esas.cheese import CheeseResult, load_cheese_result
from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.esas.naming import env_calibration_inputs
from xmm_tools.esas.spectrum.commands import validate_spectrum_extra_params
from xmm_tools.esas.spectrum.detector_coords import ensure_ref_sky_images
from xmm_tools.esas.spectrum.models import RegionSpectrumResult, SpectrumResult
from xmm_tools.esas.spectrum.region_runner import INSTRUMENTS, run_single_region_job
from xmm_tools.esas.spectrum.regions import ExpandedRegionJob, expand_region_specs
from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    CommandStepRecord,
    FileIdentity,
    SASRunner,
    StageManifest,
    StageWorkspace,
    StepLedger,
    atomic_write_json,
    normalize_lexical_path,
    resolve_workspace_path,
    workspace_relative_path,
)
from xmm_tools.exec.manifest import StageStatus, utc_now


def _rebase_step_record(
    rec: CommandStepRecord,
    *,
    step_id: str,
    from_dir: Path,
    to_dir: Path,
    workspace_root: Path,
) -> CommandStepRecord:
    abs_cwd = resolve_workspace_path(from_dir, rec.cwd)
    new_cwd = workspace_relative_path(abs_cwd, to_dir, workspace_root=workspace_root)
    new_log: str | None = None
    if rec.log_file is not None:
        abs_log = resolve_workspace_path(from_dir, rec.log_file)
        new_log = workspace_relative_path(
            abs_log, to_dir, workspace_root=workspace_root
        )
    new_inputs = {
        k: FileIdentity(
            path=resolve_workspace_path(from_dir, ident.path),
            size_bytes=ident.size_bytes,
            sha256=ident.sha256,
        )
        for k, ident in rec.inputs.items()
    }
    new_outputs = {
        k: FileIdentity(
            path=resolve_workspace_path(from_dir, ident.path),
            size_bytes=ident.size_bytes,
            sha256=ident.sha256,
        )
        for k, ident in rec.outputs.items()
    }
    return replace(
        rec,
        step_id=step_id,
        cwd=new_cwd,
        log_file=new_log,
        inputs=new_inputs,
        outputs=new_outputs,
    )


def _merge_region_steps(
    merged_steps: dict[str, CommandStepRecord],
    *,
    region_key: str,
    region_dir: Path,
    reg_manifest: StageManifest,
    spectrum_dir: Path,
    workspace_root: Path,
) -> None:
    prefix = f"region:{region_key}:"
    for old_id in [k for k in merged_steps if k.startswith(prefix)]:
        del merged_steps[old_id]
    for sid, rec in reg_manifest.steps.items():
        full_id = f"{prefix}{sid}"
        merged_steps[full_id] = _rebase_step_record(
            rec,
            step_id=full_id,
            from_dir=region_dir,
            to_dir=spectrum_dir,
            workspace_root=workspace_root,
        )


def _write_top_manifest(
    result: SpectrumResult,
    *,
    status: StageStatus,
    config: Mapping[str, Any],
    steps: Mapping[str, CommandStepRecord],
    error: str | None,
    dry_run: bool,
) -> StageManifest:
    ws_root = result.workspace_root
    products = {
        "regions": {
            key: reg.to_dict(result.spectrum_dir, workspace_root=ws_root)
            for key, reg in sorted(result.regions.items())
        }
    }
    manifest = StageManifest(
        stage="spectrum",
        status=status,
        schema_version=MANIFEST_SCHEMA_VERSION,
        updated_at=utc_now(),
        config=dict(config),
        steps=dict(steps),
        products=products,
        error=error,
    )
    result.manifest = manifest
    if not dry_run:
        atomic_write_json(
            result.manifest_path,
            manifest.to_dict(result.spectrum_dir, workspace_root=ws_root),
        )
    return manifest


def run_spectrum(
    cfg: AnaConfig,
    *,
    regions: list[SpectrumRegionSpec] | None = None,
    only: list[str] | None = None,
    flare: FlareResult | None = None,
    mask_res: CheeseResult | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> SpectrumResult:
    """Run ``mosspectra``/``pnspectra`` + ``mosback``/``pnback`` with ``StepLedger``."""
    spec_cfg = cfg.spectrum
    spectra_extra, back_extra = validate_spectrum_extra_params(spec_cfg)

    target_specs = regions if regions is not None else spec_cfg.regions
    if not target_specs:
        raise ValueError(
            "No spectrum regions configured. Add [[esas.spectrum.regions]] to ana.toml "
            "or pass --reg on the command line."
        )

    sdir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.spectrum_dir))
    ws = StageWorkspace(
        workspace_root=normalize_lexical_path(cfg.paths.workdir),
        stage_dir=sdir,
    )
    ws.ensure_dirs(dry_run=dry_run)

    result = (
        SpectrumResult.from_manifest(
            ws.manifest_path, workspace_root=ws.workspace_root
        )
        if ws.manifest_path.is_file()
        else SpectrumResult(
            spectrum_dir=sdir,
            manifest_path=ws.manifest_path,
            workspace_root=ws.workspace_root,
        )
    )

    expanded_jobs = expand_region_specs(target_specs, cfg, sdir)
    if only:
        only_set = set(only)
        expanded_jobs = [
            j for j in expanded_jobs if j.group in only_set or j.key in only_set
        ]

    flare_res = flare if flare is not None else load_flare_result(cfg)
    if mask_res is None and spec_cfg.withsrcrem:
        mask_res = load_cheese_result(cfg)

    active_insts = [i for i in INSTRUMENTS if i in flare_res.clean_events]
    if not active_insts:
        raise ValueError(
            "No clean event files found for spectrum; run 'xmm-ana flare' first."
        )

    cal_inputs = env_calibration_inputs(cfg, dry_run=dry_run)
    stage_cfg = {
        "elow": spec_cfg.elow,
        "ehigh": spec_cfg.ehigh,
        "withsrcrem": spec_cfg.withsrcrem,
        "withplotfiles": spec_cfg.withplotfiles,
    }
    top_ledger = StepLedger(
        ws.manifest_path,
        stage="spectrum",
        workspace_root=ws.workspace_root,
        config=stage_cfg,
        dry_run=dry_run,
        force=force,
    )
    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )

    ref_images: dict[str, Path] = {}
    if any(j.sky_regions is not None for j in expanded_jobs):
        ref_images = ensure_ref_sky_images(
            active_insts,
            flare_res.clean_events,
            ws,
            top_ledger,
            runner,
            cal_inputs=cal_inputs,
            dry_run=dry_run,
        )

    merged_steps: dict[str, CommandStepRecord] = (
        dict(top_ledger.previous.steps) if top_ledger.previous is not None else {}
    )
    merged_steps.update(top_ledger.steps)

    full_config_run = regions is None and only is None
    active_keys = {j.key for j in expanded_jobs}
    manifest_lock = threading.Lock()
    workers = max(1, min(spec_cfg.max_workers, max(1, len(expanded_jobs))))
    executed_manifests: list[StageManifest] = []
    succeeded_count = 0
    errors: list[tuple[str, str]] = []

    def _execute_and_record(job: ExpandedRegionJob) -> RegionSpectrumResult:
        nonlocal succeeded_count
        reg_res, reg_manifest = run_single_region_job(
            job,
            workspace_root=ws.workspace_root,
            spec_cfg=spec_cfg,
            spectra_extra=spectra_extra,
            back_extra=back_extra,
            clean_events=flare_res.clean_events,
            mask_res=mask_res,
            ref_images=ref_images,
            cal_inputs=cal_inputs,
            runner=runner,
            dry_run=dry_run,
            force=force,
        )
        with manifest_lock:
            result.regions[reg_res.key] = reg_res
            _merge_region_steps(
                merged_steps,
                region_key=job.key,
                region_dir=job.region_dir,
                reg_manifest=reg_manifest,
                spectrum_dir=sdir,
                workspace_root=ws.workspace_root,
            )
            executed_manifests.append(reg_manifest)
            succeeded_count += 1
            if not dry_run:
                _write_top_manifest(
                    result,
                    status="running",
                    config=stage_cfg,
                    steps=merged_steps,
                    error=None,
                    dry_run=False,
                )
        return reg_res

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_execute_and_record, job): job.key for job in expanded_jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append((key, str(exc)))

    if full_config_run:
        for key, reg in result.regions.items():
            reg.active = key in active_keys

    if errors:
        failed_keys = {k for k, _ in errors}
        for key in failed_keys:
            if key in result.regions:
                result.regions[key].active = False
        err_msg = f"Spectrum extraction failed for {errors}"
        if not dry_run:
            fail_status: StageStatus = "partial" if succeeded_count > 0 else "failed"
            _write_top_manifest(
                result,
                status=fail_status,
                config=stage_cfg,
                steps=merged_steps,
                error=err_msg,
                dry_run=False,
            )
        raise RuntimeError(err_msg)

    if dry_run:
        final_status: StageStatus = "dry_run"
    else:
        top_reused = all(rec.status == "reused" for rec in top_ledger.steps.values())
        reg_reused = all(
            rec.status == "reused"
            for m in executed_manifests
            for rec in m.steps.values()
        )
        final_status = "reused" if (top_reused and reg_reused) else "completed"

    _write_top_manifest(
        result,
        status=final_status,
        config=stage_cfg,
        steps=merged_steps,
        error=None,
        dry_run=dry_run,
    )
    return result
