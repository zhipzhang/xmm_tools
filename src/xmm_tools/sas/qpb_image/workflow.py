"""Workflow orchestration for the ``qpb-image`` SAS stage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig, BandSpec
from xmm_tools.exec import (
    SASCommandError,
    SASRunner,
    StageWorkspace,
    StepLedger,
    is_nonempty_file,
    normalize_lexical_path,
    read_log_tail,
)
from xmm_tools.sas.qpb_image.combine import (
    IsolatedCombineRunner,
    validate_eimagecombine_outputs,
)
from xmm_tools.sas.qpb_image.commands import (
    build_eimagecombine_command,
    build_eimageget_command,
    build_evqpb_command,
    resolve_subbands,
    validate_qpb_extra_params,
)
from xmm_tools.sas.qpb_image.export import (
    build_band_product_paths,
    export_band_products,
)
from xmm_tools.sas.qpb_image.inputs import (
    collect_step_inputs,
    env_file_inputs,
    resolve_attfile,
    resolve_gti_files,
    resolve_qpb_dir,
    resolve_raw_events,
)
from xmm_tools.sas.qpb_image.models import BandProductPaths, QPBImageResult


def _collect_eimageget_outputs(
    eimg_dir: Path, *, instrument: str, band_name: str
) -> dict[str, Path]:
    found = {
        p.name: p
        for p in sorted(eimg_dir.glob("*_ima_*.fits"))
        if is_nonempty_file(p)
    }
    if not found:
        log_p = eimg_dir / f"{instrument}_eimageget.log"
        raise SASCommandError(
            f"eimageget finished for {instrument} ({band_name}) but no "
            f"'*_ima_*.fits' outputs were found in {eimg_dir}.\n"
            f"Log: {log_p}\n{read_log_tail(log_p)}"
        )
    return found


def run_qpb_image(
    cfg: AnaConfig,
    *,
    gti_files: Mapping[str, Path] | None = None,
    flare: Any | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> QPBImageResult:
    """Generate QPB-subtracted, exposure-corrected flux images via ``evqpb`` + ``eimageget`` + ``eimagecombine``."""
    qi = cfg.qpb_image
    validate_qpb_extra_params(qi)

    qdir = resolve_qpb_dir(cfg)
    ws = StageWorkspace(workspace_root=normalize_lexical_path(cfg.paths.workdir), stage_dir=qdir)
    ws.ensure_dirs(dry_run=dry_run)

    raw_events, pn_oot_path = resolve_raw_events(cfg, dry_run=dry_run)
    active_insts = [k for k in ("mos1", "mos2", "pn") if k in raw_events]
    resolved_gtis = resolve_gti_files(
        cfg, active_insts, gti_files=gti_files, flare=flare, dry_run=dry_run
    )
    env_inputs = env_file_inputs(cfg)

    stage_cfg = {
        "bin_size": qi.bin_size,
        "ecut": qi.ecut,
        "bands": {
            bname: {
                "emin": b.emin,
                "emax": b.emax,
                "subbands": resolve_subbands(b, qi.default_subband_edges),
            }
            for bname, b in qi.bands.items()
        },
    }
    ledger = StepLedger(
        ws.manifest_path, stage="qpb_image", workspace_root=ws.workspace_root, config=stage_cfg, dry_run=dry_run, force=force
    )
    runner = SASRunner(
        env=cfg.env, dry_run=dry_run, skip_if_exists=False, docker_container=docker_container
    )
    attfile = resolve_attfile(cfg, ws, ledger, runner, env_inputs, dry_run=dry_run)

    # 1. Shared QPB/FWC event lists once per active instrument
    qpb_files: dict[str, Path] = {}
    for inst in active_insts:
        evqpb_dir = ws.step_dir("evqpb", inst)
        evqpb_cmd = build_evqpb_command(
            instrument=inst, raw_event=raw_events[inst], attfile=attfile, step_dir=evqpb_dir, config=qi
        )
        evqpb_inputs = collect_step_inputs(
            env_inputs, {"raw_event": raw_events[inst], "attfile": attfile}, dry_run=dry_run
        )
        ledger.run_step(f"evqpb:{inst}", runner, evqpb_cmd, inputs=evqpb_inputs)
        qpb_files[inst] = evqpb_dir / f"{inst}_qpb.fits"

    # 2. Per-band eimageget + isolated eimagecombine + derived export/flux sum
    band_flux_images: dict[str, Path] = {}
    subband_count_images: dict[str, list[Path]] = {}
    bands_products: dict[str, BandProductPaths] = {}

    for band_name, band in qi.bands.items():
        subbands = resolve_subbands(band, qi.default_subband_edges)
        band_ima_inputs: dict[str, Path] = {}
        band_outputs: list[Path] = []

        for inst in active_insts:
            eimg_dir = ws.step_dir("bands", band_name, f"eimageget_{inst}")
            oot_evt = pn_oot_path if inst == "pn" else None
            eimg_cmd = build_eimageget_command(
                instrument=inst,
                raw_event=raw_events[inst],
                qpb_event=qpb_files[inst],
                gti_file=resolved_gtis[inst],
                attfile=attfile,
                oot_event=oot_evt,
                subbands=subbands,
                step_dir=eimg_dir,
                config=qi,
            )
            eimg_inputs = collect_step_inputs(
                env_inputs,
                {
                    "raw_event": raw_events[inst],
                    "qpb_event": qpb_files[inst],
                    "gti_file": resolved_gtis[inst],
                    "attfile": attfile,
                    "oot_event": oot_evt,
                },
                dry_run=dry_run,
            )
            step_id = f"eimageget:{band_name}:{inst}"
            if (
                not dry_run
                and ledger.can_reuse(step_id, eimg_cmd, inputs=eimg_inputs) is None
                and eimg_dir.is_dir()
            ):
                for stale in eimg_dir.glob("*_ima_*.fits"):
                    if stale.is_file() or stale.is_symlink():
                        stale.unlink()

            eimg_rec = ledger.run_step(
                step_id,
                runner,
                eimg_cmd,
                inputs=eimg_inputs,
                collect_outputs=lambda d, i=inst, b=band_name: _collect_eimageget_outputs(
                    d, instrument=i, band_name=b
                ),
            )
            for name, ident in eimg_rec.outputs.items():
                band_ima_inputs[name] = ident.path
                band_outputs.append(ident.path)

        subband_count_images[band_name] = band_outputs

        combine_dir = ws.step_dir("bands", band_name, "combine")
        comb_cmd = build_eimagecombine_command(
            band_name=band_name, band=band, subbands=subbands, combine_dir=combine_dir, config=qi
        )
        out_prefix = str(comb_cmd.params["prefix"])
        comb_runner: Any = (
            runner
            if dry_run
            else IsolatedCombineRunner(
                base_runner=runner, combine_dir=combine_dir, out_prefix=out_prefix, input_files=band_ima_inputs
            )
        )
        n_subbands = len(subbands)
        combine_rec = ledger.run_step(
            f"eimagecombine:{band_name}",
            comb_runner,
            comb_cmd,
            inputs=band_ima_inputs,
            collect_outputs=lambda d, b=band_name, p=out_prefix, c=comb_cmd.log_file, n=n_subbands: validate_eimagecombine_outputs(
                d, b, p, c, n
            ),
        )

        flux_path = qdir / f"{band_name}_{band.emin}_{band.emax}_flux.fits"
        combine_outputs = {role: ident.path for role, ident in combine_rec.outputs.items()}

        def _compute_export(
            *,
            bname: str = band_name,
            bspec: BandSpec = band,
            sbs: list[tuple[int, int]] = subbands,
            c_inputs: dict[str, Path] = combine_outputs,
            fpath: Path = flux_path,
        ) -> dict[str, Path]:
            _, exported = export_band_products(qdir, bname, bspec, sbs, c_inputs, fpath)
            return exported

        export_rec = ledger.record_derived_step(
            f"export_band:{band_name}",
            description="qpb_export_and_sum_flux",
            params={"band": band_name, "emin": band.emin, "emax": band.emax, "subbands": [list(sb) for sb in subbands]},
            inputs=combine_outputs,
            compute=_compute_export,
            algorithm_version="1",
            cwd=qdir,
        )
        if not dry_run:
            bands_products[band_name] = build_band_product_paths(
                qdir, band_name, band, subbands, flux_path, has_mask=("mask" in export_rec.outputs)
            )
        band_flux_images[band_name] = flux_path

    products_payload: dict[str, Any] = {
        "qpb_files": qpb_files,
        "band_flux_images": band_flux_images,
        "bands": {bname: bprod.to_dict(qdir) for bname, bprod in bands_products.items()},
    }
    if subband_count_images:
        products_payload["subband_count_images"] = subband_count_images

    manifest = ledger.commit(products=products_payload, prune_unseen_steps=True)
    return QPBImageResult(
        qpb_image_dir=qdir, manifest_path=ws.manifest_path, qpb_files=qpb_files,
        bands=bands_products, band_flux_images=band_flux_images,
        subband_count_images=subband_count_images, manifest=manifest, workspace_root=ws.workspace_root,
    )


def load_qpb_image_result(cfg: AnaConfig) -> QPBImageResult:
    """Load ``qpb_image`` stage outputs from ``manifest.json``."""
    qdir = resolve_qpb_dir(cfg)
    manifest_path = qdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"QPB-image manifest not found at {manifest_path}; run 'xmm-ana qpb-image' first."
        )
    return QPBImageResult.from_manifest(
        manifest_path, workspace_root=normalize_lexical_path(cfg.paths.workdir)
    )
