"""Command-line interface for ``xmm-ana``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xmm_tools.config import AnaConfig, SpectrumRegionSpec
from xmm_tools.esas.cheese import run_cheese
from xmm_tools.esas.flare import run_flare
from xmm_tools.esas.spectrum import run_spectrum
from xmm_tools.exec import StageManifest
from xmm_tools.pipeline.engine import PIPELINE_STAGES, run_pipeline
from xmm_tools.sas.odf import STANDARD_EVENT_SPECS, initialize_observation
from xmm_tools.sas.qpb_image import run_qpb_image

_VALID_EXPOSURE_KEYS = frozenset(key for key, _, _ in STANDARD_EVENT_SPECS)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="ana.toml",
        help="Path to ana.toml (default: ./ana.toml).",
    )


def _add_common_exec_args(
    parser: argparse.ArgumentParser,
    *,
    dry_run_help: str,
    force_help: str,
) -> None:
    parser.add_argument(
        "--docker",
        default=None,
        help="Optional Docker container name to run SAS commands inside.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=dry_run_help,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=force_help,
    )


def _parse_select_args(
    raw_selects: list[str] | None,
    parser: argparse.ArgumentParser,
) -> dict[str, str] | None:
    if not raw_selects:
        return None
    selections: dict[str, str] = {}
    for entry in raw_selects:
        if "=" not in entry:
            parser.error(
                f"Invalid --select entry {entry!r}: expected INSTRUMENT=FILENAME."
            )
        inst_raw, filename_raw = entry.split("=", 1)
        inst = inst_raw.strip().lower()
        filename = filename_raw.strip()
        if inst not in _VALID_EXPOSURE_KEYS:
            valid = ", ".join(sorted(_VALID_EXPOSURE_KEYS))
            parser.error(
                f"Unknown instrument {inst_raw!r} in --select: expected one of {valid}."
            )
        if not filename:
            parser.error(
                f"Empty filename in --select entry {entry!r}: expected INSTRUMENT=FILENAME."
            )
        if inst in selections:
            parser.error(f"Duplicate --select entry for instrument {inst!r}.")
        selections[inst] = filename
    return selections


def _status_tag(manifest: StageManifest | None) -> str:
    if manifest is None:
        return ""
    return f" [status={manifest.status}]"


def build_parser() -> argparse.ArgumentParser:
    """Build the ``xmm-ana`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="xmm-ana",
        description="XMM-Newton SAS/ESAS analysis pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_init = sub.add_parser(
        "init",
        help="[SAS] Run cifbuild, odfingest, emchain, epchain and generate ana.toml.",
    )
    p_init.add_argument(
        "--odf-dir",
        required=True,
        help="Path to the raw XMM-Newton ODF directory.",
    )
    p_init.add_argument(
        "--workdir",
        default=".",
        help="Target analysis working directory (default: current directory).",
    )
    p_init.add_argument(
        "--ccfpath",
        default="/ccf/valid",
        help="SAS_CCFPATH calibration directory (default: /ccf/valid).",
    )
    p_init.add_argument(
        "--copy-events",
        action="store_true",
        help="Copy standardized *_evt.fits instead of symlinking from _chain_work.",
    )
    p_init.add_argument(
        "--select",
        action="append",
        default=None,
        metavar="INSTRUMENT=FILENAME",
        help=(
            "Explicitly select a chain event file for an instrument "
            f"(repeatable; keys: {', '.join(sorted(_VALID_EXPOSURE_KEYS))})."
        ),
    )
    _add_common_exec_args(
        p_init,
        dry_run_help="Validate arguments without running SAS commands or writing files.",
        force_help="Re-run steps and overwrite ana.toml even if outputs already exist.",
    )

    p_flare = sub.add_parser(
        "flare",
        help="[ESAS] Run espfilt (and ESAS FLAG evselect) from ana.toml.",
    )
    _add_config_arg(p_flare)
    _add_common_exec_args(
        p_flare,
        dry_run_help="Validate config without running SAS.",
        force_help="Re-run steps even if outputs and manifest already match.",
    )

    p_cheese = sub.add_parser(
        "cheese",
        help="[ESAS] Run cheese point-source detection & swiss-cheese masking.",
    )
    _add_config_arg(p_cheese)
    _add_common_exec_args(
        p_cheese,
        dry_run_help="Validate config without running SAS.",
        force_help="Re-run steps even if outputs and manifest already match.",
    )

    p_qi = sub.add_parser(
        "qpb-image",
        help="[SAS] Generate QPB-subtracted flux images via evqpb + eimageget + eimagecombine.",
    )
    _add_config_arg(p_qi)
    _add_common_exec_args(
        p_qi,
        dry_run_help="Validate config without running SAS.",
        force_help="Re-run steps even if outputs and manifest already match.",
    )

    p_spec = sub.add_parser(
        "spectrum",
        help="[ESAS] Extract spectra & QPB backgrounds (mosspectra/pnspectra + mosback/pnback).",
    )
    _add_config_arg(p_spec)
    p_spec.add_argument(
        "--reg",
        default=None,
        help="Ad-hoc DS9 .reg (or DET .txt) file to run instead of [esas.spectrum.regions].",
    )
    p_spec.add_argument(
        "--name",
        default=None,
        help="Region group name for --reg (defaults to file stem).",
    )
    p_spec.add_argument(
        "--mode",
        choices=("single", "split", "union"),
        default="single",
        help="Handling mode for --reg (default: single).",
    )
    p_spec.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only matching region group/key.",
    )
    p_spec.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest entries.",
    )
    _add_common_exec_args(
        p_spec,
        dry_run_help="Validate config & region expansion.",
        force_help="Force re-running all selected regions.",
    )

    p_run = sub.add_parser(
        "run",
        help="Run all or selected config-driven pipeline stages in dependency order.",
    )
    _add_config_arg(p_run)
    p_run.add_argument(
        "--stages",
        action="append",
        default=None,
        help=(
            "Comma-separated or repeatable list of stages to run "
            f"({', '.join(PIPELINE_STAGES)}). Mutually exclusive with --from/--to."
        ),
    )
    p_run.add_argument(
        "--from",
        dest="from_stage",
        default=None,
        help="First stage in contiguous range.",
    )
    p_run.add_argument(
        "--to",
        dest="to_stage",
        default=None,
        help="Last stage (inclusive) in range.",
    )
    p_run.add_argument(
        "--skip",
        action="append",
        default=None,
        help="Stages to skip.",
    )
    p_run.add_argument(
        "--only",
        action="append",
        default=None,
        help="Forward --only filter to 'spectrum'.",
    )
    _add_common_exec_args(
        p_run,
        dry_run_help="Validate config & stage plan.",
        force_help="Force re-running all selected stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``xmm-ana`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "init":
        exposure_selection = _parse_select_args(args.select, parser)
        res = initialize_observation(
            odf_dir=args.odf_dir,
            workdir=args.workdir,
            sas_ccfpath=args.ccfpath,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
            copy_events=args.copy_events,
            exposure_selection=exposure_selection,
        )
        print(f"Workdir:  {res.workdir}")
        print(f"Config:   {res.config_path}")
        print(f"Manifest: {res.manifest_path}{_status_tag(res.manifest)}")
        for key, path in res.event_files.items():
            print(f"  {key}: {path.name}")
        return 0

    if args.subcommand == "flare":
        cfg = AnaConfig.from_toml(args.config)
        res = run_flare(
            cfg,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
        )
        print(f"Flare dir: {res.flare_dir}")
        print(
            f"Manifest:  {res.manifest_path} "
            f"(flag_filtered={res.flag_filtered}){_status_tag(res.manifest)}"
        )
        for key, path in res.qdp_files.items():
            print(f"  qdp[{key}]: {path}")
        return 0

    if args.subcommand == "cheese":
        cfg = AnaConfig.from_toml(args.config)
        res = run_cheese(
            cfg,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
        )
        print(f"Cheese dir: {res.cheese_dir}")
        print(
            f"Manifest:   {res.manifest_path} "
            f"(method={res.method}){_status_tag(res.manifest)}"
        )
        if res.emllist:
            print(f"  emllist: {res.emllist.name}")
        for key, path in res.cheese_masks.items():
            print(f"  mask[{key}]: {path.name}")
        return 0

    if args.subcommand == "qpb-image":
        cfg = AnaConfig.from_toml(args.config)
        res = run_qpb_image(
            cfg,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
        )
        print(f"QPB-image dir: {res.qpb_image_dir}")
        print(f"Manifest:      {res.manifest_path}{_status_tag(res.manifest)}")
        for band, path in res.band_flux_images.items():
            print(f"  flux[{band}]: {path.name}")
        return 0

    if args.subcommand == "spectrum":
        cfg = AnaConfig.from_toml(args.config)
        ad_hoc = None
        if args.reg is not None:
            reg_name = args.name or Path(args.reg).stem
            ad_hoc = [
                SpectrumRegionSpec(
                    name=reg_name,
                    file=args.reg,
                    mode=args.mode,
                    overwrite=args.overwrite,
                )
            ]
        res = run_spectrum(
            cfg,
            regions=ad_hoc,
            only=args.only,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
        )
        print(f"Spectrum dir: {res.spectrum_dir}")
        print(
            f"Manifest:     {res.manifest_path} "
            f"({len(res.regions)} regions){_status_tag(res.manifest)}"
        )
        for key, reg in sorted(res.regions.items()):
            summary = ", ".join(
                f"{inst}:{'skip' if p.skipped else p.event_count}"
                for inst, p in reg.instruments.items()
            )
            print(f"  {key} [{reg.mode}]: {summary}")
        return 0

    if args.subcommand == "run":
        res = run_pipeline(
            config_path=args.config,
            stages=args.stages,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            skip=args.skip,
            only=args.only,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
        )
        print(f"Config:    {res.config_path}")
        if res.stages_run:
            stage_items = [
                f"{s} ({res.stage_statuses[s]})" if s in res.stage_statuses else s
                for s in res.stages_run
            ]
            print(f"Stages:    {', '.join(stage_items)}")
        else:
            print("Stages:    (none)")
        for stage, reason in res.stages_skipped.items():
            print(f"  skipped[{stage}]: {reason}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
