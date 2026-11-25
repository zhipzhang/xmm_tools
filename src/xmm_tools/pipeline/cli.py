"""Command-line interface for ``xmm-ana``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from xmm_tools.esas.cheese import run_cheese
from xmm_tools.esas.flare import run_flare
from xmm_tools.esas.spectrum import run_spectrum
from xmm_tools.pipeline.config import AnaConfig, SpectrumRegionSpec
from xmm_tools.pipeline.engine import PIPELINE_STAGES, run_pipeline
from xmm_tools.sas.odf import initialize_observation
from xmm_tools.sas.qpb_image import run_qpb_image


def build_parser() -> argparse.ArgumentParser:
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
        "--docker",
        default=None,
        help="Optional Docker container name to run SAS commands inside.",
    )
    p_init.add_argument(
        "--copy-events",
        action="store_true",
        help="Copy standardized *_evt.fits instead of symlinking from _chain_work.",
    )
    p_init.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without running SAS commands or writing files.",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps and overwrite ana.toml even if outputs already exist.",
    )

    p_flare = sub.add_parser(
        "flare",
        help="[ESAS] Run espfilt (and ESAS FLAG evselect) from ana.toml.",
    )
    p_flare.add_argument("--config", default="ana.toml", help="Path to ana.toml (default: ./ana.toml).")
    p_flare.add_argument("--docker", default=None, help="Optional Docker container name.")
    p_flare.add_argument("--dry-run", action="store_true", help="Validate config without running SAS.")
    p_flare.add_argument("--force", action="store_true", help="Re-run even if manifest exists.")

    p_cheese = sub.add_parser(
        "cheese",
        help="[ESAS] Run cheese point-source detection & swiss-cheese masking.",
    )
    p_cheese.add_argument("--config", default="ana.toml", help="Path to ana.toml (default: ./ana.toml).")
    p_cheese.add_argument("--docker", default=None, help="Optional Docker container name.")
    p_cheese.add_argument("--dry-run", action="store_true", help="Validate config without running SAS.")
    p_cheese.add_argument("--force", action="store_true", help="Re-run even if manifest exists.")

    p_qi = sub.add_parser(
        "qpb-image",
        help="[SAS] Generate QPB-subtracted flux images via evqpb + eimageget + eimagecombine.",
    )
    p_qi.add_argument("--config", default="ana.toml", help="Path to ana.toml (default: ./ana.toml).")
    p_qi.add_argument("--docker", default=None, help="Optional Docker container name.")
    p_qi.add_argument("--dry-run", action="store_true", help="Validate config without running SAS.")
    p_qi.add_argument("--force", action="store_true", help="Re-run even if manifest exists.")

    p_spec = sub.add_parser(
        "spectrum",
        help="[ESAS] Extract spectra & QPB backgrounds (mosspectra/pnspectra + mosback/pnback).",
    )
    p_spec.add_argument("--config", default="ana.toml", help="Path to ana.toml (default: ./ana.toml).")
    p_spec.add_argument(
        "--reg",
        default=None,
        help="Ad-hoc DS9 .reg (or DET .txt) file to run instead of [esas.spectrum.regions].",
    )
    p_spec.add_argument("--name", default=None, help="Region group name for --reg (defaults to file stem).")
    p_spec.add_argument(
        "--mode",
        choices=("single", "split", "union"),
        default="single",
        help="Handling mode for --reg (default: single).",
    )
    p_spec.add_argument("--only", action="append", default=None, help="Run only matching region group/key.")
    p_spec.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest entries.")
    p_spec.add_argument("--docker", default=None, help="Optional Docker container name.")
    p_spec.add_argument("--dry-run", action="store_true", help="Validate config & region expansion.")
    p_spec.add_argument("--force", action="store_true", help="Force re-running all selected regions.")

    p_run = sub.add_parser(
        "run",
        help="Run all or selected config-driven pipeline stages in dependency order.",
    )
    p_run.add_argument("--config", default="ana.toml", help="Path to ana.toml (default: ./ana.toml).")
    p_run.add_argument(
        "--stages",
        action="append",
        default=None,
        help=(
            "Comma-separated or repeatable list of stages to run "
            f"({', '.join(PIPELINE_STAGES)}). Mutually exclusive with --from/--to."
        ),
    )
    p_run.add_argument("--from", dest="from_stage", default=None, help="First stage in contiguous range.")
    p_run.add_argument("--to", dest="to_stage", default=None, help="Last stage (inclusive) in range.")
    p_run.add_argument("--skip", action="append", default=None, help="Stages to skip.")
    p_run.add_argument("--only", action="append", default=None, help="Forward --only filter to 'spectrum'.")
    p_run.add_argument("--docker", default=None, help="Optional Docker container name.")
    p_run.add_argument("--dry-run", action="store_true", help="Validate config & stage plan.")
    p_run.add_argument("--force", action="store_true", help="Force re-running all selected stages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "init":
        res = initialize_observation(
            odf_dir=args.odf_dir,
            workdir=args.workdir,
            sas_ccfpath=args.ccfpath,
            dry_run=args.dry_run,
            force=args.force,
            docker_container=args.docker,
            copy_events=args.copy_events,
        )
        print(f"Workdir: {res.workdir}")
        print(f"Config:  {res.config_path}")
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
        print(f"Manifest:  {res.manifest_path} (flag_filtered={res.flag_filtered})")
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
        print(f"Manifest:   {res.manifest_path} (method={res.method})")
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
        print(f"Manifest:      {res.manifest_path}")
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
        print(f"Manifest:     {res.manifest_path} ({len(res.regions)} regions)")
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
        print(f"Stages:    {', '.join(res.stages_run) if res.stages_run else '(none)'}")
        for stage, reason in res.stages_skipped.items():
            print(f"  skipped[{stage}]: {reason}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
