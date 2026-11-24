"""Command-line interface for ``xmm-esas``."""

from __future__ import annotations

import argparse
import sys

from xmm_tools.esas.config import ESASConfig
from xmm_tools.esas.flare import run_flare
from xmm_tools.esas.init import initialize_observation
from xmm_tools.esas.source_detect import run_source_detect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmm-esas",
        description="XMM-Newton ESAS extended-source analysis helper CLI.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_init = sub.add_parser(
        "init",
        help="Run cifbuild, odfingest, emchain, epchain and generate esas.toml.",
    )
    p_init.add_argument(
        "--odf-dir",
        required=True,
        help="Path to the raw XMM-Newton ODF directory.",
    )
    p_init.add_argument(
        "--workdir",
        default=".",
        help="Target ESAS analysis working directory (default: current directory).",
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
        help="Re-run steps even if outputs already exist.",
    )

    p_flare = sub.add_parser(
        "flare",
        help="Run espfilt (and ESAS FLAG evselect) from esas.toml.",
    )
    p_flare.add_argument(
        "--config",
        default="esas.toml",
        help="Path to esas.toml (default: ./esas.toml).",
    )
    p_flare.add_argument(
        "--docker",
        default=None,
        help="Optional Docker container name to run SAS commands inside.",
    )
    p_flare.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running SAS commands.",
    )
    p_flare.add_argument(
        "--force",
        action="store_true",
        help="Re-run espfilt and FLAG filtering even if manifest exists.",
    )

    p_cheese = sub.add_parser(
        "cheese",
        help="Run point-source detection & swiss-cheese masking from esas.toml.",
    )
    p_cheese.add_argument(
        "--config",
        default="esas.toml",
        help="Path to esas.toml (default: ./esas.toml).",
    )
    p_cheese.add_argument(
        "--docker",
        default=None,
        help="Optional Docker container name to run SAS commands inside.",
    )
    p_cheese.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running SAS commands.",
    )
    p_cheese.add_argument(
        "--force",
        action="store_true",
        help="Re-run source detection even if manifest exists.",
    )
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
        cfg = ESASConfig.from_toml(args.config)
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
        cfg = ESASConfig.from_toml(args.config)
        res = run_source_detect(
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
