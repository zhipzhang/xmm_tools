"""Dedicated observation initialization: cifbuild, odfingest, emchain, epchain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from xmm_tools.esas.config import render_template_toml
from xmm_tools.sas.command import SASCommand
from xmm_tools.sas.env import SASEnv
from xmm_tools.sas.runner import SASRunner


STANDARD_EVENT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mos1", "*M1*MIEVLI*.FIT*", "mos1_evt.fits"),
    ("mos2", "*M2*MIEVLI*.FIT*", "mos2_evt.fits"),
    ("pn", "*PN*PIEVLI*.FIT*", "pn_evt.fits"),
    ("pn_oot", "*PN*OOEVLI*.FIT*", "pn_oot_evt.fits"),
)


@dataclass
class InitResult:
    """Artifacts produced by ``initialize_observation``."""

    workdir: Path
    config_path: Path
    ccf_path: Path
    odf_sum_path: Path | None
    event_files: dict[str, Path]


def _find_sum_sas(workdir: Path) -> Path | None:
    matches = sorted(
        workdir.glob("*SUM.SAS"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _pick_primary_event_file(candidates: list[Path]) -> Path:
    """Pick the largest FITS file when multiple exposures match a detector."""
    return max(candidates, key=lambda p: p.stat().st_size)


def standardize_event_files(
    search_dir: Path,
    dest_dir: Path,
    *,
    copy: bool = False,
) -> dict[str, Path]:
    """Locate ``emchain``/``epchain`` outputs and expose standardized names."""
    resolved: dict[str, Path] = {}
    for key, pattern, std_name in STANDARD_EVENT_SPECS:
        matches = sorted(search_dir.glob(pattern))
        if not matches:
            continue
        chosen = _pick_primary_event_file(matches)
        dest = dest_dir / std_name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        if copy:
            shutil.copy2(chosen, dest)
        else:
            dest.symlink_to(chosen)
        resolved[key] = dest
    return resolved


def initialize_observation(
    odf_dir: str | Path,
    workdir: str | Path = ".",
    *,
    sas_ccfpath: str | Path = "/ccf/valid",
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
    copy_events: bool = False,
) -> InitResult:
    """Run ``cifbuild``, ``odfingest``, ``emchain``, ``epchain`` and write ``esas.toml``.

    Intermediate chain products are isolated inside ``workdir/_chain_work`` so
    ``workdir`` stays clean with ``ccf.cif``, ``*SUM.SAS``, ``esas.toml``, and
    the four standardized ``*_evt.fits`` files.
    """
    odf_path = Path(odf_dir).resolve()
    work_path = Path(workdir).resolve()
    chain_dir = work_path / "_chain_work"

    base_env = SASEnv(
        sas_ccfpath=str(sas_ccfpath),
        sas_odf=str(odf_path),
    )

    runner = SASRunner(
        env=base_env,
        dry_run=dry_run,
        skip_if_exists=True,
        docker_container=docker_container,
    )

    # 1. cifbuild
    ccf_cmd = SASCommand(
        program="cifbuild",
        cwd=work_path,
        params={"withccfpath": False, "analysisdate": "now", "category": "XMMCCF"},
        core_outputs=["ccf.cif"],
        intermediate_dir=None,
        log_file="cifbuild.log",
    )
    runner.run(ccf_cmd, force=force)
    ccf_path = work_path / "ccf.cif"

    # 2. odfingest (needs SAS_CCF pointing to ccf.cif)
    ingest_env = SASEnv(
        sas_ccf=ccf_path,
        sas_ccfpath=str(sas_ccfpath),
        sas_odf=str(odf_path),
    )
    ingest_runner = SASRunner(
        env=ingest_env,
        dry_run=dry_run,
        skip_if_exists=True,
        docker_container=docker_container,
    )
    existing_sum = None if force else _find_sum_sas(work_path)
    if existing_sum is None:
        ingest_cmd = SASCommand(
            program="odfingest",
            cwd=work_path,
            params={"odfdir": str(odf_path), "outdir": str(work_path)},
            core_outputs=[],
            intermediate_dir=None,
            log_file="odfingest.log",
        )
        ingest_runner.run(ingest_cmd, force=force)
    odf_sum = _find_sum_sas(work_path)

    # 3. emchain + epchain + epchain (OOT) inside _chain_work
    chain_env = SASEnv(
        sas_ccf=ccf_path,
        sas_ccfpath=str(sas_ccfpath),
        sas_odf=odf_sum or odf_path,
    )
    chain_runner = SASRunner(
        env=chain_env,
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )

    std_events: dict[str, Path] = {}
    if not dry_run:
        std_events = {
            key: work_path / std_name
            for key, _, std_name in STANDARD_EVENT_SPECS
            if (work_path / std_name).exists()
        }

    if force or len(std_events) < len(STANDARD_EVENT_SPECS):
        for prog, extra_params, log_name in (
            ("emchain", {}, "emchain.log"),
            ("epchain", {}, "epchain.log"),
            ("epchain", {"withoutoftime": True}, "epchain_oot.log"),
        ):
            cmd = SASCommand(
                program=prog,
                cwd=chain_dir,
                params=extra_params,
                core_outputs=[],
                intermediate_dir=None,
                log_file=log_name,
            )
            chain_runner.run(cmd, force=force)

        if not dry_run:
            std_events = standardize_event_files(
                chain_dir, work_path, copy=copy_events
            )

    config_path = work_path / "esas.toml"
    if not dry_run:
        work_path.mkdir(parents=True, exist_ok=True)
        odf_name = odf_sum.name if odf_sum is not None else "SUM.SAS"
        config_path.write_text(
            render_template_toml(
                workdir=".",
                sas_ccf="ccf.cif",
                sas_ccfpath=str(sas_ccfpath),
                sas_odf=odf_name,
            ),
            encoding="utf-8",
        )

    return InitResult(
        workdir=work_path,
        config_path=config_path,
        ccf_path=ccf_path,
        odf_sum_path=odf_sum,
        event_files=std_events,
    )
