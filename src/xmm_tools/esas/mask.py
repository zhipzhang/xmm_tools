"""ESAS ``region`` custom point-source mask stage from external/merged source lists."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from xmm_tools.config import AnaConfig, MaskConfig
from xmm_tools.esas.cheese import CheeseResult
from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.esas.naming import (
    env_calibration_inputs,
    strip_esas_suffix,
    validate_scalar_extra_params,
)
from xmm_tools.exec import (
    SASCommand,
    SASCommandError,
    SASRunner,
    StageWorkspace,
    StepLedger,
    compute_sha256,
    is_nonempty_file,
    normalize_lexical_path,
    publish_artifact,
    read_log_tail,
)

_MASK_RESERVED_PARAMS = frozenset(
    {
        "eventset",
        "tempset",
        "operationstyle",
        "srclisttab",
        "regionset",
        "bkgregionset",
        "outunit",
        "radiusstyle",
        "bkgfraction",
        "bkgratestyle",
        "bkgratecol",
        "bkgrateuser",
        "nosrcellipse",
        "energyfraction",
        "fixedradius",
        "expression",
        "withboresightfudge",
    }
)

_INSTRUMENTS: tuple[str, ...] = ("mos1", "mos2", "pn")

_SKY_WCS_KEYS: tuple[tuple[str, str], ...] = (
    ("REFXCRPX", "TCRPX2"),
    ("REFXCRVL", "TCRVL2"),
    ("REFXCDLT", "TCDLT2"),
    ("REFXCTYP", "TCTYP2"),
    ("REFXCUNI", "TCUNI2"),
    ("REFYCRPX", "TCRPX3"),
    ("REFYCRVL", "TCRVL3"),
    ("REFYCDLT", "TCDLT3"),
    ("REFYCTYP", "TCTYP3"),
    ("REFYCUNI", "TCUNI3"),
)


def _resolve_srclist_path(cfg: AnaConfig, srclist: str | Path) -> Path:
    raw = Path(srclist)
    if raw.is_absolute() or raw.is_file():
        return normalize_lexical_path(raw)
    return normalize_lexical_path(cfg.paths.resolve(srclist))


def _is_empty_fits_srclist(srclist_path: Path) -> bool:
    """Return ``True`` only if ``srclist_path`` parses as FITS with a 0-row SRCLIST."""
    try:
        with fits.open(srclist_path) as hdul:
            if "SRCLIST" not in hdul:
                return False
            data = hdul["SRCLIST"].data
            return data is None or len(data) == 0
    except Exception:  # noqa: BLE001
        return False


def _read_sky_wcs_cards(evt_path: Path) -> dict[str, Any]:
    """Extract ``REFX*`` / ``REFY*`` sky WCS cards from ``evt_path`` if present."""
    cards: dict[str, Any] = {}
    try:
        with fits.open(evt_path) as hdul:
            candidate_hdus = []
            if "EVENTS" in hdul:
                candidate_hdus.append(hdul["EVENTS"])
            candidate_hdus.extend(hdu for hdu in hdul if hdu not in candidate_hdus)
            for hdu in candidate_hdus:
                hdr = hdu.header
                for src_key, dst_key in _SKY_WCS_KEYS:
                    if dst_key not in cards and src_key in hdr:
                        cards[dst_key] = hdr[src_key]
    except Exception:  # noqa: BLE001
        return {}
    return cards


def _write_empty_region_table(
    path: Path,
    *,
    coord_cols: tuple[str, str],
    extra_header: dict[str, Any] | None = None,
) -> Path:
    """Write a valid 0-row SAS ``REGION`` ``BinTableHDU`` FITS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x_name, y_name = coord_cols
    cols = [
        fits.Column(name="SHAPE", format="16A", array=np.empty(0, dtype="S16")),
        fits.Column(name=x_name, format="4E", array=np.empty((0, 4), dtype=np.float32)),
        fits.Column(name=y_name, format="4E", array=np.empty((0, 4), dtype=np.float32)),
        fits.Column(name="R", format="4E", array=np.empty((0, 4), dtype=np.float32)),
        fits.Column(name="ROTANG", format="4E", array=np.empty((0, 4), dtype=np.float32)),
        fits.Column(name="COMPONENT", format="J", array=np.empty(0, dtype=np.int32)),
    ]
    tbl = fits.BinTableHDU.from_columns(cols, name="REGION")
    tbl.header["HDUVERS"] = "1.0.0"
    tbl.header["HDUCLASS"] = "ASC"
    tbl.header["HDUCLAS1"] = "REGION"
    tbl.header["HDUCLAS2"] = "STANDARD"
    tbl.header["MTYPE1"] = "pos"
    tbl.header["MFORM1"] = f"{x_name},{y_name}"
    if extra_header:
        for key, val in extra_header.items():
            tbl.header[key] = val
    fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(path, overwrite=True)
    return path


def _is_no_sources_selected_error(exc: SASCommandError) -> bool:
    """Return ``True`` if SAS ``region`` failed with ``noSourcesSelected``."""
    if "noSourcesSelected" in str(exc):
        return True
    if exc.result is not None and exc.result.log_path is not None:
        return "noSourcesSelected" in read_log_tail(exc.result.log_path)
    return False


def _record_empty_region_step(
    ledger: StepLedger,
    *,
    inst: str,
    work_dir: Path,
    det_name: str,
    sky_name: str,
    evt_path: Path,
    step_inputs: dict[str, Path],
    reason: str,
    expression: str = "",
) -> None:
    """Record a derived step writing valid 0-row det and sky ``REGION`` tables."""
    det_work = work_dir / det_name
    sky_work = work_dir / sky_name

    def _write_empty_pair() -> dict[str, Path]:
        _write_empty_region_table(det_work, coord_cols=("DETX", "DETY"))
        sky_cards = _read_sky_wcs_cards(evt_path)
        _write_empty_region_table(
            sky_work, coord_cols=("X", "Y"), extra_header=sky_cards
        )
        return {"det": det_work, "sky": sky_work}

    params: dict[str, Any] = {
        "instrument": inst,
        "empty_srclist": reason == "empty_srclist",
        "empty_reason": reason,
        "det_name": det_name,
        "sky_name": sky_name,
    }
    if expression:
        params["expression"] = expression

    ledger.record_derived_step(
        f"empty_region:{inst}",
        description="write_empty_region_tables",
        params=params,
        inputs=step_inputs,
        compute=_write_empty_pair,
        cwd=work_dir,
    )


def _build_region_params(
    mcfg: MaskConfig,
    *,
    evt_name: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "eventset": evt_name,
        "operationstyle": "global",
        "srclisttab": "srclist.fits:SRCLIST",
        "expression": mcfg.expression,
        "radiusstyle": mcfg.radius_style,
        "withboresightfudge": "yes",
    }
    if mcfg.radius_style == "contour":
        params.update(
            {
                "bkgratestyle": mcfg.bkgrate_style,
                "bkgratecol": "BG_MAP",
                "bkgfraction": mcfg.bkg_fraction,
                "nosrcellipse": "yes" if mcfg.nosrcellipse else "no",
            }
        )
        if mcfg.bkgrate_style == "user":
            params["bkgrateuser"] = mcfg.bkgrate_user
    elif mcfg.radius_style == "enfrac":
        params["energyfraction"] = mcfg.energy_fraction
    elif mcfg.radius_style == "userfixed":
        params["fixedradius"] = mcfg.fixed_radius
    params.update(extra)
    return params


def run_region_mask(
    cfg: AnaConfig,
    *,
    srclist: str | Path,
    out_dir: str | Path,
    mask_cfg: MaskConfig | None = None,
    flare: FlareResult | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> CheeseResult:
    """Generate per-instrument det/sky exclusion masks via SAS ``region``."""
    mcfg = mask_cfg if mask_cfg is not None else cfg.mask
    extra = validate_scalar_extra_params(
        mcfg.extra_params,
        section="mask.extra_params",
        reserved_keys=_MASK_RESERVED_PARAMS,
    )

    srclist_path = _resolve_srclist_path(cfg, srclist)
    if not dry_run and not is_nonempty_file(srclist_path):
        raise FileNotFoundError(
            f"Source list file not found or empty: {srclist_path}"
        )

    target_dir = normalize_lexical_path(cfg.paths.resolve(out_dir))
    orig_cheese_dir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.cheese_dir))
    if (
        target_dir == orig_cheese_dir
        or Path(os.path.realpath(target_dir)) == Path(os.path.realpath(orig_cheese_dir))
    ):
        raise ValueError(
            f"--out-dir ({target_dir}) must differ from [paths].cheese_dir "
            f"({orig_cheese_dir}) to avoid overwriting native cheese outputs."
        )

    flare_res = flare if flare is not None else load_flare_result(cfg)
    active_events: dict[str, Path] = {}
    for inst in _INSTRUMENTS:
        raw_evt = flare_res.clean_events.get(inst)
        if raw_evt is None:
            continue
        evt_path = normalize_lexical_path(raw_evt)
        if not dry_run and not is_nonempty_file(evt_path):
            raise FileNotFoundError(
                f"Clean event file for {inst!r} not found or empty: {evt_path}"
            )
        active_events[inst] = evt_path

    if not active_events:
        raise ValueError(
            "No clean event files found for mask generation; run 'xmm-ana flare' first."
        )

    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )
    if docker_container and not dry_run:
        mounts = runner._resolve_docker_mounts()
        if mounts:
            real_target = Path(os.path.realpath(target_dir))
            if not any(
                real_target.is_relative_to(Path(os.path.realpath(host_dir)))
                for host_dir, _ in mounts
            ):
                raise ValueError(
                    f"--out-dir ({target_dir}) is not mounted inside Docker "
                    f"container {docker_container!r}"
                )

    ws = StageWorkspace(
        workspace_root=normalize_lexical_path(cfg.paths.workdir),
        stage_dir=target_dir,
    )
    ws.ensure_dirs(dry_run=dry_run)

    staged_srclist = ws.step_dir("srclist") / "emllist.fits"
    if not dry_run:
        if not (
            staged_srclist.is_file()
            and staged_srclist.stat().st_size == srclist_path.stat().st_size
            and compute_sha256(staged_srclist) == compute_sha256(srclist_path)
        ):
            publish_artifact(srclist_path, staged_srclist, mode="copy")
        ws.publish(staged_srclist, "emllist.fits", mode="symlink")

    ledger = StepLedger(
        ws.manifest_path,
        stage="mask",
        workspace_root=ws.workspace_root,
        config={
            "method": "region_mask",
            "radius_style": mcfg.radius_style,
            "bkg_fraction": mcfg.bkg_fraction,
            "bkgrate_style": mcfg.bkgrate_style,
            "bkgrate_user": mcfg.bkgrate_user,
            "nosrcellipse": mcfg.nosrcellipse,
            "energy_fraction": mcfg.energy_fraction,
            "fixed_radius": mcfg.fixed_radius,
            "expression": mcfg.expression,
            "extra_params": dict(extra),
        },
        dry_run=dry_run,
        force=force,
    )
    cal_inputs = env_calibration_inputs(cfg, dry_run=dry_run)

    empty_srclist = False if dry_run else _is_empty_fits_srclist(staged_srclist)

    region_det: dict[str, Path] = {}
    region_sky: dict[str, Path] = {}

    for inst, evt_path in active_events.items():
        prefix = strip_esas_suffix(evt_path)
        det_name = f"{prefix}-bkgregtdet.fits"
        sky_name = f"{prefix}-bkgregtsky.fits"
        empty_dir = ws.step_dir("region", inst)
        det_dir = ws.step_dir("region_det", inst)
        sky_dir = ws.step_dir("region_sky", inst)

        step_inputs = dict(cal_inputs)
        if not dry_run or is_nonempty_file(evt_path):
            step_inputs["clean_event"] = evt_path
        if not dry_run or is_nonempty_file(staged_srclist):
            step_inputs["srclist"] = staged_srclist

        if empty_srclist:
            _record_empty_region_step(
                ledger,
                inst=inst,
                work_dir=empty_dir,
                det_name=det_name,
                sky_name=sky_name,
                evt_path=evt_path,
                step_inputs=step_inputs,
                reason="empty_srclist",
            )
            det_src = empty_dir / det_name
            sky_src = empty_dir / sky_name
        else:
            common_params = _build_region_params(
                mcfg, evt_name=evt_path.name, extra=extra
            )
            symlinks: dict[str, Path | str] = {
                evt_path.name: evt_path,
                "srclist.fits": staged_srclist,
            }

            det_cmd = SASCommand(
                program="region",
                cwd=det_dir,
                params={
                    **common_params,
                    "outunit": "detxy",
                    "bkgregionset": det_name,
                    "tempset": f"tempset_{inst}_det.ds",
                    "regionset": f"src_{inst}_det.ds",
                },
                symlinks=symlinks,
                expected_outputs=[det_name],
                log_file=f"region_det_{inst}.log",
            )
            sky_cmd = SASCommand(
                program="region",
                cwd=sky_dir,
                params={
                    **common_params,
                    "outunit": "xy",
                    "bkgregionset": sky_name,
                    "tempset": f"tempset_{inst}_sky.ds",
                    "regionset": f"src_{inst}_sky.ds",
                },
                symlinks=symlinks,
                expected_outputs=[sky_name],
                log_file=f"region_sky_{inst}.log",
            )
            det_src = det_dir / det_name
            sky_src = sky_dir / sky_name
            try:
                ledger.run_step(
                    f"region_det:{inst}",
                    runner,
                    det_cmd,
                    inputs=step_inputs,
                )
                ledger.run_step(
                    f"region_sky:{inst}",
                    runner,
                    sky_cmd,
                    inputs=step_inputs,
                )
            except SASCommandError as exc:
                if mcfg.expression.strip() and _is_no_sources_selected_error(exc):
                    ledger.steps.pop(f"region_det:{inst}", None)
                    ledger.steps.pop(f"region_sky:{inst}", None)
                    _record_empty_region_step(
                        ledger,
                        inst=inst,
                        work_dir=empty_dir,
                        det_name=det_name,
                        sky_name=sky_name,
                        evt_path=evt_path,
                        step_inputs=step_inputs,
                        reason="no_sources_selected",
                        expression=mcfg.expression,
                    )
                    det_src = empty_dir / det_name
                    sky_src = empty_dir / sky_name
                else:
                    raise

        if not dry_run:
            ws.publish(det_src, det_name, mode="symlink")
            ws.publish(sky_src, sky_name, mode="symlink")
        region_det[inst] = target_dir / det_name
        region_sky[inst] = target_dir / sky_name

    emllist_published = target_dir / "emllist.fits"

    manifest = ledger.commit(
        products={
            "method": "region_mask",
            "emllist": emllist_published,
            "cheese_masks": {},
            "region_det": region_det,
            "region_sky": region_sky,
        },
        prune_unseen_steps=True,
    )
    return CheeseResult(
        cheese_dir=target_dir,
        manifest_path=ws.manifest_path,
        method="region_mask",
        emllist=emllist_published,
        cheese_masks={},
        region_det=region_det,
        region_sky=region_sky,
        workspace_root=ws.workspace_root,
        manifest=manifest,
    )
