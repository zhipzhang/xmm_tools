"""Soft-proton flare filtering stage (``espfilt`` + non-destructive FLAG ``evselect``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig
from xmm_tools.esas.naming import (
    ALLEVC_SUFFIX,
    discover_esas_prefix,
    env_calibration_inputs,
    load_stage_manifest_payload,
    relativize_path_map,
    resolve_path_map,
    save_stage_manifest,
    strip_esas_suffix,
)
from xmm_tools.exec import (
    SASCommand,
    SASCommandError,
    SASRunner,
    StageManifest,
    StageWorkspace,
    StepLedger,
    is_nonempty_file,
    normalize_lexical_path,
)

_PUBLISHED_PATTERNS = ("*-allevc.fits", "*-allevcoot.fits", "*-hist.qdp", "*-gti.fits")


@dataclass
class FlareResult:
    """Core artifacts produced by the ``flare`` stage."""

    flare_dir: Path
    manifest_path: Path
    flag_filtered: bool
    clean_events: dict[str, Path] = field(default_factory=dict)
    qdp_files: dict[str, Path] = field(default_factory=dict)
    gti_files: dict[str, Path] = field(default_factory=dict)
    workspace_root: Path | None = None
    manifest: StageManifest | None = None

    def save_manifest(self) -> Path:
        """Persist a v2 ``manifest.json`` preserving ``workspace_root`` portability."""
        ws_root = self.workspace_root
        products_payload: dict[str, Any] = {
            "flag_filtered": self.flag_filtered,
            "clean_events": relativize_path_map(
                self.clean_events, self.flare_dir, workspace_root=ws_root
            ),
            "qdp_files": relativize_path_map(
                self.qdp_files, self.flare_dir, workspace_root=ws_root
            ),
            "gti_files": relativize_path_map(
                self.gti_files, self.flare_dir, workspace_root=ws_root
            ),
        }
        self.manifest, written = save_stage_manifest(
            self.manifest_path,
            self.flare_dir,
            stage="flare",
            products=products_payload,
            existing_manifest=self.manifest,
            workspace_root=ws_root,
        )
        return written

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        workspace_root: Path | str | None = None,
        require_completed: bool = False,
    ) -> FlareResult:
        """Load a ``FlareResult`` from a v2 or legacy v1 ``manifest.json``."""
        mpath = normalize_lexical_path(manifest_path)
        fdir = mpath.parent
        ws_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else mpath.parent.parent
        )
        stage_manifest, payload = load_stage_manifest_payload(
            mpath, stage_label="Flare", require_completed=require_completed
        )
        return cls(
            flare_dir=fdir,
            manifest_path=mpath,
            flag_filtered=bool(payload.get("flag_filtered", False)),
            clean_events=resolve_path_map(fdir, payload.get("clean_events")),
            qdp_files=resolve_path_map(fdir, payload.get("qdp_files")),
            gti_files=resolve_path_map(fdir, payload.get("gti_files")),
            workspace_root=ws_root,
            manifest=stage_manifest,
        )


def _resolve_raw_inputs(
    cfg: AnaConfig, *, dry_run: bool
) -> tuple[list[tuple[str, Path]], Path | None]:
    active: list[tuple[str, Path]] = []
    for inst, evt_name in (
        ("mos1", cfg.instruments.mos1),
        ("mos2", cfg.instruments.mos2),
        ("pn", cfg.instruments.pn),
    ):
        if not evt_name:
            continue
        evt_path = normalize_lexical_path(cfg.paths.resolve(evt_name))
        if not dry_run and not is_nonempty_file(evt_path):
            raise FileNotFoundError(
                f"Raw event file for {inst!r} not found or empty: {evt_path}"
            )
        active.append((inst, evt_path))
    if not active:
        raise ValueError("No active instruments configured in [instruments]")

    pn_oot: Path | None = None
    if cfg.instruments.pn and cfg.instruments.pn_oot:
        oot_path = normalize_lexical_path(cfg.paths.resolve(cfg.instruments.pn_oot))
        if not dry_run and not is_nonempty_file(oot_path):
            raise FileNotFoundError(
                f"Raw PN OOT event file not found or empty: {oot_path}"
            )
        pn_oot = oot_path
    return active, pn_oot


def _collect_espfilt_outputs(
    esp_dir: Path, *, instrument: str, with_oot: bool
) -> dict[str, Path]:
    prefix = discover_esas_prefix(esp_dir, instrument, ALLEVC_SUFFIX)
    outputs = {
        "allevc": esp_dir / f"{prefix}-allevc.fits",
        "hist_qdp": esp_dir / f"{prefix}-hist.qdp",
        "gti": esp_dir / f"{prefix}-gti.fits",
    }
    if with_oot:
        outputs["allevcoot"] = esp_dir / f"{prefix}-allevcoot.fits"
    return outputs


def _resolve_prefix(
    esp_rec_outputs: Mapping[str, Any],
    esp_dir: Path,
    inst: str,
    evt_path: Path,
    *,
    dry_run: bool,
) -> str:
    if "allevc" in esp_rec_outputs:
        return strip_esas_suffix(esp_rec_outputs["allevc"].path)
    if dry_run:
        try:
            return discover_esas_prefix(esp_dir, inst, ALLEVC_SUFFIX)
        except SASCommandError:
            return strip_esas_suffix(evt_path)
    return discover_esas_prefix(esp_dir, inst, ALLEVC_SUFFIX)


def _prune_stale_published(flare_dir: Path, keep_names: set[str]) -> None:
    if not flare_dir.is_dir():
        return
    for pattern in _PUBLISHED_PATTERNS:
        for entry in flare_dir.glob(pattern):
            if entry.name not in keep_names and (
                entry.is_symlink() or entry.is_file()
            ):
                entry.unlink()


def load_flare_result(cfg: AnaConfig) -> FlareResult:
    """Load flare stage outputs from ``manifest.json`` (or deterministic scan)."""
    flare_dir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.flare_dir))
    ws_root = normalize_lexical_path(cfg.paths.workdir)
    manifest_path = flare_dir / "manifest.json"
    if manifest_path.is_file():
        return FlareResult.from_manifest(
            manifest_path, workspace_root=ws_root, require_completed=True
        )

    clean_events: dict[str, Path] = {}
    qdp_files: dict[str, Path] = {}
    gti_files: dict[str, Path] = {}

    for inst in cfg.instruments.active_keys():
        if not flare_dir.is_dir() or not any(
            is_nonempty_file(p) for p in flare_dir.glob(f"{inst}*{ALLEVC_SUFFIX}")
        ):
            continue
        prefix = discover_esas_prefix(flare_dir, inst, ALLEVC_SUFFIX)
        clean_events[inst] = flare_dir / f"{prefix}-allevc.fits"
        if inst == "pn":
            allevcoot = flare_dir / f"{prefix}-allevcoot.fits"
            if is_nonempty_file(allevcoot):
                clean_events["pn_oot"] = allevcoot
        hist = flare_dir / f"{prefix}-hist.qdp"
        if is_nonempty_file(hist):
            qdp_files[inst] = hist
        gti = flare_dir / f"{prefix}-gti.fits"
        if is_nonempty_file(gti):
            gti_files[inst] = gti

    return FlareResult(
        flare_dir=flare_dir,
        manifest_path=manifest_path,
        flag_filtered=False,
        clean_events=clean_events,
        qdp_files=qdp_files,
        gti_files=gti_files,
        workspace_root=ws_root,
    )


def run_flare(
    cfg: AnaConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> FlareResult:
    """Run ``espfilt`` (and non-destructive ESAS FLAG ``evselect``) per instrument."""
    flare_dir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.flare_dir))
    ws = StageWorkspace(
        workspace_root=normalize_lexical_path(cfg.paths.workdir),
        stage_dir=flare_dir,
    )
    ws.ensure_dirs(dry_run=dry_run)

    active_events, pn_oot_path = _resolve_raw_inputs(cfg, dry_run=dry_run)
    esp = cfg.flare
    cal_inputs = env_calibration_inputs(cfg, dry_run=dry_run)
    ledger = StepLedger(
        ws.manifest_path,
        stage="flare",
        workspace_root=ws.workspace_root,
        config={
            "apply_flag_filter": esp.apply_flag_filter,
            "elow": esp.elow,
            "ehigh": esp.ehigh,
            "method": esp.method,
            "instruments": [inst for inst, _ in active_events],
        },
        dry_run=dry_run,
        force=force,
    )
    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )

    clean_events: dict[str, Path] = {}
    qdp_files: dict[str, Path] = {}
    gti_files: dict[str, Path] = {}
    publish_sources: list[tuple[str, str, Path, dict[str, Path]]] = []

    for inst, evt_path in active_events:
        esp_dir = ws.step_dir("espfilt", inst)
        symlinks: dict[str, Path | str] = {evt_path.name: evt_path}
        params: dict[str, Any] = {
            "eventfile": evt_path.name,
            "elow": esp.elow,
            "ehigh": esp.ehigh,
            "withsmoothing": esp.withsmoothing,
            "smooth": esp.smooth,
            "rangescale": esp.rangescale_pn if inst == "pn" else esp.rangescale_mos,
            "allowsigma": esp.allowsigma,
            "method": esp.method,
            "keepinterfiles": esp.keepinterfiles,
        }
        esp_inputs = dict(cal_inputs)
        if not dry_run or is_nonempty_file(evt_path):
            esp_inputs["raw_event"] = evt_path

        has_oot = inst == "pn" and pn_oot_path is not None
        if has_oot and pn_oot_path is not None:
            symlinks[pn_oot_path.name] = pn_oot_path
            params["withoot"] = "Y"
            params["ootfile"] = pn_oot_path.name
            if not dry_run or is_nonempty_file(pn_oot_path):
                esp_inputs["pn_oot"] = pn_oot_path

        esp_cmd = SASCommand(
            program="espfilt",
            cwd=esp_dir,
            params=params,
            symlinks=symlinks,
            expected_outputs=[],
            log_file=f"{inst}_espfilt.log",
        )
        step_id = f"espfilt:{inst}"
        if (
            not dry_run
            and ledger.can_reuse(step_id, esp_cmd, inputs=esp_inputs) is None
            and esp_dir.is_dir()
        ):
            for pat in ("*-allevc*.fits", "*-hist.qdp", "*-gti.fits"):
                for stale in esp_dir.glob(pat):
                    if stale.is_file() or stale.is_symlink():
                        stale.unlink()

        esp_rec = ledger.run_step(
            step_id,
            runner,
            esp_cmd,
            inputs=esp_inputs,
            collect_outputs=lambda d, i=inst, o=has_oot: _collect_espfilt_outputs(
                d, instrument=i, with_oot=o
            ),
        )
        prefix = _resolve_prefix(
            esp_rec.outputs, esp_dir, inst, evt_path, dry_run=dry_run
        )

        raw_allevc = esp_dir / f"{prefix}-allevc.fits"
        expr = esp.pn_flag_expr if inst == "pn" else esp.mos_flag_expr
        flag_sources: dict[str, Path] = {inst: raw_allevc}
        if has_oot:
            flag_sources["pn_oot"] = esp_dir / f"{prefix}-allevcoot.fits"

        selected_sources: dict[str, Path] = dict(flag_sources)
        if esp.apply_flag_filter:
            selected_sources = {}
            for key, raw_src in flag_sources.items():
                flag_dir = ws.step_dir("flag_filter", key)
                flag_inputs = dict(cal_inputs)
                if not dry_run or is_nonempty_file(raw_src):
                    flag_inputs["unflagged"] = raw_src
                flag_cmd = SASCommand(
                    program="evselect",
                    cwd=flag_dir,
                    params={
                        "table": "unflagged_events.fits",
                        "expression": expr,
                        "filtertype": "expression",
                        "keepfilteroutput": True,
                        "filteredset": "clean_events.fits",
                    },
                    symlinks={"unflagged_events.fits": raw_src},
                    expected_outputs=["clean_events.fits"],
                    log_file=f"{key}_flag.log",
                )
                ledger.run_step(
                    f"flag_filter:{key}", runner, flag_cmd, inputs=flag_inputs
                )
                selected_sources[key] = flag_dir / "clean_events.fits"

        publish_sources.append((inst, prefix, esp_dir, selected_sources))

    keep_names: set[str] = set()
    for inst, prefix, esp_dir, selected_sources in publish_sources:
        allevc_name = f"{prefix}-allevc.fits"
        hist_name = f"{prefix}-hist.qdp"
        gti_name = f"{prefix}-gti.fits"
        keep_names.update({allevc_name, hist_name, gti_name})
        if dry_run:
            clean_events[inst] = flare_dir / allevc_name
            qdp_files[inst] = flare_dir / hist_name
            gti_files[inst] = flare_dir / gti_name
            if "pn_oot" in selected_sources:
                oot_name = f"{prefix}-allevcoot.fits"
                keep_names.add(oot_name)
                clean_events["pn_oot"] = flare_dir / oot_name
        else:
            clean_events[inst] = ws.publish(
                selected_sources[inst], allevc_name, mode="symlink"
            )
            qdp_files[inst] = ws.publish(
                esp_dir / hist_name, hist_name, mode="symlink"
            )
            gti_files[inst] = ws.publish(
                esp_dir / gti_name, gti_name, mode="symlink"
            )
            if "pn_oot" in selected_sources:
                oot_name = f"{prefix}-allevcoot.fits"
                keep_names.add(oot_name)
                clean_events["pn_oot"] = ws.publish(
                    selected_sources["pn_oot"], oot_name, mode="symlink"
                )

    if not dry_run:
        _prune_stale_published(flare_dir, keep_names)

    if ledger.previous is not None:
        active_suffixes = {inst for inst, _ in active_events}
        if pn_oot_path is not None:
            active_suffixes.add("pn_oot")
        ledger.previous.steps = {
            sid: rec
            for sid, rec in ledger.previous.steps.items()
            if sid.split(":", 1)[-1] in active_suffixes
        }

    flag_filtered = bool(esp.apply_flag_filter and not dry_run)
    manifest = ledger.commit(
        products={
            "flag_filtered": flag_filtered,
            "clean_events": clean_events,
            "qdp_files": qdp_files,
            "gti_files": gti_files,
        },
        prune_unseen_steps=False,
    )
    return FlareResult(
        flare_dir=flare_dir,
        manifest_path=ws.manifest_path,
        flag_filtered=flag_filtered,
        clean_events=clean_events,
        qdp_files=qdp_files,
        gti_files=gti_files,
        workspace_root=ws.workspace_root,
        manifest=manifest,
    )
