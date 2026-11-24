"""Soft-proton flare filtering stage (``espfilt`` + ESAS FLAG ``evselect``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from xmm_tools.esas.config import ESASConfig
from xmm_tools.sas.command import SASCommand
from xmm_tools.sas.runner import SASRunner


@dataclass
class FlareResult:
    """Core artifacts produced by the ``flare`` stage."""

    flare_dir: Path
    manifest_path: Path
    flag_filtered: bool
    clean_events: dict[str, Path] = field(default_factory=dict)
    qdp_files: dict[str, Path] = field(default_factory=dict)
    gti_files: dict[str, Path] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, object]:
        def _rel(mapping: dict[str, Path]) -> dict[str, str]:
            return {k: p.name for k, p in mapping.items()}

        return {
            "stage": "flare",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "flag_filtered": self.flag_filtered,
            "clean_events": _rel(self.clean_events),
            "qdp_files": _rel(self.qdp_files),
            "gti_files": _rel(self.gti_files),
        }

    def save_manifest(self) -> Path:
        self.flare_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.to_manifest_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> FlareResult:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        flare_dir = manifest_path.parent

        def _abs(section: str) -> dict[str, Path]:
            entries = raw.get(section, {})
            return {k: flare_dir / v for k, v in entries.items()}

        return cls(
            flare_dir=flare_dir,
            manifest_path=manifest_path,
            flag_filtered=bool(raw.get("flag_filtered", False)),
            clean_events=_abs("clean_events"),
            qdp_files=_abs("qdp_files"),
            gti_files=_abs("gti_files"),
        )


def _prefix_from_evt(evt_filename: str) -> str:
    name = Path(evt_filename).name
    for suffix in (".fits", ".fit", ".FITS", ".FIT"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_flare_result(cfg: ESASConfig) -> FlareResult:
    """Load flare stage outputs from ``manifest.json`` (or scan directory)."""
    flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
    manifest_path = flare_dir / "manifest.json"
    if manifest_path.exists():
        return FlareResult.from_manifest(manifest_path)

    # Fallback for directories created manually without manifest.json
    clean_events: dict[str, Path] = {}
    qdp_files: dict[str, Path] = {}
    gti_files: dict[str, Path] = {}

    for inst, evt_name in cfg.instruments.active_mos():
        prefix = _prefix_from_evt(evt_name)
        allevc = flare_dir / f"{prefix}-allevc.fits"
        if allevc.exists():
            clean_events[inst] = allevc
        hist = flare_dir / f"{prefix}-hist.qdp"
        if hist.exists():
            qdp_files[inst] = hist
        gti = flare_dir / f"{prefix}-gti.fits"
        if gti.exists():
            gti_files[inst] = gti

    if cfg.instruments.pn:
        prefix = _prefix_from_evt(cfg.instruments.pn)
        allevc = flare_dir / f"{prefix}-allevc.fits"
        if allevc.exists():
            clean_events["pn"] = allevc
        allevcoot = flare_dir / f"{prefix}-allevcoot.fits"
        if allevcoot.exists():
            clean_events["pn_oot"] = allevcoot
        hist = flare_dir / f"{prefix}-hist.qdp"
        if hist.exists():
            qdp_files["pn"] = hist
        gti = flare_dir / f"{prefix}-gti.fits"
        if gti.exists():
            gti_files["pn"] = gti

    return FlareResult(
        flare_dir=flare_dir,
        manifest_path=manifest_path,
        flag_filtered=False,
        clean_events=clean_events,
        qdp_files=qdp_files,
        gti_files=gti_files,
    )


def _all_nonempty(paths: dict[str, Path]) -> bool:
    return bool(paths) and all(p.exists() and p.stat().st_size > 0 for p in paths.values())


def run_flare(
    cfg: ESASConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> FlareResult:
    """Run ``espfilt`` (and optional ESAS FLAG ``evselect``) for active instruments."""
    flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
    manifest_path = flare_dir / "manifest.json"
    esp = cfg.espfilt

    # Check if already completed (including FLAG requirement)
    if not force and manifest_path.exists():
        existing = FlareResult.from_manifest(manifest_path)
        flag_ok = (not esp.apply_flag_filter) or existing.flag_filtered
        if flag_ok and _all_nonempty(existing.clean_events):
            return existing

    runner = SASRunner(
        env=cfg.env,
        dry_run=dry_run,
        skip_if_exists=not force,
        docker_container=docker_container,
    )

    clean_events: dict[str, Path] = {}
    qdp_files: dict[str, Path] = {}
    gti_files: dict[str, Path] = {}
    flag_targets: list[tuple[str, str, str]] = []  # (key, filename, expr)

    # 1. Run espfilt for MOS1 / MOS2
    for inst, evt_name in cfg.instruments.active_mos():
        prefix = _prefix_from_evt(evt_name)
        allevc_name = f"{prefix}-allevc.fits"
        hist_name = f"{prefix}-hist.qdp"
        gti_name = f"{prefix}-gti.fits"

        cmd = SASCommand(
            program="espfilt",
            cwd=flare_dir,
            params={
                "eventfile": Path(evt_name).name,
                "elow": esp.elow,
                "ehigh": esp.ehigh,
                "withsmoothing": esp.withsmoothing,
                "smooth": esp.smooth,
                "rangescale": esp.rangescale_mos,
                "allowsigma": esp.allowsigma,
                "method": esp.method,
                "keepinterfiles": esp.keepinterfiles,
            },
            symlinks={Path(evt_name).name: cfg.paths.resolve(evt_name)},
            core_outputs=[allevc_name, hist_name, gti_name],
            intermediate_dir="_intermediate",
            log_file=f"{inst}_espfilt.log",
        )
        runner.run(cmd, force=force)
        clean_events[inst] = flare_dir / allevc_name
        qdp_files[inst] = flare_dir / hist_name
        gti_files[inst] = flare_dir / gti_name
        flag_targets.append((inst, allevc_name, esp.mos_flag_expr))

    # 2. Run espfilt for PN (+ OOT)
    if cfg.instruments.pn:
        pn_evt = cfg.instruments.pn
        prefix = _prefix_from_evt(pn_evt)
        allevc_name = f"{prefix}-allevc.fits"
        allevcoot_name = f"{prefix}-allevcoot.fits"
        hist_name = f"{prefix}-hist.qdp"
        gti_name = f"{prefix}-gti.fits"

        symlinks: dict[str, Path | str] = {
            Path(pn_evt).name: cfg.paths.resolve(pn_evt)
        }
        pn_params: dict[str, object] = {
            "eventfile": Path(pn_evt).name,
            "elow": esp.elow,
            "ehigh": esp.ehigh,
            "withsmoothing": esp.withsmoothing,
            "smooth": esp.smooth,
            "rangescale": esp.rangescale_pn,
            "allowsigma": esp.allowsigma,
            "method": esp.method,
            "keepinterfiles": esp.keepinterfiles,
        }
        core = [allevc_name, hist_name, gti_name]

        if cfg.instruments.pn_oot:
            oot_evt = cfg.instruments.pn_oot
            symlinks[Path(oot_evt).name] = cfg.paths.resolve(oot_evt)
            pn_params["withoot"] = "Y"
            pn_params["ootfile"] = Path(oot_evt).name
            core.append(allevcoot_name)

        cmd = SASCommand(
            program="espfilt",
            cwd=flare_dir,
            params=pn_params,
            symlinks=symlinks,
            core_outputs=core,
            intermediate_dir="_intermediate",
            log_file="pn_espfilt.log",
        )
        runner.run(cmd, force=force)
        clean_events["pn"] = flare_dir / allevc_name
        qdp_files["pn"] = flare_dir / hist_name
        gti_files["pn"] = flare_dir / gti_name
        flag_targets.append(("pn", allevc_name, esp.pn_flag_expr))
        if cfg.instruments.pn_oot:
            clean_events["pn_oot"] = flare_dir / allevcoot_name
            flag_targets.append(("pn_oot", allevcoot_name, esp.pn_flag_expr))

    # 3. Apply ESAS FLAG filtering via evselect and atomically replace allevc files
    flag_filtered = False
    if esp.apply_flag_filter:
        flag_runner = SASRunner(
            env=cfg.env,
            dry_run=dry_run,
            skip_if_exists=False,
            docker_container=docker_container,
        )
        for key, filename, expr in flag_targets:
            stem = _prefix_from_evt(filename)
            new_name = f"{stem}-new.fits"
            ev_cmd = SASCommand(
                program="evselect",
                cwd=flare_dir,
                params={
                    "table": filename,
                    "expression": expr,
                    "filtertype": "expression",
                    "keepfilteroutput": True,
                    "filteredset": new_name,
                },
                core_outputs=[new_name],
                intermediate_dir=None,
                log_file=f"{key}_flag.log",
            )
            flag_runner.run(ev_cmd, force=True)
            if not dry_run:
                shutil.move(str(flare_dir / new_name), str(flare_dir / filename))
        flag_filtered = not dry_run

    result = FlareResult(
        flare_dir=flare_dir,
        manifest_path=manifest_path,
        flag_filtered=flag_filtered,
        clean_events=clean_events,
        qdp_files=qdp_files,
        gti_files=gti_files,
    )
    if not dry_run:
        result.save_manifest()
    return result
