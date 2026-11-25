"""ESAS ``cheese`` point-source detection and swiss-cheese masking stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.exec.command import SASCommand
from xmm_tools.exec.runner import SASRunner
from xmm_tools.pipeline.config import AnaConfig
from xmm_tools.pipeline.stage import (
    StageManifest,
    all_nonempty,
    read_manifest_json,
    relativize_manifest_paths,
    resolve_manifest_paths,
)


@dataclass
class CheeseResult(StageManifest):
    """Core artifacts produced by the ``cheese`` stage."""

    cheese_dir: Path
    manifest_path: Path
    method: str = "cheese"
    emllist: Path | None = None
    cheese_masks: dict[str, Path] = field(default_factory=dict)
    region_det: dict[str, Path] = field(default_factory=dict)
    region_sky: dict[str, Path] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "stage": "cheese",
            "method": self.method,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "emllist": self.emllist.name if self.emllist is not None else None,
            "cheese_masks": relativize_manifest_paths(self.cheese_masks),
            "region_det": relativize_manifest_paths(self.region_det),
            "region_sky": relativize_manifest_paths(self.region_sky),
        }

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> CheeseResult:
        raw = read_manifest_json(manifest_path)
        cheese_dir = manifest_path.parent
        emllist_name = raw.get("emllist")
        return cls(
            cheese_dir=cheese_dir,
            manifest_path=manifest_path,
            method=str(raw.get("method", "cheese")),
            emllist=cheese_dir / emllist_name if emllist_name else None,
            cheese_masks=resolve_manifest_paths(cheese_dir, raw, "cheese_masks"),
            region_det=resolve_manifest_paths(cheese_dir, raw, "region_det"),
            region_sky=resolve_manifest_paths(cheese_dir, raw, "region_sky"),
        )


class SourceDetector(Protocol):
    """Protocol for pluggable point-source detection backends."""

    def detect(
        self,
        cfg: AnaConfig,
        flare: FlareResult,
        *,
        dry_run: bool = False,
        force: bool = False,
        docker_container: str | None = None,
    ) -> CheeseResult: ...


def _stem_from_allevc(path: Path) -> str:
    """Strip ``-allevc.fits`` (or ``.fits``) to get the ESAS output prefix."""
    name = path.name
    for suffix in ("-allevc.fits", "-allevcoot.fits", ".fits"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


class CheeseDetector:
    """Default ESAS ``cheese`` point-source detector and mask generator."""

    name = "cheese"

    def detect(
        self,
        cfg: AnaConfig,
        flare: FlareResult,
        *,
        dry_run: bool = False,
        force: bool = False,
        docker_container: str | None = None,
    ) -> CheeseResult:
        cheese_dir = cfg.paths.resolve(cfg.paths.cheese_dir)
        manifest_path = cheese_dir / "manifest.json"

        if not force and manifest_path.exists():
            existing = CheeseResult.from_manifest(manifest_path)
            active_insts = [k for k in ("mos1", "mos2", "pn") if k in flare.clean_events]
            if (
                existing.emllist is not None
                and existing.emllist.exists()
                and existing.emllist.stat().st_size > 0
                and all(k in existing.cheese_masks for k in active_insts)
                and all_nonempty(existing.cheese_masks)
                and all_nonempty(existing.region_det)
                and all_nonempty(existing.region_sky)
            ):
                return existing

        sd = cfg.cheese
        symlinks: dict[str, Path | str] = {}
        params: dict[str, Any] = {}
        core_outputs: list[str] = ["emllist.fits"]

        cheese_masks: dict[str, Path] = {}
        region_det: dict[str, Path] = {}
        region_sky: dict[str, Path] = {}

        inst_param_map = (
            ("mos1", "mos1file"),
            ("mos2", "mos2file"),
            ("pn", "pnfile"),
        )
        for inst_key, param_name in inst_param_map:
            evt_path = flare.clean_events.get(inst_key)
            if evt_path is None:
                continue
            symlinks[evt_path.name] = evt_path
            params[param_name] = evt_path.name

            prefix = _stem_from_allevc(evt_path)
            mask_name = f"{prefix}-cheeset.fits"
            det_name = f"{prefix}-bkgregtdet.fits"
            sky_name = f"{prefix}-bkgregtsky.fits"
            core_outputs.extend([mask_name, det_name, sky_name])
            cheese_masks[inst_key] = cheese_dir / mask_name
            region_det[inst_key] = cheese_dir / det_name
            region_sky[inst_key] = cheese_dir / sky_name

        if not cheese_masks:
            raise ValueError(
                "No clean event files found for cheese; run 'xmm-ana flare' first."
            )

        pn_oot_path = flare.clean_events.get("pn_oot")
        if pn_oot_path is not None:
            symlinks[pn_oot_path.name] = pn_oot_path
            params["pnootfile"] = pn_oot_path.name

        params.update(
            {
                "elowlist": sd.elowlist,
                "ehighlist": sd.ehighlist,
                "scale": sd.scale,
                "mlmin": sd.mlmin,
                "dist": sd.dist,
                "ratesoft": sd.ratesoft,
                "ratehard": sd.ratehard,
                "ratetotal": sd.ratetotal,
                "keepinterfiles": sd.keepinterfiles,
            }
        )
        params.update(sd.extra_params)

        cmd = SASCommand(
            program="cheese",
            cwd=cheese_dir,
            params=params,
            symlinks=symlinks,
            core_outputs=core_outputs,
            intermediate_dir="_intermediate",
            log_file="cheese.log",
        )
        runner = SASRunner(
            env=cfg.env,
            dry_run=dry_run,
            skip_if_exists=not force,
            docker_container=docker_container,
        )
        runner.run(cmd, force=force)

        result = CheeseResult(
            cheese_dir=cheese_dir,
            manifest_path=manifest_path,
            method=self.name,
            emllist=cheese_dir / "emllist.fits",
            cheese_masks=cheese_masks,
            region_det=region_det,
            region_sky=region_sky,
        )
        if not dry_run:
            result.save_manifest()
        return result


def run_cheese(
    cfg: AnaConfig,
    *,
    flare: FlareResult | None = None,
    detector: SourceDetector | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> CheeseResult:
    """Run ESAS ``cheese`` point-source detection and masking."""
    flare_res = flare if flare is not None else load_flare_result(cfg)
    active_detector = detector if detector is not None else CheeseDetector()
    return active_detector.detect(
        cfg,
        flare_res,
        dry_run=dry_run,
        force=force,
        docker_container=docker_container,
    )


def load_cheese_result(cfg: AnaConfig) -> CheeseResult:
    """Load ``cheese`` outputs from ``manifest.json`` (or scan directory)."""
    cheese_dir = cfg.paths.resolve(cfg.paths.cheese_dir)
    manifest_path = cheese_dir / "manifest.json"
    if manifest_path.exists():
        return CheeseResult.from_manifest(manifest_path)

    emllist = cheese_dir / "emllist.fits"
    cheese_masks: dict[str, Path] = {}
    region_det: dict[str, Path] = {}
    region_sky: dict[str, Path] = {}

    flare = load_flare_result(cfg)
    for inst_key in ("mos1", "mos2", "pn"):
        evt = flare.clean_events.get(inst_key)
        if evt is None:
            continue
        prefix = _stem_from_allevc(evt)
        mask = cheese_dir / f"{prefix}-cheeset.fits"
        det = cheese_dir / f"{prefix}-bkgregtdet.fits"
        sky = cheese_dir / f"{prefix}-bkgregtsky.fits"
        if mask.exists():
            cheese_masks[inst_key] = mask
        if det.exists():
            region_det[inst_key] = det
        if sky.exists():
            region_sky[inst_key] = sky

    return CheeseResult(
        cheese_dir=cheese_dir,
        manifest_path=manifest_path,
        method="cheese",
        emllist=emllist if emllist.exists() else None,
        cheese_masks=cheese_masks,
        region_det=region_det,
        region_sky=region_sky,
    )
