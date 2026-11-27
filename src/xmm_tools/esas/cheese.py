"""ESAS ``cheese`` point-source detection and swiss-cheese masking stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from xmm_tools.config import AnaConfig
from xmm_tools.esas.flare import FlareResult, load_flare_result
from xmm_tools.esas.naming import (
    env_calibration_inputs,
    load_stage_manifest_payload,
    relativize_path_map,
    resolve_path_map,
    save_stage_manifest,
    strip_esas_suffix,
    validate_scalar_extra_params,
)
from xmm_tools.exec import (
    SASCommand,
    SASRunner,
    StageManifest,
    StageWorkspace,
    StepLedger,
    is_nonempty_file,
    normalize_lexical_path,
    resolve_workspace_path,
    workspace_relative_path,
)

_CHEESE_RESERVED_PARAMS = frozenset({"mos1file", "mos2file", "pnfile", "pnootfile"})


@dataclass
class CheeseResult:
    """Core artifacts produced by the ``cheese`` stage."""

    cheese_dir: Path
    manifest_path: Path
    method: str = "cheese"
    emllist: Path | None = None
    cheese_masks: dict[str, Path] = field(default_factory=dict)
    region_det: dict[str, Path] = field(default_factory=dict)
    region_sky: dict[str, Path] = field(default_factory=dict)
    workspace_root: Path | None = None
    manifest: StageManifest | None = None

    def save_manifest(self) -> Path:
        """Persist a v2 ``manifest.json`` preserving ``workspace_root`` portability."""
        ws_root = self.workspace_root
        products_payload: dict[str, Any] = {
            "method": self.method,
            "emllist": (
                workspace_relative_path(
                    self.emllist, self.cheese_dir, workspace_root=ws_root
                )
                if self.emllist is not None
                else None
            ),
            "cheese_masks": relativize_path_map(
                self.cheese_masks, self.cheese_dir, workspace_root=ws_root
            ),
            "region_det": relativize_path_map(
                self.region_det, self.cheese_dir, workspace_root=ws_root
            ),
            "region_sky": relativize_path_map(
                self.region_sky, self.cheese_dir, workspace_root=ws_root
            ),
        }
        self.manifest, written = save_stage_manifest(
            self.manifest_path,
            self.cheese_dir,
            stage="cheese",
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
    ) -> CheeseResult:
        """Load a ``CheeseResult`` from a v2 or legacy v1 ``manifest.json``."""
        mpath = normalize_lexical_path(manifest_path)
        cdir = mpath.parent
        ws_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else mpath.parent.parent
        )
        stage_manifest, payload = load_stage_manifest_payload(
            mpath, stage_label="Cheese", require_completed=require_completed
        )
        emllist_raw = payload.get("emllist")
        return cls(
            cheese_dir=cdir,
            manifest_path=mpath,
            method=str(payload.get("method", "cheese")),
            emllist=(
                resolve_workspace_path(cdir, str(emllist_raw)) if emllist_raw else None
            ),
            cheese_masks=resolve_path_map(cdir, payload.get("cheese_masks")),
            region_det=resolve_path_map(cdir, payload.get("region_det")),
            region_sky=resolve_path_map(cdir, payload.get("region_sky")),
            workspace_root=ws_root,
            manifest=stage_manifest,
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
        sd = cfg.cheese
        extra = validate_scalar_extra_params(
            sd.extra_params,
            section="cheese.extra_params",
            reserved_keys=_CHEESE_RESERVED_PARAMS,
        )

        cheese_dir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.cheese_dir))
        ws = StageWorkspace(
            workspace_root=normalize_lexical_path(cfg.paths.workdir),
            stage_dir=cheese_dir,
        )
        ws.ensure_dirs(dry_run=dry_run)

        work_dir = ws.step_dir("cheese")
        symlinks: dict[str, Path | str] = {}
        params: dict[str, Any] = {}
        inputs: dict[str, Path] = env_calibration_inputs(cfg, dry_run=dry_run)
        expected_outputs: list[str] = ["emllist.fits"]
        publish_names: list[str] = ["emllist.fits"]

        cheese_masks: dict[str, Path] = {}
        region_det: dict[str, Path] = {}
        region_sky: dict[str, Path] = {}

        for inst_key, param_name in (
            ("mos1", "mos1file"),
            ("mos2", "mos2file"),
            ("pn", "pnfile"),
        ):
            raw_evt = flare.clean_events.get(inst_key)
            if raw_evt is None:
                continue
            evt_path = normalize_lexical_path(raw_evt)
            if not dry_run and not is_nonempty_file(evt_path):
                raise FileNotFoundError(
                    f"Clean event file for {inst_key!r} not found or empty: {evt_path}"
                )
            symlinks[evt_path.name] = evt_path
            params[param_name] = evt_path.name
            if not dry_run or is_nonempty_file(evt_path):
                inputs[inst_key] = evt_path

            prefix = strip_esas_suffix(evt_path)
            mask_name = f"{prefix}-cheeset.fits"
            det_name = f"{prefix}-bkgregtdet.fits"
            sky_name = f"{prefix}-bkgregtsky.fits"
            expected_outputs.extend([mask_name, det_name, sky_name])
            publish_names.extend([mask_name, det_name, sky_name])
            cheese_masks[inst_key] = cheese_dir / mask_name
            region_det[inst_key] = cheese_dir / det_name
            region_sky[inst_key] = cheese_dir / sky_name

        if not cheese_masks:
            raise ValueError(
                "No clean event files found for cheese; run 'xmm-ana flare' first."
            )

        pn_oot_raw = flare.clean_events.get("pn_oot")
        if pn_oot_raw is not None:
            pn_oot_path = normalize_lexical_path(pn_oot_raw)
            if not dry_run and not is_nonempty_file(pn_oot_path):
                raise FileNotFoundError(
                    f"Clean PN OOT event file not found or empty: {pn_oot_path}"
                )
            symlinks[pn_oot_path.name] = pn_oot_path
            params["pnootfile"] = pn_oot_path.name
            if not dry_run or is_nonempty_file(pn_oot_path):
                inputs["pn_oot"] = pn_oot_path

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
                **extra,
            }
        )

        ledger = StepLedger(
            ws.manifest_path,
            stage="cheese",
            workspace_root=ws.workspace_root,
            config={
                "method": self.name,
                "elowlist": list(sd.elowlist),
                "ehighlist": list(sd.ehighlist),
                "mlmin": sd.mlmin,
                "scale": sd.scale,
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
        cmd = SASCommand(
            program="cheese",
            cwd=work_dir,
            params=params,
            symlinks=symlinks,
            expected_outputs=expected_outputs,
            log_file="cheese.log",
        )
        ledger.run_step("cheese", runner, cmd, inputs=inputs)

        if not dry_run:
            for name in publish_names:
                ws.publish(work_dir / name, name, mode="symlink")

        emllist_path = cheese_dir / "emllist.fits"
        manifest = ledger.commit(
            products={
                "method": self.name,
                "emllist": emllist_path,
                "cheese_masks": cheese_masks,
                "region_det": region_det,
                "region_sky": region_sky,
            },
            prune_unseen_steps=True,
        )
        return CheeseResult(
            cheese_dir=cheese_dir,
            manifest_path=ws.manifest_path,
            method=self.name,
            emllist=emllist_path,
            cheese_masks=cheese_masks,
            region_det=region_det,
            region_sky=region_sky,
            workspace_root=ws.workspace_root,
            manifest=manifest,
        )


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


def _resolve_mask_dir(cfg: AnaConfig, mask_dir: str | Path | None) -> Path:
    if mask_dir is not None and str(mask_dir).strip():
        raw = Path(mask_dir)
        if raw.is_absolute():
            return normalize_lexical_path(raw)
        ws_candidate = normalize_lexical_path(cfg.paths.resolve(raw))
        cwd_candidate = normalize_lexical_path(raw)
        if ws_candidate.is_dir():
            return ws_candidate
        if cwd_candidate.is_dir():
            return cwd_candidate
        return ws_candidate
    return normalize_lexical_path(cfg.paths.resolve(cfg.paths.cheese_dir))


def _scan_unique_mask_file(
    target_dir: Path, inst_key: str, suffix: str
) -> Path | None:
    if not target_dir.is_dir():
        return None
    matches: list[Path] = []
    for p in sorted(target_dir.glob(f"{inst_key}*{suffix}")):
        if not is_nonempty_file(p):
            continue
        mid = p.name[len(inst_key) : -len(suffix)]
        if mid[:1].isdigit():
            continue
        matches.append(p)
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise ValueError(
            f"Ambiguous mask files for {inst_key!r} (*{suffix}) in {target_dir}: {names}"
        )
    return matches[0] if matches else None


def load_cheese_result(
    cfg: AnaConfig,
    *,
    mask_dir: str | Path | None = None,
) -> CheeseResult:
    """Load ``cheese`` or ``mask`` outputs from ``manifest.json`` (or scan directory)."""
    target_dir = _resolve_mask_dir(cfg, mask_dir)
    ws_root = normalize_lexical_path(cfg.paths.workdir)
    manifest_path = target_dir / "manifest.json"
    if manifest_path.is_file():
        return CheeseResult.from_manifest(
            manifest_path, workspace_root=ws_root, require_completed=True
        )

    emllist = target_dir / "emllist.fits"
    cheese_masks: dict[str, Path] = {}
    region_det: dict[str, Path] = {}
    region_sky: dict[str, Path] = {}

    flare: FlareResult | None = None
    try:
        flare = load_flare_result(cfg)
    except (FileNotFoundError, RuntimeError):
        flare = None

    if flare is not None:
        for inst_key in ("mos1", "mos2", "pn"):
            evt = flare.clean_events.get(inst_key)
            if evt is None:
                continue
            prefix = strip_esas_suffix(evt)
            mask = target_dir / f"{prefix}-cheeset.fits"
            det = target_dir / f"{prefix}-bkgregtdet.fits"
            sky = target_dir / f"{prefix}-bkgregtsky.fits"
            if is_nonempty_file(mask):
                cheese_masks[inst_key] = mask
            if is_nonempty_file(det):
                region_det[inst_key] = det
            if is_nonempty_file(sky):
                region_sky[inst_key] = sky

    for inst_key in ("mos1", "mos2", "pn"):
        if inst_key not in cheese_masks:
            found = _scan_unique_mask_file(target_dir, inst_key, "-cheeset.fits")
            if found is not None:
                cheese_masks[inst_key] = found
        if inst_key not in region_det:
            found = _scan_unique_mask_file(target_dir, inst_key, "-bkgregtdet.fits")
            if found is not None:
                region_det[inst_key] = found
        if inst_key not in region_sky:
            found = _scan_unique_mask_file(target_dir, inst_key, "-bkgregtsky.fits")
            if found is not None:
                region_sky[inst_key] = found

    return CheeseResult(
        cheese_dir=target_dir,
        manifest_path=manifest_path,
        method="cheese",
        emllist=emllist if is_nonempty_file(emllist) else None,
        cheese_masks=cheese_masks,
        region_det=region_det,
        region_sky=region_sky,
        workspace_root=ws_root,
    )
