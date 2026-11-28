"""Shared ESAS naming, prefix discovery, calibration inputs, and param validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig
from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    SASCommandError,
    StageManifest,
    atomic_write_json,
    is_nonempty_file,
    normalize_lexical_path,
    read_json,
    resolve_workspace_path,
    workspace_relative_path,
)

ALLEVC_SUFFIX = "-allevc.fits"
ALLEVCOOT_SUFFIX = "-allevcoot.fits"
FOVT_SUFFIX = "-fovt.pi"

_STRIP_SUFFIXES: tuple[str, ...] = (
    ALLEVCOOT_SUFFIX,
    ALLEVC_SUFFIX,
    FOVT_SUFFIX,
    ".fits",
    ".fit",
    ".FITS",
    ".FIT",
)


def strip_esas_suffix(filename: str | Path) -> str:
    """Strip ESAS/FITS filename suffixes to recover the exposure prefix."""
    name = Path(filename).name
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_esas_prefix(
    directory: Path | str,
    instrument: str,
    suffix: str,
) -> str:
    """Discover the unique non-empty ESAS exposure prefix for ``instrument``."""
    root = normalize_lexical_path(directory)
    if not root.is_dir():
        raise SASCommandError(
            f"Directory for {instrument!r} (*{suffix}) does not exist: {root}"
        )
    pattern = f"{instrument}*{suffix}"
    matches = sorted(p for p in root.glob(pattern) if is_nonempty_file(p))
    if not matches:
        raise SASCommandError(
            f"No non-empty '{pattern}' output found for {instrument} in {root}"
        )
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise SASCommandError(
            f"Multiple '{pattern}' outputs found for {instrument} in {root}: {names}"
        )
    return matches[0].name[: -len(suffix)]


def env_calibration_inputs(cfg: AnaConfig, *, dry_run: bool) -> dict[str, Path]:
    """Return existing calibration file inputs (``ccf``, ``odf_sum``) for hashing."""
    inputs: dict[str, Path] = {}
    for key, raw in (("ccf", cfg.env.sas_ccf), ("odf_sum", cfg.env.sas_odf)):
        if raw is None or not str(raw).strip():
            continue
        path = normalize_lexical_path(cfg.paths.resolve(str(raw).strip()))
        if is_nonempty_file(path) and (not dry_run or path.is_file()):
            inputs[key] = path
    return inputs


def _contains_path_value(value: Any) -> bool:
    if isinstance(value, Path):
        return True
    if isinstance(value, str):
        return "/" in value or "\\" in value
    if isinstance(value, (list, tuple)):
        return any(_contains_path_value(item) for item in value)
    return False


def validate_scalar_extra_params(
    extra: Mapping[str, Any],
    *,
    section: str,
    reserved_keys: frozenset[str],
) -> dict[str, Any]:
    """Validate ``extra_params`` against reserved structural keys and path values."""
    validated: dict[str, Any] = {}
    for raw_key, val in extra.items():
        key = str(raw_key)
        if key.lower() in reserved_keys:
            raise ValueError(
                f"Reserved structural parameter {key!r} is not allowed in "
                f"{section}; reserved keys: {sorted(reserved_keys)}"
            )
        if _contains_path_value(val):
            raise ValueError(
                f"Path-valued parameter {key!r}={val!r} is not allowed in "
                f"{section}; extra_params are restricted to scalar SAS options "
                "to prevent untracked path leakage."
            )
        validated[key] = val
    return validated


def relativize_path_map(
    paths: Mapping[str, Path],
    base_dir: Path,
    *,
    workspace_root: Path | None,
) -> dict[str, str]:
    """Convert a ``{key: Path}`` mapping into workspace-portable relative strings."""
    return {
        k: workspace_relative_path(p, base_dir, workspace_root=workspace_root)
        for k, p in paths.items()
    }


def resolve_path_map(base_dir: Path, raw: Any) -> dict[str, Path]:
    """Resolve a serialized ``{key: path_str}`` mapping against ``base_dir``."""
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(k): resolve_workspace_path(base_dir, str(v))
        for k, v in raw.items()
        if v is not None and str(v).strip()
    }


def save_stage_manifest(
    manifest_path: Path,
    stage_dir: Path,
    *,
    stage: str,
    products: Mapping[str, Any],
    existing_manifest: StageManifest | None,
    workspace_root: Path | None,
) -> tuple[StageManifest, Path]:
    """Persist a v2 ``manifest.json`` preserving ``workspace_root`` portability."""
    manifest = (
        StageManifest(
            stage=existing_manifest.stage,
            status=existing_manifest.status,
            schema_version=MANIFEST_SCHEMA_VERSION,
            config=dict(existing_manifest.config),
            steps=dict(existing_manifest.steps),
            products=dict(products),
            error=existing_manifest.error,
        )
        if existing_manifest is not None
        else StageManifest(
            stage=stage,
            status="completed",
            schema_version=MANIFEST_SCHEMA_VERSION,
            products=dict(products),
        )
    )
    written = atomic_write_json(
        manifest_path,
        manifest.to_dict(stage_dir, workspace_root=workspace_root),
    )
    return manifest, written


def load_stage_manifest_payload(
    manifest_path: Path,
    *,
    stage_label: str,
    require_completed: bool = False,
) -> tuple[StageManifest | None, Mapping[str, Any]]:
    """Read a v2 or legacy v1 stage manifest and return ``(StageManifest, payload)``."""
    base_dir = manifest_path.parent
    raw = read_json(manifest_path)
    if (
        isinstance(raw, Mapping)
        and raw.get("schema_version") == MANIFEST_SCHEMA_VERSION
    ):
        stage_manifest = StageManifest.from_dict(base_dir, raw)
        if require_completed and stage_manifest.status not in {"completed", "reused"}:
            raise RuntimeError(
                f"{stage_label} stage at {manifest_path} has status "
                f"{stage_manifest.status!r}; expected 'completed' or 'reused'."
            )
        return stage_manifest, stage_manifest.products
    payload = raw if isinstance(raw, Mapping) else {}
    if (
        require_completed
        and "status" in payload
        and payload["status"] not in {"completed", "reused"}
    ):
        raise RuntimeError(
            f"{stage_label} stage at {manifest_path} has status "
            f"{payload['status']!r}; expected 'completed' or 'reused'."
        )
    return None, payload
