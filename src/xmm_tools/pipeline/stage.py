"""Shared stage utilities, ``StageManifest`` base class, and ``StageSpec`` contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from xmm_tools.pipeline.context import PipelineContext


def all_nonempty(paths: Mapping[str, Path]) -> bool:
    """Return True when ``paths`` is non-empty and every file exists with size > 0."""
    return bool(paths) and all(
        p.exists() and p.stat().st_size > 0 for p in paths.values()
    )


def read_log_tail(log_path: Path, limit: int = 1500) -> str:
    """Return the last ``limit`` characters of a SAS execution log if present."""
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")[-limit:]


def write_manifest_json(manifest_path: Path, payload: dict[str, Any]) -> Path:
    """Write a stage manifest JSON file with deterministic formatting."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_manifest_json(manifest_path: Path) -> dict[str, Any]:
    """Load a stage manifest JSON dictionary."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def relativize_manifest_paths(mapping: Mapping[str, Path]) -> dict[str, str]:
    """Convert ``{key: Path}`` to ``{key: filename}`` for manifest serialization."""
    return {k: p.name for k, p in mapping.items()}


def resolve_manifest_paths(
    base_dir: Path, raw: Mapping[str, Any], section: str
) -> dict[str, Path]:
    """Resolve ``{key: rel_path}`` under ``raw[section]`` against ``base_dir``."""
    entries = raw.get(section, {})
    return {k: base_dir / v for k, v in entries.items()}


class StageManifest:
    """Base mixin providing unified ``save_manifest`` for stage result classes."""

    manifest_path: Path

    def to_manifest_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def save_manifest(self) -> Path:
        return write_manifest_json(self.manifest_path, self.to_manifest_dict())


@dataclass(frozen=True)
class StageSpec:
    """Declarative pipeline stage specification with explicit upstream ``requires``."""

    name: str
    requires: tuple[str, ...]
    run: Callable[[PipelineContext], Any]
    should_skip: Callable[[PipelineContext, set[str]], str | None] | None = None
