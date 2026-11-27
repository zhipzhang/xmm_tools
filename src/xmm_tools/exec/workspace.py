"""Generic stage workspace, log inspection, and file-discovery helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from xmm_tools.exec.paths import (
    create_relative_symlink,
    is_nonempty_file,
    normalize_lexical_path,
    publish_artifact,
    read_log_tail,
    workspace_relative_path,
)

__all__ = [
    "StageWorkspace",
    "check_log_markers",
    "find_unique_file",
    "read_log_tail",
]


def check_log_markers(
    log_path: Path | str,
    error_markers: Sequence[str],
) -> str | None:
    """Return the first marker substring present in ``log_path``, else ``None``."""
    p = Path(log_path)
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    for marker in error_markers:
        if marker in text:
            return marker
    return None


def find_unique_file(
    directory: Path | str,
    pattern: str,
    *,
    description: str = "file",
) -> Path:
    """Locate exactly one non-empty file matching ``pattern`` in ``directory``.

    Raises ``FileNotFoundError`` if no matches exist and ``ValueError`` if more
    than one match exists (never silently picks newest or largest).
    """
    root = normalize_lexical_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Directory for {description} does not exist: {root}"
        )
    matches = sorted(p for p in root.glob(pattern) if is_nonempty_file(p))
    if not matches:
        raise FileNotFoundError(
            f"No non-empty {description} matching {pattern!r} found in {root}"
        )
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise ValueError(
            f"Multiple {description} candidates matching {pattern!r} in {root}: {names}"
        )
    return matches[0]


@dataclass
class StageWorkspace:
    """Generic layout manager for a pipeline stage's working & published files."""

    workspace_root: Path
    stage_dir: Path
    work_dirname: str = "_work"

    def __post_init__(self) -> None:
        self.workspace_root = normalize_lexical_path(self.workspace_root)
        self.stage_dir = normalize_lexical_path(self.stage_dir)

    @property
    def work_dir(self) -> Path:
        return self.stage_dir / self.work_dirname

    @property
    def manifest_path(self) -> Path:
        return self.stage_dir / "manifest.json"

    def step_dir(self, *parts: str) -> Path:
        """Return a step-specific subdirectory path under ``work_dir``."""
        return self.work_dir.joinpath(*parts)

    def ensure_dirs(self, *, dry_run: bool = False) -> None:
        """Create ``stage_dir`` and ``work_dir`` unless ``dry_run=True``."""
        if dry_run:
            return
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def link_input(
        self,
        source: Path | str,
        dest_dir: Path | str,
        link_name: str | None = None,
    ) -> Path:
        """Create a portable relative symlink for an input file inside ``dest_dir``."""
        src = normalize_lexical_path(source)
        dest = normalize_lexical_path(dest_dir)
        name = link_name if link_name is not None else src.name
        return create_relative_symlink(src, dest / name, overwrite_file=False)

    def publish(
        self,
        source: Path | str,
        target_rel_path: str | Path,
        *,
        mode: Literal["symlink", "copy"] = "symlink",
    ) -> Path:
        """Publish an output artifact from ``work_dir`` into ``stage_dir``."""
        dest = normalize_lexical_path(self.stage_dir / target_rel_path)
        return publish_artifact(source, dest, mode=mode)

    def relpath(self, path: Path | str) -> str:
        """Return a portable POSIX path relative to ``stage_dir`` / ``workspace_root``."""
        return workspace_relative_path(
            path, self.stage_dir, workspace_root=self.workspace_root
        )
