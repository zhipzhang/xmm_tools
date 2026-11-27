"""Declarative SAS command specification and portable spec serialization."""

from __future__ import annotations

import os
import shlex
from dataclasses import InitVar, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from xmm_tools.exec.env import SASEnv
from xmm_tools.exec.paths import (
    _to_posix,
    normalize_lexical_path,
    serialize_portable_value,
    workspace_relative_path,
)


def format_sas_value(value: Any) -> str:
    """Format a Python parameter value into a SAS CLI argument value."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _portable_symlink_target(
    link_name: str,
    target: Path | str,
    cwd: Path,
    *,
    workspace_root: Path | None,
) -> str:
    link_parent = normalize_lexical_path(cwd / link_name).parent
    raw_target = Path(os.path.expanduser(str(target)))
    if raw_target.is_absolute():
        if workspace_root is not None:
            return workspace_relative_path(
                raw_target, link_parent, workspace_root=workspace_root
            )
        rel = os.path.relpath(normalize_lexical_path(raw_target), link_parent)
        return PurePosixPath(rel.replace(os.sep, "/")).as_posix()
    return PurePosixPath(os.path.normpath(str(raw_target)).replace(os.sep, "/")).as_posix()


@dataclass
class SASCommand:
    """Specification for a single SAS/ESAS CLI invocation."""

    program: str
    cwd: Path | str
    params: dict[str, Any] = field(default_factory=dict)
    symlinks: dict[str, Path | str] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    log_file: str = "execution.log"
    # Transitional compatibility fields used by Stages 2-3 until Stage 4 cleanup.
    core_outputs: InitVar[list[str] | None] = None
    intermediate_dir: str | None = None

    def __post_init__(self, core_outputs: list[str] | None) -> None:
        if core_outputs and not self.expected_outputs:
            self.expected_outputs = list(core_outputs)
        else:
            self.expected_outputs = list(self.expected_outputs)

    @property
    def cwd_path(self) -> Path:
        return Path(self.cwd)

    def to_spec(
        self,
        *,
        base_dir: Path | str | None = None,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Return a canonical, workspace-portable command specification.

        Excludes ``log_file``, ``docker_container``, and rendered ``shell`` so
        step reuse compares purely the scientific command identity.
        """
        norm_cwd = normalize_lexical_path(self.cwd)
        norm_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else None
        )
        if base_dir is not None:
            ref_base = normalize_lexical_path(base_dir)
            cwd_str = workspace_relative_path(
                norm_cwd, ref_base, workspace_root=norm_root
            )
        elif norm_root is not None:
            ref_base = norm_root
            cwd_str = workspace_relative_path(
                norm_cwd, norm_root, workspace_root=norm_root
            )
        else:
            ref_base = norm_cwd
            cwd_str = norm_cwd.as_posix()

        params_spec = serialize_portable_value(
            self.params,
            norm_cwd,
            workspace_root=norm_root or ref_base,
            sort_keys=True,
            drop_none=True,
        )
        symlinks_spec = {
            str(link_name): _portable_symlink_target(
                str(link_name),
                target,
                norm_cwd,
                workspace_root=norm_root,
            )
            for link_name, target in sorted(
                self.symlinks.items(), key=lambda kv: str(kv[0])
            )
        }
        return {
            "program": self.program,
            "cwd": cwd_str,
            "params": params_spec,
            "symlinks": symlinks_spec,
            "expected_outputs": sorted(_to_posix(p) for p in self.expected_outputs),
        }

    def argv(self) -> list[str]:
        """Build the ``[program, key=value, ...]`` argument vector."""
        args = [self.program]
        for key, val in self.params.items():
            if val is None:
                continue
            args.append(f"{key}={format_sas_value(val)}")
        return args

    def to_shell(self, env: SASEnv | None = None) -> str:
        """Render a reproducible bash snippet for logging or dry-run export."""
        lines: list[str] = []
        if env is not None:
            lines.extend(env.shell_exports())
        lines.append(f"cd {shlex.quote(str(self.cwd_path))}")
        for link_name, target in self.symlinks.items():
            lines.append(
                f"ln -sfn {shlex.quote(str(target))} {shlex.quote(link_name)}"
            )
        cmd_str = " ".join(shlex.quote(tok) for tok in self.argv())
        lines.append(cmd_str)
        return "\n".join(lines)


def _get_core_outputs(self: SASCommand) -> list[str]:
    return self.expected_outputs


def _set_core_outputs(self: SASCommand, value: list[str]) -> None:
    self.expected_outputs = list(value)


SASCommand.core_outputs = property(  # type: ignore[method-assign]
    _get_core_outputs,
    _set_core_outputs,
    doc="Backward-compatible alias backed directly by :attr:`expected_outputs`.",
)
