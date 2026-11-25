"""Declarative SAS command specification and shell rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
from typing import Any

from xmm_tools.exec.env import SASEnv


def format_sas_value(value: Any) -> str:
    """Format a Python parameter value into a SAS CLI argument value."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


@dataclass
class SASCommand:
    """Specification for a single SAS/ESAS CLI invocation."""

    program: str
    cwd: Path | str
    params: dict[str, Any] = field(default_factory=dict)
    core_outputs: list[str] = field(default_factory=list)
    symlinks: dict[str, Path | str] = field(default_factory=dict)
    intermediate_dir: str | None = "_intermediate"
    log_file: str = "execution.log"

    @property
    def cwd_path(self) -> Path:
        return Path(self.cwd)

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
