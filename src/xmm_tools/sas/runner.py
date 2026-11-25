"""Subprocess runner for SAS commands with logging and output isolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Literal

from xmm_tools.sas.command import SASCommand
from xmm_tools.sas.env import SASEnv


RunStatus = Literal["completed", "skipped", "dry_run"]


class SASCommandError(RuntimeError):
    """Raised when a SAS command fails or misses expected core outputs."""


@dataclass
class RunResult:
    """Outcome of running (or skipping/dry-running) a ``SASCommand``."""

    command: SASCommand
    status: RunStatus
    returncode: int
    log_path: Path | None = None
    core_outputs: dict[str, Path] = field(default_factory=dict)
    moved_intermediate: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class SASRunner:
    """Execute ``SASCommand`` objects with clean logging and artifact isolation."""

    def __init__(
        self,
        env: SASEnv | None = None,
        *,
        dry_run: bool = False,
        skip_if_exists: bool = True,
        docker_container: str | None = None,
    ) -> None:
        self.env = env or SASEnv.from_environ()
        self.dry_run = dry_run
        self.skip_if_exists = skip_if_exists
        self.docker_container = docker_container

    def _core_outputs_ready(self, cmd: SASCommand) -> dict[str, Path] | None:
        if not cmd.core_outputs:
            return None
        cwd = cmd.cwd_path
        resolved: dict[str, Path] = {}
        for rel in cmd.core_outputs:
            p = cwd / rel
            if not p.exists() or p.stat().st_size == 0:
                return None
            resolved[rel] = p
        return resolved

    def _prepare_symlinks(self, cmd: SASCommand) -> None:
        cwd = cmd.cwd_path
        for link_name, target in cmd.symlinks.items():
            link_path = cwd / link_name
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(target)

    def _build_exec_argv(self, cmd: SASCommand) -> list[str]:
        argv = cmd.argv()
        if not self.docker_container:
            return argv
        # Run inside an existing Docker container with matching cwd and env
        docker_cmd = ["docker", "exec", "-w", str(cmd.cwd_path)]
        for k, v in self.env.to_env_dict({}).items():
            docker_cmd.extend(["-e", f"{k}={v}"])
        docker_cmd.append(self.docker_container)
        docker_cmd.extend(argv)
        return docker_cmd

    def _isolate_intermediate(
        self, cmd: SASCommand, pre_existing: set[str]
    ) -> list[Path]:
        if not cmd.intermediate_dir:
            return []

        cwd = cmd.cwd_path
        keep = {Path(p).parts[0] for p in cmd.core_outputs}
        keep.update(Path(k).parts[0] for k in cmd.symlinks)
        keep.add(cmd.log_file)
        keep.add(cmd.intermediate_dir)

        dest_dir = cwd / cmd.intermediate_dir
        moved: list[Path] = []
        for entry in sorted(cwd.iterdir()):
            if entry.name in pre_existing or entry.name in keep:
                continue
            if entry.is_dir():
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / entry.name
            if target.exists():
                target.unlink()
            shutil.move(str(entry), str(target))
            moved.append(target)
        return moved

    def run(self, cmd: SASCommand, *, force: bool = False) -> RunResult:
        """Execute ``cmd`` (or skip/dry-run according to runner settings)."""
        if self.skip_if_exists and not force:
            existing = self._core_outputs_ready(cmd)
            if existing is not None:
                return RunResult(
                    command=cmd,
                    status="skipped",
                    returncode=0,
                    log_path=cmd.cwd_path / cmd.log_file,
                    core_outputs=existing,
                )

        if self.dry_run:
            return RunResult(
                command=cmd,
                status="dry_run",
                returncode=0,
            )

        cwd = cmd.cwd_path
        cwd.mkdir(parents=True, exist_ok=True)
        self._prepare_symlinks(cmd)

        pre_existing = {p.name for p in cwd.iterdir()}
        log_path = cwd / cmd.log_file
        exec_argv = self._build_exec_argv(cmd)
        proc_env = None if self.docker_container else self.env.to_env_dict()

        t0 = time.monotonic()
        if proc_env is not None and proc_env.get("PFILES"):
            orig_pfiles = proc_env["PFILES"]
            sys_pfiles = (
                orig_pfiles.split(";", 1)[1] if ";" in orig_pfiles else orig_pfiles
            )
            with tempfile.TemporaryDirectory(prefix="xmm_pfiles_") as tmp_pfiles:
                proc_env["PFILES"] = f"{tmp_pfiles};{sys_pfiles}"
                proc = subprocess.run(
                    exec_argv,
                    cwd=cwd,
                    env=proc_env,
                    text=True,
                    capture_output=True,
                )
        else:
            proc = subprocess.run(
                exec_argv,
                cwd=None if self.docker_container else cwd,
                env=proc_env,
                text=True,
                capture_output=True,
            )
        elapsed = time.monotonic() - t0

        combined_output = ""
        if proc.stdout:
            combined_output += proc.stdout
        if proc.stderr:
            if combined_output and not combined_output.endswith("\n"):
                combined_output += "\n"
            combined_output += proc.stderr
        log_path.write_text(combined_output, encoding="utf-8")

        if proc.returncode != 0:
            tail = combined_output[-2000:]
            raise SASCommandError(
                f"{cmd.program} failed with exit code {proc.returncode}.\n"
                f"Log: {log_path}\n{tail}"
            )

        core_resolved: dict[str, Path] = {}
        missing_or_empty: list[str] = []
        for rel in cmd.core_outputs:
            p = cwd / rel
            if not p.exists() or p.stat().st_size == 0:
                missing_or_empty.append(rel)
            else:
                core_resolved[rel] = p
        if missing_or_empty:
            raise SASCommandError(
                f"{cmd.program} finished but required outputs are missing or empty: "
                f"{', '.join(missing_or_empty)}"
            )

        moved = self._isolate_intermediate(cmd, pre_existing)
        return RunResult(
            command=cmd,
            status="completed",
            returncode=0,
            log_path=log_path,
            core_outputs=core_resolved,
            moved_intermediate=moved,
            elapsed_seconds=elapsed,
        )
