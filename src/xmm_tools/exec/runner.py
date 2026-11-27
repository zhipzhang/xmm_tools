"""Subprocess runner for SAS commands with PFILES isolation and logging."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from xmm_tools.exec.command import SASCommand
from xmm_tools.exec.env import SASEnv
from xmm_tools.exec.paths import (
    create_relative_symlink,
    is_nonempty_file,
    normalize_lexical_path,
)

RunStatus = Literal["completed", "failed", "skipped", "dry_run"]

_DOCKER_MOUNT_CACHE: dict[str, list[tuple[Path, Path]]] = {}

_DOCKER_PATH_ENV_VARS = frozenset(
    {
        "SAS_CCF",
        "SAS_CCFPATH",
        "SAS_ODF",
        "SAS_DIR",
        "CALDB",
        "XMM_REGIONS_PATH",
        "PYTHONPATH",
    }
)

_DOCKER_PFILES_SCRIPT = (
    'tmp_pf=$(mktemp -d /tmp/xmm_pfiles.XXXXXX); '
    "trap 'rm -rf \"$tmp_pf\"' EXIT; "
    'if [ -n "${PFILES:-}" ]; then '
    'export PFILES="$tmp_pf;${PFILES#*;}"; '
    'elif [ -n "${HEADAS:-}" ]; then '
    'export PFILES="$tmp_pf;$HEADAS/syspfiles"; '
    'else '
    'export PFILES="$tmp_pf"; '
    'fi; '
    'export HEADASNOQUERY="${HEADASNOQUERY:-1}" HEADASPROMPT="${HEADASPROMPT:-/dev/null}"; '
    '"$@"'
)


def _sort_docker_mounts(
    mounts: Sequence[tuple[Path | str, Path | str]],
) -> list[tuple[Path, Path]]:
    pairs = [
        (normalize_lexical_path(host), normalize_lexical_path(container))
        for host, container in mounts
    ]
    pairs.sort(
        key=lambda pair: (len(pair[0].parts), len(pair[0].as_posix())),
        reverse=True,
    )
    return pairs


def _translate_host_path(
    raw_path: Path | str,
    mounts: Sequence[tuple[Path, Path]],
) -> str:
    if not mounts:
        return str(raw_path)
    expanded = Path(os.path.expanduser(str(raw_path)))
    if not expanded.is_absolute():
        return str(raw_path)
    norm = normalize_lexical_path(expanded)
    for host_dir, container_dir in mounts:
        if norm.is_relative_to(host_dir):
            rel = norm.relative_to(host_dir)
            return (container_dir / rel).as_posix()
    return str(raw_path)


def _translate_env_value(
    value: str,
    mounts: Sequence[tuple[Path, Path]],
) -> str:
    if not mounts or not value:
        return value
    if ":" in value:
        return ":".join(
            _translate_host_path(seg, mounts) if seg else seg
            for seg in value.split(":")
        )
    return _translate_host_path(value, mounts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SASCommandError(RuntimeError):
    """Raised when a SAS command fails or misses expected outputs."""

    def __init__(self, message: str, *, result: RunResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class RunResult:
    """Outcome of executing (or skipping/dry-running) a ``SASCommand``."""

    command: SASCommand
    status: RunStatus
    returncode: int | None
    log_path: Path | None = None
    core_outputs: dict[str, Path] = field(default_factory=dict)
    moved_intermediate: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    shell: str = ""
    error: str | None = None


class SASRunner:
    """Execute ``SASCommand`` objects with PFILES isolation and clean logging."""

    def __init__(
        self,
        env: SASEnv | None = None,
        *,
        dry_run: bool = False,
        skip_if_exists: bool = False,
        docker_container: str | None = None,
        docker_mounts: Sequence[tuple[Path | str, Path | str]] | None = None,
    ) -> None:
        self.env = env or SASEnv.from_environ()
        self.dry_run = dry_run
        self.skip_if_exists = skip_if_exists
        self.docker_container = docker_container
        self.docker_mounts = list(docker_mounts) if docker_mounts is not None else None

    def _resolve_docker_mounts(self) -> list[tuple[Path, Path]]:
        if self.docker_mounts is not None:
            return _sort_docker_mounts(self.docker_mounts)
        if not self.docker_container:
            return []
        cached = _DOCKER_MOUNT_CACHE.get(self.docker_container)
        if cached is not None:
            return cached

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .Mounts}}",
                    self.docker_container,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            _DOCKER_MOUNT_CACHE[self.docker_container] = []
            return []

        if proc.returncode != 0 or not proc.stdout.strip():
            _DOCKER_MOUNT_CACHE[self.docker_container] = []
            return []

        try:
            raw_mounts = json.loads(proc.stdout)
        except ValueError:
            _DOCKER_MOUNT_CACHE[self.docker_container] = []
            return []

        raw_pairs: list[tuple[str, str]] = []
        if isinstance(raw_mounts, list):
            for item in raw_mounts:
                if not isinstance(item, dict):
                    continue
                src = item.get("Source")
                dst = item.get("Destination")
                if src and dst:
                    raw_pairs.append((str(src), str(dst)))
        mounts = _sort_docker_mounts(raw_pairs)
        _DOCKER_MOUNT_CACHE[self.docker_container] = mounts
        return mounts

    def _expected_outputs_ready(self, cmd: SASCommand) -> dict[str, Path] | None:
        outputs = cmd.expected_outputs or cmd.core_outputs
        if not outputs:
            return None
        cwd = cmd.cwd_path
        resolved: dict[str, Path] = {}
        for rel in outputs:
            p = cwd / rel
            if not is_nonempty_file(p):
                return None
            resolved[rel] = p
        return resolved

    def _prepare_symlinks(self, cmd: SASCommand) -> None:
        cwd = normalize_lexical_path(cmd.cwd_path)
        for link_name, target in cmd.symlinks.items():
            link_path = cwd / link_name
            raw_target = Path(os.path.expanduser(str(target)))
            if raw_target.is_absolute():
                source = raw_target
            else:
                source = normalize_lexical_path(link_path.parent / raw_target)
            create_relative_symlink(source, link_path, overwrite_file=False)

    def _build_exec_argv(self, cmd: SASCommand) -> list[str]:
        argv = cmd.argv()
        if not self.docker_container:
            return argv
        mounts = self._resolve_docker_mounts()
        docker_cwd = _translate_host_path(cmd.cwd_path, mounts)
        docker_cmd = ["docker", "exec", "-w", docker_cwd]
        for k, v in self.env.to_env_dict({}).items():
            val = _translate_env_value(v, mounts) if k in _DOCKER_PATH_ENV_VARS else v
            docker_cmd.extend(["-e", f"{k}={val}"])
        docker_cmd.append(self.docker_container)
        docker_cmd.extend(
            [
                "bash",
                "-lc",
                _DOCKER_PFILES_SCRIPT,
                "--",
                *argv,
            ]
        )
        return docker_cmd

    def _isolate_intermediate(
        self, cmd: SASCommand, pre_existing: set[str]
    ) -> list[Path]:
        if not cmd.intermediate_dir:
            return []

        cwd = cmd.cwd_path
        outputs = cmd.expected_outputs or cmd.core_outputs
        keep = {Path(p).parts[0] for p in outputs}
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
        shell_str = cmd.to_shell(self.env)
        now = _utc_now()

        if self.skip_if_exists and not force:
            existing = self._expected_outputs_ready(cmd)
            if existing is not None:
                return RunResult(
                    command=cmd,
                    status="skipped",
                    returncode=0,
                    log_path=cmd.cwd_path / cmd.log_file,
                    core_outputs=existing,
                    started_at=now,
                    finished_at=now,
                    shell=shell_str,
                )

        if self.dry_run:
            return RunResult(
                command=cmd,
                status="dry_run",
                returncode=0,
                started_at=now,
                finished_at=now,
                shell=shell_str,
            )

        cwd = cmd.cwd_path
        cwd.mkdir(parents=True, exist_ok=True)
        self._prepare_symlinks(cmd)

        pre_existing = {p.name for p in cwd.iterdir()}
        log_path = cwd / cmd.log_file
        exec_argv = self._build_exec_argv(cmd)
        proc_env = None if self.docker_container else self.env.to_env_dict()

        started_at = _utc_now()
        t0 = time.monotonic()
        with contextlib.ExitStack() as stack:
            if proc_env is not None:
                tmp_pfiles = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="xmm_pfiles_")
                )
                orig_pfiles = proc_env.get("PFILES")
                if orig_pfiles:
                    sys_pfiles = (
                        orig_pfiles.split(";", 1)[1]
                        if ";" in orig_pfiles
                        else orig_pfiles
                    )
                    proc_env["PFILES"] = f"{tmp_pfiles};{sys_pfiles}"
                elif (
                    proc_env.get("HEADAS")
                    and (Path(proc_env["HEADAS"]) / "syspfiles").is_dir()
                ):
                    proc_env["PFILES"] = f"{tmp_pfiles};{proc_env['HEADAS']}/syspfiles"
                else:
                    proc_env["PFILES"] = tmp_pfiles

            try:
                proc = subprocess.run(
                    exec_argv,
                    cwd=None if self.docker_container else cwd,
                    env=proc_env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except OSError as exc:
                elapsed = time.monotonic() - t0
                finished_at = _utc_now()
                err_msg = f"Failed to execute {cmd.program}: {exc}\nLog: {log_path}"
                log_path.write_text(f"{err_msg}\n", encoding="utf-8")
                fail_res = RunResult(
                    command=cmd,
                    status="failed",
                    returncode=None,
                    log_path=log_path,
                    elapsed_seconds=elapsed,
                    started_at=started_at,
                    finished_at=finished_at,
                    shell=shell_str,
                    error=err_msg,
                )
                raise SASCommandError(err_msg, result=fail_res) from exc

        elapsed = time.monotonic() - t0
        finished_at = _utc_now()

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
            err_msg = (
                f"{cmd.program} failed with exit code {proc.returncode}.\n"
                f"Log: {log_path}\n{tail}"
            )
            fail_res = RunResult(
                command=cmd,
                status="failed",
                returncode=proc.returncode,
                log_path=log_path,
                elapsed_seconds=elapsed,
                started_at=started_at,
                finished_at=finished_at,
                shell=shell_str,
                error=err_msg,
            )
            raise SASCommandError(err_msg, result=fail_res)

        expected = cmd.expected_outputs or cmd.core_outputs
        resolved_outputs: dict[str, Path] = {}
        missing_or_empty: list[str] = []
        for rel in expected:
            p = cwd / rel
            if not is_nonempty_file(p):
                missing_or_empty.append(rel)
            else:
                resolved_outputs[rel] = p
        if missing_or_empty:
            err_msg = (
                f"{cmd.program} finished but required outputs are missing or empty: "
                f"{', '.join(missing_or_empty)}"
            )
            fail_res = RunResult(
                command=cmd,
                status="failed",
                returncode=proc.returncode,
                log_path=log_path,
                elapsed_seconds=elapsed,
                started_at=started_at,
                finished_at=finished_at,
                shell=shell_str,
                error=err_msg,
            )
            raise SASCommandError(err_msg, result=fail_res)

        moved = self._isolate_intermediate(cmd, pre_existing)
        return RunResult(
            command=cmd,
            status="completed",
            returncode=0,
            log_path=log_path,
            core_outputs=resolved_outputs,
            moved_intermediate=moved,
            elapsed_seconds=elapsed,
            started_at=started_at,
            finished_at=finished_at,
            shell=shell_str,
        )
