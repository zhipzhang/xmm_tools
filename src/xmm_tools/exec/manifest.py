"""Content-addressed stage manifest (schema v2) and incremental step ledger."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from xmm_tools.exec.command import SASCommand
from xmm_tools.exec.paths import (
    FileIdentity,
    atomic_write_json,
    is_nonempty_file,
    normalize_lexical_path,
    read_json,
    resolve_workspace_path,
    serialize_portable_value,
    workspace_relative_path,
)
from xmm_tools.exec.runner import SASCommandError, SASRunner

MANIFEST_SCHEMA_VERSION = 2

StepStatus = Literal["completed", "reused", "skipped", "dry_run", "failed"]
StageStatus = Literal["running", "completed", "reused", "partial", "failed", "dry_run"]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_inputs(
    inputs: Mapping[str, Path | str] | None,
    cwd: Path | str,
) -> dict[str, Path]:
    if not inputs:
        return {}
    return {str(k): resolve_workspace_path(cwd, v) for k, v in inputs.items()}


@dataclass
class CommandStepRecord:
    """Serializable execution record for one command or derived computation step."""

    step_id: str
    status: StepStatus
    program: str
    params: dict[str, Any]
    cwd: str
    symlinks: dict[str, str] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    shell: str = ""
    log_file: str | None = None
    returncode: int | None = None
    elapsed_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    inputs: dict[str, FileIdentity] = field(default_factory=dict)
    outputs: dict[str, FileIdentity] = field(default_factory=dict)
    algorithm_version: str | None = None
    error: str | None = None
    reason: str | None = None

    def to_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize this step record with workspace-portable relative paths."""
        return {
            "step_id": self.step_id,
            "status": self.status,
            "program": self.program,
            "params": dict(self.params),
            "cwd": self.cwd,
            "symlinks": dict(self.symlinks),
            "expected_outputs": list(self.expected_outputs),
            "shell": self.shell,
            "log_file": self.log_file,
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "inputs": {
                k: ident.to_dict(base_dir, workspace_root=workspace_root)
                for k, ident in sorted(self.inputs.items())
            },
            "outputs": {
                k: ident.to_dict(base_dir, workspace_root=workspace_root)
                for k, ident in sorted(self.outputs.items())
            },
            "algorithm_version": self.algorithm_version,
            "error": self.error,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls,
        base_dir: Path | str,
        raw: Mapping[str, Any],
    ) -> CommandStepRecord:
        """Deserialize a step record, resolving relative file identities to ``base_dir``."""
        raw_inputs = raw.get("inputs", {})
        raw_outputs = raw.get("outputs", {})
        return cls(
            step_id=str(raw["step_id"]),
            status=raw["status"],
            program=str(raw["program"]),
            params=dict(raw.get("params", {})),
            cwd=str(raw.get("cwd", ".")),
            symlinks={str(k): str(v) for k, v in raw.get("symlinks", {}).items()},
            expected_outputs=[str(x) for x in raw.get("expected_outputs", [])],
            shell=str(raw.get("shell", "")),
            log_file=str(raw["log_file"]) if raw.get("log_file") is not None else None,
            returncode=(
                int(raw["returncode"]) if raw.get("returncode") is not None else None
            ),
            elapsed_seconds=float(raw.get("elapsed_seconds", 0.0)),
            started_at=(
                str(raw["started_at"]) if raw.get("started_at") is not None else None
            ),
            finished_at=(
                str(raw["finished_at"]) if raw.get("finished_at") is not None else None
            ),
            inputs={
                str(k): FileIdentity.from_dict(base_dir, v)
                for k, v in raw_inputs.items()
            },
            outputs={
                str(k): FileIdentity.from_dict(base_dir, v)
                for k, v in raw_outputs.items()
            },
            algorithm_version=(
                str(raw["algorithm_version"])
                if raw.get("algorithm_version") is not None
                else None
            ),
            error=str(raw["error"]) if raw.get("error") is not None else None,
            reason=str(raw["reason"]) if raw.get("reason") is not None else None,
        )


@dataclass
class StageManifest:
    """Versioned stage manifest recording step provenance and published products."""

    stage: str
    status: StageStatus = "running"
    schema_version: int = MANIFEST_SCHEMA_VERSION
    updated_at: str = field(default_factory=utc_now)
    config: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, CommandStepRecord] = field(default_factory=dict)
    products: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize the manifest using workspace-portable relative paths."""
        norm_base = normalize_lexical_path(base_dir)
        norm_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else None
        )
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "status": self.status,
            "updated_at": self.updated_at,
            "config": serialize_portable_value(
                self.config, norm_base, workspace_root=norm_root
            ),
            "steps": {
                sid: rec.to_dict(norm_base, workspace_root=norm_root)
                for sid, rec in self.steps.items()
            },
            "products": serialize_portable_value(
                self.products, norm_base, workspace_root=norm_root
            ),
            "error": self.error,
        }

    @classmethod
    def from_dict(
        cls,
        base_dir: Path | str,
        raw: Mapping[str, Any],
    ) -> StageManifest:
        """Deserialize a v2 stage manifest from parsed JSON."""
        raw_steps = raw.get("steps", {})
        return cls(
            stage=str(raw["stage"]),
            status=raw.get("status", "completed"),
            schema_version=int(raw.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            updated_at=str(raw.get("updated_at", "")),
            config=dict(raw.get("config", {})),
            steps={
                str(sid): CommandStepRecord.from_dict(base_dir, sraw)
                for sid, sraw in raw_steps.items()
            },
            products=dict(raw.get("products", {})),
            error=str(raw["error"]) if raw.get("error") is not None else None,
        )


class StepLedger:
    """Content-addressed incremental execution ledger backed by ``manifest.json``."""

    def __init__(
        self,
        manifest_path: Path | str,
        stage: str,
        workspace_root: Path | str | None = None,
        config: Mapping[str, Any] | None = None,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        self.manifest_path = normalize_lexical_path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.stage = stage
        self.workspace_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else None
        )
        self.config = dict(config) if config is not None else {}
        self.dry_run = dry_run
        self.force = force
        self.steps: dict[str, CommandStepRecord] = {}
        self._products: dict[str, Any] | None = None
        self.previous: StageManifest | None = self._load_previous()

    def _load_previous(self) -> StageManifest | None:
        if not self.manifest_path.is_file():
            return None
        raw = read_json(self.manifest_path)
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != MANIFEST_SCHEMA_VERSION
        ):
            return None
        prev = StageManifest.from_dict(self.base_dir, raw)
        if prev.stage != self.stage:
            raise ValueError(
                f"Manifest stage mismatch at {self.manifest_path}: "
                f"expected {self.stage!r}, got {prev.stage!r}"
            )
        return prev

    def _spec_matches_previous(
        self,
        step_id: str,
        *,
        program: str,
        params: Mapping[str, Any],
        cwd: str,
        symlinks: Mapping[str, str],
        expected_outputs: Sequence[str],
        inputs: Mapping[str, Path],
        require_outputs: bool,
        algorithm_version: str | None,
    ) -> CommandStepRecord | None:
        if self.force or self.previous is None:
            return None
        prev_step = self.previous.steps.get(step_id)
        if prev_step is None:
            return None
        if prev_step.status not in {"completed", "reused"} or prev_step.returncode != 0:
            return None
        if (
            prev_step.program != program
            or prev_step.params != dict(params)
            or prev_step.cwd != cwd
            or prev_step.symlinks != dict(symlinks)
            or prev_step.expected_outputs != list(expected_outputs)
            or prev_step.algorithm_version != algorithm_version
        ):
            return None

        if set(inputs.keys()) != set(prev_step.inputs.keys()):
            return None
        for key, cur_path in inputs.items():
            prev_ident = prev_step.inputs[key]
            if cur_path != prev_ident.path or not prev_ident.matches_disk():
                return None

        if require_outputs and not prev_step.outputs:
            return None
        for out_ident in prev_step.outputs.values():
            if not out_ident.matches_disk():
                return None

        return CommandStepRecord(
            step_id=prev_step.step_id,
            status="reused",
            program=prev_step.program,
            params=dict(prev_step.params),
            cwd=prev_step.cwd,
            symlinks=dict(prev_step.symlinks),
            expected_outputs=list(prev_step.expected_outputs),
            shell=prev_step.shell,
            log_file=prev_step.log_file,
            returncode=prev_step.returncode,
            elapsed_seconds=0.0,
            started_at=prev_step.started_at,
            finished_at=prev_step.finished_at,
            inputs=dict(prev_step.inputs),
            outputs=dict(prev_step.outputs),
            algorithm_version=prev_step.algorithm_version,
            error=None,
            reason="matched previous spec, input hashes, and output hashes",
        )

    def can_reuse(
        self,
        step_id: str,
        command: SASCommand,
        inputs: Mapping[str, Path | str] | None = None,
        *,
        require_outputs: bool = True,
        algorithm_version: str | None = None,
    ) -> CommandStepRecord | None:
        """Return a reused ``CommandStepRecord`` if ``step_id`` is valid on disk."""
        spec = command.to_spec(
            base_dir=self.base_dir, workspace_root=self.workspace_root
        )
        resolved_inputs = _resolve_inputs(inputs, command.cwd_path)
        return self._spec_matches_previous(
            step_id,
            program=spec["program"],
            params=spec["params"],
            cwd=spec["cwd"],
            symlinks=spec["symlinks"],
            expected_outputs=spec["expected_outputs"],
            inputs=resolved_inputs,
            require_outputs=require_outputs,
            algorithm_version=algorithm_version,
        )

    def _build_manifest(
        self,
        *,
        status: StageStatus,
        error: str | None = None,
    ) -> StageManifest:
        merged_steps = dict(self.previous.steps) if self.previous is not None else {}
        merged_steps.update(self.steps)
        if self._products is not None:
            products = dict(self._products)
        elif self.previous is not None:
            products = dict(self.previous.products)
        else:
            products = {}
        return StageManifest(
            stage=self.stage,
            status=status,
            schema_version=MANIFEST_SCHEMA_VERSION,
            updated_at=utc_now(),
            config=dict(self.config),
            steps=merged_steps,
            products=products,
            error=error,
        )

    def _write_checkpoint(
        self,
        *,
        status: StageStatus,
        error: str | None = None,
    ) -> StageManifest:
        manifest = self._build_manifest(status=status, error=error)
        atomic_write_json(
            self.manifest_path,
            manifest.to_dict(self.base_dir, workspace_root=self.workspace_root),
        )
        return manifest

    def fail(self, error: str) -> StageManifest:
        """Persist a failed checkpoint when stage validation fails outside ``run_step``."""
        if self.dry_run:
            return self._build_manifest(status="failed", error=error)
        return self._write_checkpoint(status="failed", error=error)

    def _validate_outputs(
        self,
        step_id: str,
        program: str,
        raw_outputs: Mapping[str, Path],
        *,
        require_outputs: bool,
        error_factory: Callable[[str], Exception],
    ) -> dict[str, FileIdentity]:
        if require_outputs and not raw_outputs:
            raise error_factory(
                f"{program} finished for step {step_id!r} but produced no outputs"
            )
        output_identities: dict[str, FileIdentity] = {}
        for role, out_path in raw_outputs.items():
            if not is_nonempty_file(out_path):
                raise error_factory(
                    f"{program} finished for step {step_id!r} but output "
                    f"{role!r} is missing or empty: {out_path}"
                )
            output_identities[str(role)] = FileIdentity.capture(out_path)
        return output_identities

    def _run_step_lifecycle(
        self,
        *,
        step_id: str,
        program: str,
        params: dict[str, Any],
        cwd: str,
        symlinks: dict[str, str],
        expected_outputs: list[str],
        resolved_inputs: dict[str, Path],
        require_outputs: bool,
        algorithm_version: str | None,
        force: bool = False,
        dry_run_meta: Callable[[], dict[str, Any]],
        execute_fn: Callable[[], tuple[Mapping[str, Path], dict[str, Any]]],
        failure_meta_fn: Callable[[Exception, float], dict[str, Any]],
    ) -> CommandStepRecord:
        reused = (
            None
            if force
            else self._spec_matches_previous(
                step_id,
                program=program,
                params=params,
                cwd=cwd,
                symlinks=symlinks,
                expected_outputs=expected_outputs,
                inputs=resolved_inputs,
                require_outputs=require_outputs,
                algorithm_version=algorithm_version,
            )
        )
        if reused is not None:
            self.steps[step_id] = reused
            if not self.dry_run:
                self._write_checkpoint(status="running")
            return reused

        if self.dry_run:
            dmeta = dry_run_meta()
            dry_inputs = {
                str(k): FileIdentity.capture(p)
                for k, p in resolved_inputs.items()
                if p.is_file()
            }
            rec = CommandStepRecord(
                step_id=step_id,
                status="dry_run",
                program=program,
                params=params,
                cwd=cwd,
                symlinks=symlinks,
                expected_outputs=expected_outputs,
                shell=dmeta.get("shell", ""),
                log_file=dmeta.get("log_file"),
                returncode=None,
                elapsed_seconds=0.0,
                started_at=dmeta.get("started_at") or utc_now(),
                finished_at=dmeta.get("finished_at") or utc_now(),
                inputs=dry_inputs,
                algorithm_version=algorithm_version,
            )
            self.steps[step_id] = rec
            return rec

        started_at = utc_now()
        input_identities: dict[str, FileIdentity] = {}
        t0 = time.monotonic()
        try:
            input_identities = {
                str(k): FileIdentity.capture(p) for k, p in resolved_inputs.items()
            }
            raw_outputs, meta = execute_fn()
            output_identities = self._validate_outputs(
                step_id,
                program,
                raw_outputs,
                require_outputs=require_outputs,
                error_factory=meta["error_factory"],
            )
            rec = CommandStepRecord(
                step_id=step_id,
                status="completed",
                program=program,
                params=params,
                cwd=cwd,
                symlinks=symlinks,
                expected_outputs=expected_outputs,
                shell=meta.get("shell", ""),
                log_file=meta.get("log_file"),
                returncode=meta.get("returncode", 0),
                elapsed_seconds=meta["elapsed_seconds"],
                started_at=meta.get("started_at") or started_at,
                finished_at=meta.get("finished_at") or utc_now(),
                inputs=input_identities,
                outputs=output_identities,
                algorithm_version=algorithm_version,
            )
            self.steps[step_id] = rec
            self._write_checkpoint(status="running")
            return rec
        except Exception as exc:
            elapsed = time.monotonic() - t0
            fmeta = failure_meta_fn(exc, elapsed)
            fail_rec = CommandStepRecord(
                step_id=step_id,
                status="failed",
                program=program,
                params=params,
                cwd=cwd,
                symlinks=symlinks,
                expected_outputs=expected_outputs,
                shell=fmeta.get("shell", ""),
                log_file=fmeta.get("log_file"),
                returncode=fmeta.get("returncode"),
                elapsed_seconds=fmeta.get("elapsed_seconds", elapsed),
                started_at=fmeta.get("started_at") or started_at,
                finished_at=fmeta.get("finished_at") or utc_now(),
                inputs=input_identities,
                algorithm_version=algorithm_version,
                error=str(exc),
            )
            self.steps[step_id] = fail_rec
            self._write_checkpoint(status="failed", error=str(exc))
            raise

    def run_step(
        self,
        step_id: str,
        runner: SASRunner,
        command: SASCommand,
        *,
        inputs: Mapping[str, Path | str] | None = None,
        collect_outputs: Callable[[Path], Mapping[str, Path]] | None = None,
        require_outputs: bool = True,
        algorithm_version: str | None = None,
        force: bool = False,
    ) -> CommandStepRecord:
        """Execute or reuse one ``SASCommand`` step and checkpoint the manifest."""
        spec = command.to_spec(
            base_dir=self.base_dir, workspace_root=self.workspace_root
        )
        resolved_inputs = _resolve_inputs(inputs, command.cwd_path)
        default_log_rel = workspace_relative_path(
            command.cwd_path / command.log_file,
            self.base_dir,
            workspace_root=self.workspace_root,
        )

        def _dry_run_meta() -> dict[str, Any]:
            dry_runner = (
                runner
                if runner.dry_run
                else SASRunner(
                    env=runner.env,
                    dry_run=True,
                    skip_if_exists=False,
                    docker_container=runner.docker_container,
                    docker_mounts=runner.docker_mounts,
                )
            )
            dry_res = dry_runner.run(command)
            return {
                "shell": dry_res.shell,
                "log_file": default_log_rel,
                "started_at": dry_res.started_at,
                "finished_at": dry_res.finished_at,
            }

        def _execute() -> tuple[Mapping[str, Path], dict[str, Any]]:
            exec_runner = (
                SASRunner(
                    env=runner.env,
                    dry_run=False,
                    skip_if_exists=False,
                    docker_container=runner.docker_container,
                    docker_mounts=runner.docker_mounts,
                )
                if runner.skip_if_exists
                else runner
            )
            res = exec_runner.run(command, force=True)
            if collect_outputs is not None:
                raw_outputs = dict(collect_outputs(command.cwd_path))
            else:
                raw_outputs = {
                    rel: command.cwd_path / rel for rel in command.expected_outputs
                }
            log_rel = (
                workspace_relative_path(
                    res.log_path, self.base_dir, workspace_root=self.workspace_root
                )
                if res.log_path is not None
                else default_log_rel
            )
            return raw_outputs, {
                "shell": res.shell,
                "log_file": log_rel,
                "returncode": res.returncode,
                "elapsed_seconds": res.elapsed_seconds,
                "started_at": res.started_at,
                "finished_at": res.finished_at,
                "error_factory": lambda msg: SASCommandError(msg, result=res),
            }

        def _failure_meta(exc: Exception, elapsed: float) -> dict[str, Any]:
            err_res = getattr(exc, "result", None)
            err_log = (
                workspace_relative_path(
                    err_res.log_path, self.base_dir, workspace_root=self.workspace_root
                )
                if err_res is not None and err_res.log_path is not None
                else default_log_rel
            )
            return {
                "shell": (
                    err_res.shell
                    if err_res is not None
                    else command.to_shell(runner.env)
                ),
                "log_file": err_log,
                "returncode": err_res.returncode if err_res is not None else None,
                "elapsed_seconds": (
                    err_res.elapsed_seconds if err_res is not None else elapsed
                ),
                "started_at": err_res.started_at if err_res is not None else None,
                "finished_at": err_res.finished_at if err_res is not None else None,
            }

        return self._run_step_lifecycle(
            step_id=step_id,
            program=spec["program"],
            params=spec["params"],
            cwd=spec["cwd"],
            symlinks=spec["symlinks"],
            expected_outputs=spec["expected_outputs"],
            resolved_inputs=resolved_inputs,
            require_outputs=require_outputs,
            algorithm_version=algorithm_version,
            force=force,
            dry_run_meta=_dry_run_meta,
            execute_fn=_execute,
            failure_meta_fn=_failure_meta,
        )

    def record_derived_step(
        self,
        step_id: str,
        *,
        description: str,
        params: Mapping[str, Any],
        inputs: Mapping[str, Path | str],
        compute: Callable[[], Mapping[str, Path]],
        algorithm_version: str = "1",
        cwd: Path | str | None = None,
        require_outputs: bool = True,
    ) -> CommandStepRecord:
        """Execute or reuse a pure-Python derived computation with SHA-256 tracking."""
        norm_cwd = normalize_lexical_path(cwd if cwd is not None else self.base_dir)
        cwd_str = workspace_relative_path(
            norm_cwd, self.base_dir, workspace_root=self.workspace_root
        )
        norm_params = serialize_portable_value(
            params,
            norm_cwd,
            workspace_root=self.workspace_root,
            sort_keys=True,
            drop_none=True,
        )
        resolved_inputs = _resolve_inputs(inputs, norm_cwd)

        def _dry_run_meta() -> dict[str, Any]:
            now = utc_now()
            return {"started_at": now, "finished_at": now}

        def _execute() -> tuple[Mapping[str, Path], dict[str, Any]]:
            t0 = time.monotonic()
            raw_outputs = dict(compute())
            return raw_outputs, {
                "returncode": 0,
                "elapsed_seconds": time.monotonic() - t0,
                "finished_at": utc_now(),
                "error_factory": RuntimeError,
            }

        def _failure_meta(exc: Exception, elapsed: float) -> dict[str, Any]:
            return {"returncode": 1, "elapsed_seconds": elapsed}

        return self._run_step_lifecycle(
            step_id=step_id,
            program=description,
            params=norm_params,
            cwd=cwd_str,
            symlinks={},
            expected_outputs=[],
            resolved_inputs=resolved_inputs,
            require_outputs=require_outputs,
            algorithm_version=algorithm_version,
            dry_run_meta=_dry_run_meta,
            execute_fn=_execute,
            failure_meta_fn=_failure_meta,
        )

    def commit(
        self,
        products: Mapping[str, Any],
        *,
        prune_unseen_steps: bool = False,
    ) -> StageManifest:
        """Finalize and atomically persist the stage manifest."""
        self._products = serialize_portable_value(
            dict(products), self.base_dir, workspace_root=self.workspace_root
        )
        if prune_unseen_steps or self.previous is None:
            merged_steps = dict(self.steps)
        else:
            merged_steps = dict(self.previous.steps)
            merged_steps.update(self.steps)

        if self.dry_run:
            status: StageStatus = "dry_run"
        elif self.steps and all(rec.status == "reused" for rec in self.steps.values()):
            status = "reused"
        else:
            status = "completed"

        manifest = StageManifest(
            stage=self.stage,
            status=status,
            schema_version=MANIFEST_SCHEMA_VERSION,
            updated_at=utc_now(),
            config=dict(self.config),
            steps=merged_steps,
            products=dict(self._products),
            error=None,
        )
        if not self.dry_run:
            atomic_write_json(
                self.manifest_path,
                manifest.to_dict(self.base_dir, workspace_root=self.workspace_root),
            )
        return manifest
