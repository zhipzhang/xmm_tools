"""Isolated ``eimagecombine`` execution and output validation for ``qpb_image``."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from xmm_tools.exec import (
    RunResult,
    SASCommand,
    SASCommandError,
    SASRunner,
    check_log_markers,
    create_relative_symlink,
    is_nonempty_file,
    read_log_tail,
)

_COMBINE_ERROR_MARKERS = ("ERROR:", "skipping file", "Error Status Returned")

_DOCKER_COMBINE_SCRATCH_SCRIPT = (
    'scratch=$(mktemp -d /tmp/xmm_eimagecombine.XXXXXX); '
    "trap 'rm -rf \"$scratch\"' EXIT; "
    'orig="$PWD"; '
    'prefix="$1"; shift; '
    'for f in "$orig"/*; do '
    '[ -f "$f" ] || continue; '
    'cp -Lf "$f" "$scratch"/ || exit $?; '
    'done; '
    'cd "$scratch" && "$@"; '
    'rc=$?; '
    'for f in "$scratch"/*; do '
    '[ -f "$f" ] || continue; '
    'base=$(basename "$f"); '
    'if [ -n "$prefix" ] && [ "${base#"$prefix"}" != "$base" ]; then '
    'cp -f "$f" "$orig/$base"; '
    'elif [ ! -e "$orig/$base" ]; then '
    'cp -f "$f" "$orig/$base"; '
    'fi; '
    'done; '
    'exit $rc'
)


@dataclass
class _DockerCombineScratchCommand(SASCommand):
    """Wrap ``eimagecombine`` inside a container-local ``/tmp`` scratch directory."""

    out_prefix: str = ""

    def argv(self) -> list[str]:
        return [
            "bash",
            "-lc",
            _DOCKER_COMBINE_SCRATCH_SCRIPT,
            "--",
            self.out_prefix,
            *super().argv(),
        ]


def discover_combined_subband_pairs(
    combine_dir: Path, prefix: str
) -> tuple[list[tuple[Path, Path]], Path | None]:
    """Locate combined ``(counts_sub, exposure)`` pairs and optional mask in ``combine_dir``."""
    if not combine_dir.is_dir():
        return [], None

    pat = re.compile(rf"^{re.escape(prefix)}ima_(\d+)_sub\.fits$")
    indexed: list[tuple[int, Path, Path]] = []
    for p in combine_dir.iterdir():
        if not is_nonempty_file(p):
            continue
        m = pat.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        exp_path = combine_dir / f"{prefix}ima_{idx}exp.fits"
        if is_nonempty_file(exp_path):
            indexed.append((idx, p, exp_path))

    indexed.sort(key=lambda item: item[0])
    pairs = [(sub_p, exp_p) for _, sub_p, exp_p in indexed]
    mask_cand = combine_dir / f"{prefix}ima_mask.fits"
    mask_path = mask_cand if is_nonempty_file(mask_cand) else None
    return pairs, mask_path


def validate_eimagecombine_outputs(
    combine_dir: Path,
    band_name: str,
    out_prefix: str,
    combine_log: str,
    expected_subbands: int,
) -> dict[str, Path]:
    """Validate ``eimagecombine`` log markers and required sub-band output files."""
    log_path = combine_dir / combine_log
    matched_marker = check_log_markers(log_path, _COMBINE_ERROR_MARKERS)
    if matched_marker is not None:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine reported an error ({matched_marker!r}) for {band_name}.\n"
            f"Log: {log_path}\n{tail}"
        )

    pairs, mask_path = discover_combined_subband_pairs(combine_dir, out_prefix)
    if not pairs:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine finished for {band_name} but no "
            f"'{out_prefix}ima_<i>_sub.fits' + 'exp.fits' pairs were "
            f"found in {combine_dir}.\nLog: {log_path}\n{tail}"
        )
    if len(pairs) != expected_subbands:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine finished for {band_name} with incomplete sub-bands: "
            f"expected {expected_subbands}, found {len(pairs)}.\n"
            f"Log: {log_path}\n{tail}"
        )

    outputs: dict[str, Path] = {}
    missing_smooth: list[str] = []
    for idx, (sub_p, exp_p) in enumerate(pairs):
        smooth_name = f"{out_prefix}ima_{idx}_subdiv_smooth.fits"
        smooth_p = combine_dir / smooth_name
        if not is_nonempty_file(smooth_p):
            missing_smooth.append(smooth_name)
        outputs[f"sub_{idx}"] = sub_p
        outputs[f"exp_{idx}"] = exp_p
        outputs[f"smooth_{idx}"] = smooth_p

    if missing_smooth:
        tail = read_log_tail(log_path)
        raise SASCommandError(
            f"eimagecombine finished for {band_name} without smoothed flux outputs: "
            f"{', '.join(missing_smooth)}.\nLog: {log_path}\n{tail}"
        )
    if mask_path is not None:
        outputs["mask"] = mask_path
    return outputs


def clean_combine_dir(
    combine_dir: Path, out_prefix: str, *, log_file: str | None = None
) -> None:
    """Remove stale ``eimagecombine`` inputs, outputs, and logs from ``combine_dir``."""
    if not combine_dir.is_dir():
        return
    for entry in combine_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            continue
        if (
            entry.name.startswith(out_prefix)
            or "_ima_" in entry.name
            or (log_file is not None and entry.name == log_file)
        ):
            entry.unlink()


class IsolatedCombineRunner:
    """Run ``eimagecombine`` inside a fresh ``/tmp`` scratch dir in local or Docker mode."""

    def __init__(
        self,
        *,
        base_runner: SASRunner,
        combine_dir: Path,
        out_prefix: str,
        input_files: Mapping[str, Path],
    ) -> None:
        self.base_runner = base_runner
        self.combine_dir = combine_dir
        self.out_prefix = out_prefix
        self.input_files = dict(input_files)
        self.env = base_runner.env
        self.dry_run = False
        self.skip_if_exists = False
        self.docker_container = base_runner.docker_container
        self.docker_mounts = base_runner.docker_mounts

    def run(self, cmd: SASCommand, *, force: bool = False) -> RunResult:
        clean_combine_dir(self.combine_dir, self.out_prefix, log_file=cmd.log_file)
        self.combine_dir.mkdir(parents=True, exist_ok=True)
        if self.docker_container is not None:
            return self._run_docker(cmd, force=force)
        return self._run_local(cmd, force=force)

    def _run_docker(self, cmd: SASCommand, *, force: bool) -> RunResult:
        for name, src_path in self.input_files.items():
            create_relative_symlink(src_path, self.combine_dir / name, overwrite_file=True)
        docker_cmd = _DockerCombineScratchCommand(
            program=cmd.program,
            cwd=self.combine_dir,
            params=dict(cmd.params),
            expected_outputs=list(cmd.expected_outputs),
            log_file=cmd.log_file,
            out_prefix=self.out_prefix,
        )
        res = self.base_runner.run(docker_cmd, force=force)
        return self._wrap_result(cmd, res)

    def _wrap_result(self, cmd: SASCommand, res: RunResult) -> RunResult:
        return RunResult(
            command=cmd,
            status=res.status,
            returncode=res.returncode,
            log_path=self.combine_dir / cmd.log_file,
            core_outputs=res.core_outputs,
            elapsed_seconds=res.elapsed_seconds,
            started_at=res.started_at,
            finished_at=res.finished_at,
            shell=res.shell,
            error=res.error,
        )

    def _run_local(self, cmd: SASCommand, *, force: bool) -> RunResult:
        with tempfile.TemporaryDirectory(prefix="xmm_eimagecombine_") as tmp_dir:
            scratch_dir = Path(tmp_dir)
            copied_inputs = set(self.input_files)
            for name, src_path in self.input_files.items():
                shutil.copy2(src_path, scratch_dir / name)

            scratch_cmd = SASCommand(
                program=cmd.program,
                cwd=scratch_dir,
                params=dict(cmd.params),
                expected_outputs=list(cmd.expected_outputs),
                log_file=cmd.log_file,
            )
            try:
                res = self.base_runner.run(scratch_cmd, force=force)
            except SASCommandError as exc:
                if exc.result is not None:
                    exc.result.log_path = self.combine_dir / cmd.log_file
                raise
            finally:
                log_src = scratch_dir / cmd.log_file
                if log_src.is_file():
                    shutil.copy2(log_src, self.combine_dir / cmd.log_file)
                for out_file in scratch_dir.iterdir():
                    if not out_file.is_file() or out_file.name == cmd.log_file:
                        continue
                    if (
                        out_file.name.startswith(self.out_prefix)
                        or out_file.name not in copied_inputs
                    ):
                        shutil.copy2(out_file, self.combine_dir / out_file.name)

        return self._wrap_result(cmd, res)
