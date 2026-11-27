"""SAS ODF initialization: ``cifbuild``, ``odfingest``, ``emchain``, ``epchain``."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xmm_tools.config import render_template_toml
from xmm_tools.exec import (
    SASCommand,
    SASCommandError,
    SASEnv,
    SASRunner,
    StageManifest,
    StepLedger,
    compute_sha256,
    find_unique_file,
    is_nonempty_file,
    normalize_lexical_path,
    publish_artifact,
    read_json,
    resolve_workspace_path,
    workspace_relative_path,
)

STANDARD_EVENT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mos1", "*M1*MIEVLI*.FIT*", "mos1_evt.fits"),
    ("mos2", "*M2*MIEVLI*.FIT*", "mos2_evt.fits"),
    ("pn", "*PN*PIEVLI*.FIT*", "pn_evt.fits"),
    ("pn_oot", "*PN*OOEVLI*.FIT*", "pn_oot_evt.fits"),
)

_VALID_EXPOSURE_KEYS = frozenset(key for key, _, _ in STANDARD_EVENT_SPECS)
_SCIENTIFIC_INSTRUMENT_KEYS = ("mos1", "mos2", "pn")
_CHAIN_STEP_IDS = ("emchain", "epchain:pn", "epchain:pn_oot")
_SCHEDULED_EXPOSURE_RE = re.compile(r"S\d{3}")
_OBSID_RE = re.compile(r"\d{10}")


@dataclass(frozen=True)
class ExposureSelection:
    """Result of selecting a chain event file for one instrument."""

    instrument: str
    selected: Path | None
    candidates: tuple[Path, ...]
    policy: str  # "none" | "explicit" | "single" | "unique_scheduled"


@dataclass
class ODFInitResult:
    """Artifacts produced by ``initialize_observation``."""

    workdir: Path
    config_path: Path
    manifest_path: Path
    ccf_path: Path
    odf_sum_path: Path | None
    event_files: dict[str, Path] = field(default_factory=dict)
    manifest: StageManifest | None = None

    @classmethod
    def from_manifest(cls, manifest_path: Path | str) -> ODFInitResult:
        """Load an ``ODFInitResult`` from a v2 ``init_manifest.json``."""
        mpath = normalize_lexical_path(manifest_path)
        base_dir = mpath.parent
        raw = read_json(mpath)
        stage_manifest = StageManifest.from_dict(base_dir, raw)
        products = stage_manifest.products

        config_raw = products.get("config_path", "ana.toml")
        ccf_raw = products.get("ccf_path", "ccf.cif")
        odf_sum_raw = products.get("odf_sum_path")
        raw_events = products.get("event_files", {})
        event_files = (
            {
                str(k): resolve_workspace_path(base_dir, str(v))
                for k, v in raw_events.items()
            }
            if isinstance(raw_events, Mapping)
            else {}
        )
        return cls(
            workdir=base_dir,
            config_path=resolve_workspace_path(base_dir, str(config_raw)),
            manifest_path=mpath,
            ccf_path=resolve_workspace_path(base_dir, str(ccf_raw)),
            odf_sum_path=(
                resolve_workspace_path(base_dir, str(odf_sum_raw))
                if odf_sum_raw
                else None
            ),
            event_files=event_files,
            manifest=stage_manifest,
        )


def _validate_exposure_selection(
    exposure_selection: Mapping[str, str] | None,
) -> dict[str, str]:
    if not exposure_selection:
        return {}
    invalid = sorted(set(exposure_selection.keys()) - _VALID_EXPOSURE_KEYS)
    if invalid:
        raise ValueError(
            f"Unsupported exposure_selection key(s) {invalid}; "
            f"expected a subset of {sorted(_VALID_EXPOSURE_KEYS)}"
        )
    return {str(k): str(v) for k, v in exposure_selection.items()}


def select_chain_exposures(
    chain_dir: Path | str,
    *,
    exposure_selection: Mapping[str, str] | None = None,
) -> dict[str, ExposureSelection]:
    """Select one event file per instrument using explicit or unique-scheduled rules."""
    explicit_map = _validate_exposure_selection(exposure_selection)
    root = Path(chain_dir)
    selections: dict[str, ExposureSelection] = {}

    for key, pattern, _ in STANDARD_EVENT_SPECS:
        candidates = (
            tuple(sorted(p for p in root.glob(pattern) if is_nonempty_file(p)))
            if root.is_dir()
            else ()
        )
        if key in explicit_map:
            requested = explicit_map[key]
            req_path = Path(requested)
            matched = [
                cand
                for cand in candidates
                if cand.name == requested
                or cand == req_path
                or (
                    req_path.is_absolute()
                    and normalize_lexical_path(cand) == normalize_lexical_path(req_path)
                )
            ]
            if len(matched) != 1:
                cand_names = [c.name for c in candidates]
                raise ValueError(
                    f"Explicit exposure_selection[{key!r}]={requested!r} matched "
                    f"{len(matched)} candidate(s) in {root}; available: {cand_names}"
                )
            selections[key] = ExposureSelection(
                instrument=key,
                selected=matched[0],
                candidates=candidates,
                policy="explicit",
            )
            continue

        if not candidates:
            selections[key] = ExposureSelection(
                instrument=key,
                selected=None,
                candidates=(),
                policy="none",
            )
        elif len(candidates) == 1:
            selections[key] = ExposureSelection(
                instrument=key,
                selected=candidates[0],
                candidates=candidates,
                policy="single",
            )
        else:
            scheduled = [
                cand for cand in candidates if _SCHEDULED_EXPOSURE_RE.search(cand.stem)
            ]
            if len(scheduled) == 1:
                selections[key] = ExposureSelection(
                    instrument=key,
                    selected=scheduled[0],
                    candidates=candidates,
                    policy="unique_scheduled",
                )
            else:
                cand_names = [c.name for c in candidates]
                raise ValueError(
                    f"Ambiguous {key!r} exposures in {root} ({len(candidates)} candidates: "
                    f"{cand_names}; {len(scheduled)} scheduled). "
                    f"Pass exposure_selection={{{key!r}: '<filename>'}} to disambiguate."
                )

    return selections


def standardize_event_files(
    search_dir: Path | str,
    dest_dir: Path | str,
    *,
    copy: bool = False,
    exposure_selection: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Locate ``emchain``/``epchain`` outputs and publish standardized names."""
    selections = select_chain_exposures(
        search_dir, exposure_selection=exposure_selection
    )
    dest_root = Path(dest_dir)
    mode = "copy" if copy else "symlink"
    resolved: dict[str, Path] = {}
    for key, _, std_name in STANDARD_EVENT_SPECS:
        sel = selections[key]
        dst_path = dest_root / std_name
        if sel.selected is None:
            if dst_path.is_symlink() or dst_path.is_file():
                dst_path.unlink()
            continue
        resolved[key] = publish_artifact(sel.selected, dst_path, mode=mode)
    return resolved


def _fingerprint_directory_files(dir_path: Path) -> str | None:
    """Compute a deterministic content/metadata fingerprint for regular files in ``dir_path``."""
    if not dir_path.is_dir():
        return None
    with os.scandir(dir_path) as it:
        entries = sorted(
            (entry for entry in it if entry.is_file(follow_symlinks=True)),
            key=lambda entry: entry.name,
        )
    if not entries:
        return None

    digest = hashlib.sha256()
    for entry in entries:
        size = entry.stat(follow_symlinks=True).st_size
        digest.update(entry.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(compute_sha256(Path(entry.path)).encode("ascii"))
        digest.update(b"\n")
    return f"files:{digest.hexdigest()}"


def _resolve_local_or_mounted_dir(
    raw_path: Path,
    mounts: Sequence[tuple[Path | str, Path | str]],
) -> Path:
    """Resolve ``raw_path`` locally, falling back to reverse Docker mount lookup."""
    if raw_path.exists() or not mounts:
        return raw_path
    expanded = Path(os.path.expanduser(str(raw_path)))
    if not expanded.is_absolute():
        return raw_path
    norm = normalize_lexical_path(expanded)
    sorted_mounts = sorted(
        (
            (
                normalize_lexical_path(host_src),
                normalize_lexical_path(container_dst),
            )
            for host_src, container_dst in mounts
        ),
        key=lambda pair: (len(pair[1].parts), len(pair[1].as_posix())),
        reverse=True,
    )
    for host_dir, container_dir in sorted_mounts:
        if norm.is_relative_to(container_dir):
            rel = norm.relative_to(container_dir)
            candidate = host_dir / rel
            if candidate.exists():
                return candidate
    return raw_path


def _compute_odf_fingerprint(
    odf_path: Path,
    *,
    workspace_root: Path,
    mounts: Sequence[tuple[Path | str, Path | str]] = (),
) -> str:
    """Compute a mount-independent fingerprint for an ODF directory."""
    local_odf = _resolve_local_or_mounted_dir(odf_path, mounts)
    if local_odf.is_dir():
        summary_files = [
            p
            for p in (
                sorted(local_odf.glob("*SUM.ASC"))
                + sorted(local_odf.glob("*SUMMRY.ASC"))
            )
            if is_nonempty_file(p)
        ]
        if summary_files:
            digest = hashlib.sha256()
            for f in summary_files:
                digest.update(f.name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(compute_sha256(f).encode("ascii"))
                digest.update(b"\n")
            return f"sum:{digest.hexdigest()}"

    files_fp = _fingerprint_directory_files(local_odf)
    if files_fp is not None:
        return files_fp

    rel_path = workspace_relative_path(
        resolve_workspace_path(workspace_root, odf_path),
        workspace_root,
        workspace_root=workspace_root,
    )
    path_digest = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
    return f"path:{path_digest}"


def _compute_ccfpath_fingerprint(
    sas_ccfpath: str | Path,
    *,
    workspace_root: Path,
    mounts: Sequence[tuple[Path | str, Path | str]] = (),
) -> str:
    """Compute a portable content/path fingerprint for ``SAS_CCFPATH``."""
    segments = [seg for seg in str(sas_ccfpath).split(":") if seg]
    if not segments:
        return "empty"

    tokens: list[str] = []
    for seg in segments:
        seg_path = Path(seg)
        local_seg = _resolve_local_or_mounted_dir(seg_path, mounts)
        tok = _fingerprint_directory_files(local_seg)
        if tok is None:
            rel_seg = workspace_relative_path(
                resolve_workspace_path(workspace_root, seg_path),
                workspace_root,
                workspace_root=workspace_root,
            )
            seg_digest = hashlib.sha256(rel_seg.encode("utf-8")).hexdigest()
            tok = f"path:{seg_digest}"
        tokens.append(tok)

    digest = hashlib.sha256()
    for tok in tokens:
        digest.update(tok.encode("utf-8"))
        digest.update(b"\n")
    return f"ccf:{digest.hexdigest()}"


def discover_odf_summary(
    workdir: Path | str,
    *,
    odf_dir: Path | str | None = None,
) -> Path:
    """Locate the unique ``*SUM.SAS`` summary file and validate its ObsID if known."""
    sum_path = find_unique_file(workdir, "*SUM.SAS", description="ODF summary (*SUM.SAS)")
    if odf_dir is not None:
        obs_match = _OBSID_RE.search(Path(odf_dir).name)
        if obs_match:
            obsid = obs_match.group(0)
            if obsid not in sum_path.name:
                text = sum_path.read_text(encoding="utf-8", errors="replace")
                if obsid not in text:
                    raise ValueError(
                        f"ODF summary {sum_path.name!r} does not match ObsID {obsid!r} "
                        f"from odf_dir={str(odf_dir)!r}"
                    )
    return sum_path


def _collect_matching_events(chain_dir: Path, patterns: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not chain_dir.is_dir():
        return found
    for pattern in patterns:
        for cand in sorted(chain_dir.glob(pattern)):
            if is_nonempty_file(cand):
                found[cand.name] = cand
    return found


def _write_ana_config(
    config_path: Path,
    *,
    rendered: str,
    force: bool,
) -> None:
    if config_path.exists():
        if not force:
            return
        existing_text = config_path.read_text(encoding="utf-8", errors="replace")
        if existing_text != rendered:
            backup_path = config_path.with_suffix(".toml.bak")
            backup_path.write_text(existing_text, encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered, encoding="utf-8")


def initialize_observation(
    odf_dir: str | Path,
    workdir: str | Path = ".",
    *,
    sas_ccfpath: str | Path = "/ccf/valid",
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
    copy_events: bool = False,
    config_filename: str = "ana.toml",
    exposure_selection: Mapping[str, str] | None = None,
) -> ODFInitResult:
    """Run ``cifbuild``, ``odfingest``, ``emchain``, ``epchain`` and write ``ana.toml``."""
    explicit_selection = _validate_exposure_selection(exposure_selection)
    odf_path = Path(odf_dir).resolve()
    work_path = Path(workdir).resolve()
    chain_dir = work_path / "_chain_work"
    ccf_path = work_path / "ccf.cif"
    config_path = work_path / config_filename
    manifest_path = work_path / "init_manifest.json"

    ledger = StepLedger(
        manifest_path,
        stage="init",
        workspace_root=work_path,
        config={
            "copy_events": copy_events,
            "exposure_selection": explicit_selection,
        },
        dry_run=dry_run,
        force=force,
    )

    # 1. cifbuild (construct ccf_runner first so Docker mount mapping is available)
    ccf_runner = SASRunner(
        env=SASEnv(sas_ccfpath=str(sas_ccfpath), sas_odf=str(odf_path)),
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )
    mounts = ccf_runner._resolve_docker_mounts() if docker_container else ()
    odf_fingerprint = _compute_odf_fingerprint(
        odf_path, workspace_root=work_path, mounts=mounts
    )
    ccf_fingerprint = _compute_ccfpath_fingerprint(
        sas_ccfpath, workspace_root=work_path, mounts=mounts
    )
    init_algo_version = f"odf:{odf_fingerprint}|ccfpath:{ccf_fingerprint}"
    ccf_cmd = SASCommand(
        program="cifbuild",
        cwd=work_path,
        params={"withccfpath": False, "analysisdate": "now", "category": "XMMCCF"},
        expected_outputs=["ccf.cif"],
        log_file="cifbuild.log",
    )
    ledger.run_step(
        "cifbuild",
        ccf_runner,
        ccf_cmd,
        inputs={},
        algorithm_version=init_algo_version,
    )

    # 2. odfingest (SAS 22.1 reads $SAS_ODF automatically; keep host paths out of params)
    ingest_runner = SASRunner(
        env=SASEnv(
            sas_ccf=ccf_path,
            sas_ccfpath=str(sas_ccfpath),
            sas_odf=str(odf_path),
        ),
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )
    ingest_cmd = SASCommand(
        program="odfingest",
        cwd=work_path,
        params={"outdir": "."},
        log_file="odfingest.log",
    )
    ingest_inputs = {"ccf": ccf_path} if (not dry_run or ccf_path.is_file()) else {}
    ingest_rec = ledger.run_step(
        "odfingest",
        ingest_runner,
        ingest_cmd,
        inputs=ingest_inputs,
        collect_outputs=lambda d: {"odf_sum": discover_odf_summary(d, odf_dir=odf_path)},
        algorithm_version=init_algo_version,
    )

    odf_sum_path: Path | None
    if not dry_run:
        odf_sum_path = discover_odf_summary(work_path, odf_dir=odf_path)
    elif "odf_sum" in ingest_rec.outputs:
        odf_sum_path = ingest_rec.outputs["odf_sum"].path
    else:
        try:
            odf_sum_path = discover_odf_summary(work_path, odf_dir=odf_path)
        except FileNotFoundError:
            odf_sum_path = None

    # 3. emchain + epchain + epchain (OOT) inside _chain_work
    chain_runner = SASRunner(
        env=SASEnv(
            sas_ccf=ccf_path,
            sas_ccfpath=str(sas_ccfpath),
            sas_odf=odf_sum_path or odf_path,
        ),
        dry_run=dry_run,
        skip_if_exists=False,
        docker_container=docker_container,
    )
    chain_inputs: dict[str, Path] = {}
    if not dry_run or ccf_path.is_file():
        chain_inputs["ccf"] = ccf_path
    if odf_sum_path is not None and (not dry_run or odf_sum_path.is_file()):
        chain_inputs["odf_sum"] = odf_sum_path

    force_chains = (
        ledger.previous is not None
        and ledger.previous.status == "failed"
        and all(
            (rec := ledger.previous.steps.get(sid)) is not None and not rec.outputs
            for sid in _CHAIN_STEP_IDS
        )
    )

    ledger.run_step(
        "emchain",
        chain_runner,
        SASCommand(program="emchain", cwd=chain_dir, params={}, log_file="emchain.log"),
        inputs=chain_inputs,
        require_outputs=False,
        collect_outputs=lambda d: _collect_matching_events(
            d, ("*M1*MIEVLI*.FIT*", "*M2*MIEVLI*.FIT*")
        ),
        force=force_chains,
    )
    ledger.run_step(
        "epchain:pn",
        chain_runner,
        SASCommand(program="epchain", cwd=chain_dir, params={}, log_file="epchain.log"),
        inputs=chain_inputs,
        require_outputs=False,
        collect_outputs=lambda d: _collect_matching_events(d, ("*PN*PIEVLI*.FIT*",)),
        force=force_chains,
    )
    ledger.run_step(
        "epchain:pn_oot",
        chain_runner,
        SASCommand(
            program="epchain",
            cwd=chain_dir,
            params={"withoutoftime": True},
            log_file="epchain_oot.log",
        ),
        inputs=chain_inputs,
        require_outputs=False,
        collect_outputs=lambda d: _collect_matching_events(d, ("*PN*OOEVLI*.FIT*",)),
        force=force_chains,
    )

    std_events: dict[str, Path] = {}
    if not dry_run:
        std_events = standardize_event_files(
            chain_dir,
            work_path,
            copy=copy_events,
            exposure_selection=explicit_selection,
        )
        if not any(k in std_events for k in _SCIENTIFIC_INSTRUMENT_KEYS):
            err_msg = (
                f"Chain tasks finished in {chain_dir} but produced no scientific "
                "MOS1, MOS2, or PN event files."
            )
            ledger.fail(err_msg)
            raise SASCommandError(err_msg)

        odf_name = odf_sum_path.name if odf_sum_path is not None else "SUM.SAS"
        rendered = render_template_toml(
            workdir=".",
            sas_ccf="ccf.cif",
            sas_ccfpath=str(sas_ccfpath),
            sas_odf=odf_name,
            mos1=std_events["mos1"].name if "mos1" in std_events else "",
            mos2=std_events["mos2"].name if "mos2" in std_events else "",
            pn=std_events["pn"].name if "pn" in std_events else "",
            pn_oot=std_events["pn_oot"].name if "pn_oot" in std_events else "",
        )
        _write_ana_config(config_path, rendered=rendered, force=force)

    manifest = ledger.commit(
        products={
            "config_path": config_path,
            "ccf_path": ccf_path,
            "odf_sum_path": odf_sum_path,
            "event_files": std_events,
        }
    )

    return ODFInitResult(
        workdir=work_path,
        config_path=config_path,
        manifest_path=manifest_path,
        ccf_path=ccf_path,
        odf_sum_path=odf_sum_path,
        event_files=std_events,
        manifest=manifest,
    )
