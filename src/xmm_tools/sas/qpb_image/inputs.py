"""Input and workspace path resolution helpers for ``xmm_tools.sas.qpb_image``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig
from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    SASRunner,
    StageWorkspace,
    StepLedger,
    is_nonempty_file,
    normalize_lexical_path,
    read_json,
    resolve_workspace_path,
)
from xmm_tools.sas.qpb_image.commands import build_atthkgen_command


def resolve_qpb_dir(cfg: AnaConfig) -> Path:
    """Resolve the canonical ``qpb_image`` stage directory for ``cfg``."""
    raw_dir = cfg.paths.qpb_image_dir
    if raw_dir == "qpb_image" and cfg.qpb_image.dir != "qpb_image":
        raw_dir = cfg.qpb_image.dir
    return normalize_lexical_path(cfg.paths.resolve(raw_dir))


def env_file_inputs(cfg: AnaConfig) -> dict[str, Path]:
    """Collect configured ``SAS_CCF`` and ``SAS_ODF`` regular files for input hashing."""
    inputs: dict[str, Path] = {}
    for key, raw in (("ccf", cfg.env.sas_ccf), ("odf_sum", cfg.env.sas_odf)):
        if raw is not None and str(raw).strip():
            p = normalize_lexical_path(cfg.paths.resolve(str(raw).strip()))
            if is_nonempty_file(p):
                inputs[key] = p
    return inputs


def collect_step_inputs(
    env_inputs: Mapping[str, Path],
    file_inputs: Mapping[str, Path | None],
    *,
    dry_run: bool,
) -> dict[str, Path]:
    """Merge ``env_inputs`` with step file inputs present (or expected in real runs)."""
    merged = dict(env_inputs)
    for key, path in file_inputs.items():
        if path is not None and (not dry_run or is_nonempty_file(path)):
            merged[key] = path
    return merged


def resolve_raw_events(
    cfg: AnaConfig, *, dry_run: bool
) -> tuple[dict[str, Path], Path | None]:
    """Resolve configured raw event files and optional PN out-of-time event file."""
    raw_events: dict[str, Path] = {}
    for inst, evt_name in (
        ("mos1", cfg.instruments.mos1),
        ("mos2", cfg.instruments.mos2),
        ("pn", cfg.instruments.pn),
    ):
        if not evt_name:
            continue
        p = normalize_lexical_path(cfg.paths.resolve(evt_name))
        if not dry_run and not is_nonempty_file(p):
            raise FileNotFoundError(
                f"Raw event file for {inst!r} not found or empty: {p}"
            )
        raw_events[inst] = p

    if not raw_events:
        raise ValueError("No active instruments configured in [instruments]")

    pn_oot: Path | None = None
    if cfg.instruments.pn and cfg.instruments.pn_oot:
        p_oot = normalize_lexical_path(cfg.paths.resolve(cfg.instruments.pn_oot))
        if not dry_run and not is_nonempty_file(p_oot):
            raise FileNotFoundError(
                f"Raw PN OOT event file not found or empty: {p_oot}"
            )
        pn_oot = p_oot

    return raw_events, pn_oot


def _load_flare_gti_map(cfg: AnaConfig, *, dry_run: bool) -> dict[str, Path]:
    flare_dir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.flare_dir))
    manifest_path = flare_dir / "manifest.json"
    if not manifest_path.is_file():
        if dry_run:
            return {}
        raise FileNotFoundError(
            f"Flare manifest not found at {manifest_path}; configure "
            "[sas.qpb_image.gti_files] in ana.toml or run 'xmm-ana flare' first."
        )

    raw = read_json(manifest_path)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Invalid flare manifest format at {manifest_path}")

    if raw.get("schema_version") == MANIFEST_SCHEMA_VERSION:
        status = raw.get("status")
        if status not in {"completed", "reused"}:
            raise RuntimeError(
                f"Flare stage at {manifest_path} has status {status!r}; "
                "expected 'completed' or 'reused' before running qpb-image."
            )
        raw_gtis = raw.get("products", {}).get("gti_files", {})
    else:
        raw_gtis = raw.get("gti_files", {})

    if not isinstance(raw_gtis, Mapping):
        return {}
    return {
        str(k): resolve_workspace_path(flare_dir, str(v))
        for k, v in raw_gtis.items()
        if v is not None and str(v).strip()
    }


def resolve_gti_files(
    cfg: AnaConfig,
    active_insts: list[str],
    *,
    gti_files: Mapping[str, Path] | None = None,
    flare: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Resolve GTI paths: config override -> explicit arg -> flare obj -> manifest."""
    flare_attr_gtis = getattr(flare, "gti_files", None) if flare is not None else None
    manifest_gtis: dict[str, Path] | None = None
    flare_dir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.flare_dir))

    resolved: dict[str, Path] = {}
    for inst in active_insts:
        override = cfg.qpb_image.gti_files.get(inst)
        if override:
            gti_path = normalize_lexical_path(cfg.paths.resolve(override))
        elif gti_files is not None and inst in gti_files:
            gti_path = normalize_lexical_path(gti_files[inst])
        elif isinstance(flare_attr_gtis, Mapping) and inst in flare_attr_gtis:
            gti_path = normalize_lexical_path(flare_attr_gtis[inst])
        else:
            if manifest_gtis is None:
                manifest_gtis = _load_flare_gti_map(cfg, dry_run=dry_run)
            cand = manifest_gtis.get(inst)
            if cand is None:
                if dry_run:
                    gti_path = flare_dir / f"{inst}-gti.fits"
                else:
                    raise FileNotFoundError(
                        f"Missing GTI file for {inst!r}: configure "
                        f"[sas.qpb_image.gti_files].{inst} in ana.toml or run "
                        "'xmm-ana flare' first."
                    )
            else:
                gti_path = cand

        if not dry_run and not is_nonempty_file(gti_path):
            raise FileNotFoundError(
                f"GTI file for {inst!r} does not exist or is empty: {gti_path}"
            )
        resolved[inst] = gti_path
    return resolved


def resolve_attfile(
    cfg: AnaConfig,
    ws: StageWorkspace,
    ledger: StepLedger,
    runner: SASRunner,
    env_inputs: Mapping[str, Path],
    *,
    dry_run: bool,
) -> Path:
    """Use configured ``attfile``, existing ``workdir/atthk.fits``, or run ``atthkgen``."""
    if cfg.qpb_image.attfile:
        att_path = normalize_lexical_path(cfg.paths.resolve(cfg.qpb_image.attfile))
        if not dry_run and not is_nonempty_file(att_path):
            raise FileNotFoundError(
                f"Configured attitude file not found or empty: {att_path}"
            )
        return att_path

    root_att = normalize_lexical_path(cfg.paths.workdir / "atthk.fits")
    if is_nonempty_file(root_att):
        return root_att

    atthk_dir = ws.step_dir("atthk")
    cmd = build_atthkgen_command(atthk_dir)
    ledger.run_step("atthkgen", runner, cmd, inputs=dict(env_inputs))
    return atthk_dir / "atthk.fits"
