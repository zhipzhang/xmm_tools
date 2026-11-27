"""Symlink-first portable SAS command builders for ``xmm_tools.sas.qpb_image``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xmm_tools.config import BandSpec, QPBImageConfig
from xmm_tools.exec import SASCommand

_EVQPB_RESERVED_PARAMS = frozenset({"table", "attfile", "outset"})
_EIMAGEGET_RESERVED_PARAMS = frozenset(
    {"evtfile", "fwcfile", "gtifile", "attfile", "ootfile", "pimin", "pimax"}
)
_EIMAGECOMBINE_RESERVED_PARAMS = frozenset({"prefix", "withasmooth"})


def _contains_path_value(value: Any) -> bool:
    if isinstance(value, Path):
        return True
    if isinstance(value, str):
        return "/" in value or "\\" in value
    if isinstance(value, (list, tuple)):
        return any(_contains_path_value(item) for item in value)
    return False


def _validate_extra_params(
    extra_params: Mapping[str, Any],
    *,
    param_group: str,
    reserved_keys: frozenset[str],
) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for raw_key, val in extra_params.items():
        key = str(raw_key)
        if key.lower() in reserved_keys:
            raise ValueError(
                f"Reserved structural parameter {key!r} is not allowed in "
                f"{param_group}; reserved keys: {sorted(reserved_keys)}"
            )
        if _contains_path_value(val):
            raise ValueError(
                f"Path-valued parameter {key!r}={val!r} is not allowed in "
                f"{param_group}; extra_params are restricted to scalar SAS options "
                "to prevent untracked path leakage."
            )
        validated[key] = val
    return validated


def validate_qpb_extra_params(config: QPBImageConfig) -> None:
    """Validate ``evqpb``, ``eimageget``, and ``eimagecombine`` extra_params."""
    _validate_extra_params(
        config.evqpb_extra_params,
        param_group="evqpb_extra_params",
        reserved_keys=_EVQPB_RESERVED_PARAMS,
    )
    _validate_extra_params(
        config.eimageget_extra_params,
        param_group="eimageget_extra_params",
        reserved_keys=_EIMAGEGET_RESERVED_PARAMS,
    )
    _validate_extra_params(
        config.eimagecombine_extra_params,
        param_group="eimagecombine_extra_params",
        reserved_keys=_EIMAGECOMBINE_RESERVED_PARAMS,
    )


def resolve_subbands(
    band: BandSpec, default_edges: Sequence[int]
) -> list[tuple[int, int]]:
    """Return narrow sub-bands for ``band``, splitting along ``default_edges`` if needed."""
    if band.subbands:
        return [(int(lo), int(hi)) for lo, hi in band.subbands]

    inner = sorted(e for e in default_edges if band.emin < e < band.emax)
    pts = [band.emin, *inner, band.emax]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def build_atthkgen_command(step_dir: Path) -> SASCommand:
    """Build a portable ``atthkgen`` command writing ``atthk.fits`` in ``step_dir``."""
    return SASCommand(
        program="atthkgen",
        cwd=step_dir,
        params={"atthkset": "atthk.fits", "timestep": 1.0},
        expected_outputs=["atthk.fits"],
        log_file="atthkgen.log",
    )


def build_evqpb_command(
    *,
    instrument: str,
    raw_event: Path,
    attfile: Path,
    step_dir: Path,
    config: QPBImageConfig,
) -> SASCommand:
    """Build a symlink-isolated ``evqpb`` command with local filenames in ``params``."""
    extra = _validate_extra_params(
        config.evqpb_extra_params,
        param_group="evqpb_extra_params",
        reserved_keys=_EVQPB_RESERVED_PARAMS,
    )
    evt_link = f"{instrument}_evt.fits"
    att_link = "atthk.fits"
    out_name = f"{instrument}_qpb.fits"
    params: dict[str, Any] = {
        "table": evt_link,
        "attfile": att_link,
        "outset": out_name,
        **extra,
    }
    return SASCommand(
        program="evqpb",
        cwd=step_dir,
        params=params,
        symlinks={evt_link: raw_event, att_link: attfile},
        expected_outputs=[out_name],
        log_file=f"{instrument}_evqpb.log",
    )


def build_eimageget_command(
    *,
    instrument: str,
    raw_event: Path,
    qpb_event: Path,
    gti_file: Path,
    attfile: Path,
    oot_event: Path | None,
    subbands: Sequence[tuple[int, int]],
    step_dir: Path,
    config: QPBImageConfig,
) -> SASCommand:
    """Build a symlink-isolated ``eimageget`` command in its own ``step_dir``."""
    extra = _validate_extra_params(
        config.eimageget_extra_params,
        param_group="eimageget_extra_params",
        reserved_keys=_EIMAGEGET_RESERVED_PARAMS,
    )
    evt_link = f"{instrument}_evt.fits"
    fwc_link = f"{instrument}_qpb.fits"
    gti_link = f"{instrument}_gti.fits"
    att_link = "atthk.fits"

    symlinks: dict[str, Path | str] = {
        evt_link: raw_event,
        fwc_link: qpb_event,
        gti_link: gti_file,
        att_link: attfile,
    }
    params: dict[str, Any] = {
        "evtfile": evt_link,
        "fwcfile": fwc_link,
        "gtifile": gti_link,
        "attfile": att_link,
        "pimin": [lo for lo, _ in subbands],
        "pimax": [hi for _, hi in subbands],
        "ximagebinsize": config.bin_size,
        "yimagebinsize": config.bin_size,
        "withexposure": True,
        "withfwcimages": True,
        "withmask": True,
    }
    if instrument == "pn" and oot_event is not None:
        oot_link = "pn_oot_evt.fits"
        symlinks[oot_link] = oot_event
        params["ootfile"] = oot_link

    params.update(extra)
    return SASCommand(
        program="eimageget",
        cwd=step_dir,
        params=params,
        symlinks=symlinks,
        expected_outputs=[],
        log_file=f"{instrument}_eimageget.log",
    )


def build_eimagecombine_command(
    *,
    band_name: str,
    band: BandSpec,
    subbands: Sequence[tuple[int, int]],
    combine_dir: Path,
    config: QPBImageConfig,
) -> SASCommand:
    """Build a portable ``eimagecombine`` command for one target energy band."""
    extra = _validate_extra_params(
        config.eimagecombine_extra_params,
        param_group="eimagecombine_extra_params",
        reserved_keys=_EIMAGECOMBINE_RESERVED_PARAMS,
    )
    out_prefix = f"{band_name}_{band.emin}_{band.emax}_"
    n_sub = len(subbands)
    params: dict[str, Any] = {
        "prefix": out_prefix,
        "withcheckinput": True,
        "withaddimages": True,
        "withcombineimages": True,
        "withasmooth": True,
        "keepinterstage": True,
        "ecut": config.ecut,
        "epn_weight": [config.epn_weight] * n_sub,
        "em1_weight": [config.em1_weight] * n_sub,
        "em2_weight": [config.em2_weight] * n_sub,
        "smoothstyle": config.smoothstyle,
        "convolverstyle": config.convolverstyle,
        "width": config.width,
        "minwidth": config.minwidth,
        "maxwidth": config.maxwidth,
        "desiredsnr": config.desiredsnr,
        "nconvolvers": config.nconvolvers,
        **extra,
    }
    return SASCommand(
        program="eimagecombine",
        cwd=combine_dir,
        params=params,
        expected_outputs=[],
        log_file=f"{band_name}_combine.log",
    )
