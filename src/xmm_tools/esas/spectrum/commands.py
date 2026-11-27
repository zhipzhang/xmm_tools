"""Pure SAS command builders and log/FITS parsers for ``xmm_tools.esas.spectrum``."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astropy.io import fits

from xmm_tools.config import SpectrumConfig
from xmm_tools.esas.naming import validate_scalar_extra_params
from xmm_tools.exec import SASCommand

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_PROTON_AREA_RE = re.compile(rf"Area\s*:\s*({_NUMBER_RE})")
_PROTON_FLUX_RE = re.compile(rf"(?<!Scaled\s)Ave Flux\s*:\s*({_NUMBER_RE})")
_PROTON_SCALED_RE = re.compile(rf"Scaled Ave Flux\s*:\s*({_NUMBER_RE})")

_SPECTRA_RESERVED_PARAMS = frozenset(
    {
        "eventfile",
        "ootevtfile",
        "regionfile",
        "withregion",
        "withsrcrem",
        "maskdet",
        "masksky",
    }
)
_BACK_RESERVED_PARAMS = frozenset({"inspecfile", "inspecoot"})


def validate_spectrum_extra_params(
    spec_cfg: SpectrumConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate ``spectra_extra_params`` and ``back_extra_params``."""
    spectra_extra = validate_scalar_extra_params(
        spec_cfg.spectra_extra_params,
        section="spectrum.spectra_extra_params",
        reserved_keys=_SPECTRA_RESERVED_PARAMS,
    )
    back_extra = validate_scalar_extra_params(
        spec_cfg.back_extra_params,
        section="spectrum.back_extra_params",
        reserved_keys=_BACK_RESERVED_PARAMS,
    )
    return spectra_extra, back_extra


def build_count_events_command(
    event_file: Path,
    expr_line: str,
    work_dir: Path,
    inst: str,
) -> SASCommand:
    """Build an ``evselect`` command to count events inside ``expr_line``."""
    bare_expr = expr_line.removeprefix("&&")
    filtered_name = f"_val_{inst}.fits"
    return SASCommand(
        program="evselect",
        cwd=work_dir,
        params={
            "table": event_file.name,
            "expression": bare_expr,
            "withfilteredset": "yes",
            "keepfilteroutput": "yes",
            "filteredset": filtered_name,
        },
        symlinks={event_file.name: event_file},
        expected_outputs=[filtered_name],
        log_file=f"validate_{inst}.log",
    )


def read_filtered_event_count(filtered_fits: Path) -> int:
    """Read row count from the ``EVENTS`` extension of a filtered FITS file."""
    with fits.open(filtered_fits) as hdul:
        return len(hdul["EVENTS"].data)


def _attach_mask_params(
    params: dict[str, Any],
    symlinks: dict[str, Path | str],
    *,
    withsrcrem: bool,
    mask_det: Path | None,
    mask_sky: Path | None,
) -> None:
    if withsrcrem and mask_det is not None and mask_sky is not None:
        symlinks[mask_det.name] = mask_det
        symlinks[mask_sky.name] = mask_sky
        params["maskdet"] = mask_det.name
        params["masksky"] = mask_sky.name


def build_mosspectra_command(
    *,
    inst: str,
    event_file: Path,
    region_file: Path,
    work_dir: Path,
    spec_cfg: SpectrumConfig,
    elow: int,
    ehigh: int,
    mask_det: Path | None,
    mask_sky: Path | None,
    extra_params: Mapping[str, Any],
) -> SASCommand:
    """Build a symlink-isolated ``mosspectra`` command."""
    ccds = spec_cfg.mos1_ccds if inst == "mos1" else spec_cfg.mos2_ccds
    symlinks: dict[str, Path | str] = {event_file.name: event_file}
    params: dict[str, Any] = {
        "eventfile": event_file.name,
        "withsrcrem": "yes" if spec_cfg.withsrcrem else "no",
        "withregion": "yes",
        "regionfile": region_file.name,
        "keepinterfiles": "yes" if spec_cfg.keepinterfiles else "no",
        "pattern": spec_cfg.pattern_mos,
        "elow": elow,
        "ehigh": ehigh,
        "ccds": ccds,
    }
    _attach_mask_params(
        params,
        symlinks,
        withsrcrem=spec_cfg.withsrcrem,
        mask_det=mask_det,
        mask_sky=mask_sky,
    )
    params.update(extra_params)
    return SASCommand(
        program="mosspectra",
        cwd=work_dir,
        params=params,
        symlinks=symlinks,
        expected_outputs=[],
        log_file=f"mosspectra_{inst}.log",
    )


def build_mosback_command(
    *,
    inst: str,
    prefix: str,
    work_dir: Path,
    spec_cfg: SpectrumConfig,
    elow: int,
    ehigh: int,
    extra_params: Mapping[str, Any],
) -> SASCommand:
    """Build a ``mosback`` command operating in ``work_dir``."""
    ccds = spec_cfg.mos1_ccds if inst == "mos1" else spec_cfg.mos2_ccds
    params: dict[str, Any] = {
        "inspecfile": f"{prefix}-fovt.pi",
        "withplotfiles": "yes" if spec_cfg.withplotfiles else "no",
        "elow": elow,
        "ehigh": ehigh,
        "ccds": ccds,
        **extra_params,
    }
    return SASCommand(
        program="mosback",
        cwd=work_dir,
        params=params,
        expected_outputs=[f"{prefix}-bkg.pi"],
        log_file=f"mosback_{inst}.log",
    )


def build_pnspectra_command(
    *,
    event_file: Path,
    oot_event_file: Path | None,
    region_file: Path,
    work_dir: Path,
    spec_cfg: SpectrumConfig,
    elow: int,
    ehigh: int,
    mask_det: Path | None,
    mask_sky: Path | None,
    extra_params: Mapping[str, Any],
) -> SASCommand:
    """Build a symlink-isolated ``pnspectra`` command."""
    symlinks: dict[str, Path | str] = {event_file.name: event_file}
    if oot_event_file is not None:
        symlinks[oot_event_file.name] = oot_event_file
    params: dict[str, Any] = {
        "eventfile": event_file.name,
        "ootevtfile": oot_event_file.name if oot_event_file is not None else "",
        "withsrcrem": "yes" if spec_cfg.withsrcrem else "no",
        "withregion": "yes",
        "regionfile": region_file.name,
        "keepinterfiles": "yes" if spec_cfg.keepinterfiles else "no",
        "pattern": spec_cfg.pattern_pn,
        "elow": elow,
        "ehigh": ehigh,
        "quads": spec_cfg.pn_quads,
    }
    _attach_mask_params(
        params,
        symlinks,
        withsrcrem=spec_cfg.withsrcrem,
        mask_det=mask_det,
        mask_sky=mask_sky,
    )
    params.update(extra_params)
    return SASCommand(
        program="pnspectra",
        cwd=work_dir,
        params=params,
        symlinks=symlinks,
        expected_outputs=[],
        log_file="pnspectra_pn.log",
    )


def build_pnback_command(
    *,
    prefix: str,
    work_dir: Path,
    spec_cfg: SpectrumConfig,
    elow: int,
    ehigh: int,
    extra_params: Mapping[str, Any],
) -> SASCommand:
    """Build a ``pnback`` command operating in ``work_dir``."""
    params: dict[str, Any] = {
        "inspecfile": f"{prefix}-fovt.pi",
        "inspecoot": f"{prefix}-fovtoot.pi",
        "withplotfiles": "yes" if spec_cfg.withplotfiles else "no",
        "elow": elow,
        "ehigh": ehigh,
        "quads": spec_cfg.pn_quads,
        **extra_params,
    }
    return SASCommand(
        program="pnback",
        cwd=work_dir,
        params=params,
        expected_outputs=[f"{prefix}-bkg.pi", f"{prefix}-fovtootsub.pi"],
        log_file="pnback_pn.log",
    )


def build_protonscale_command(
    work_dir: Path,
    prefix: str,
    inst: str,
) -> SASCommand:
    """Build a ``protonscale`` command for ``<prefix>-fovimspdet.fits`` and ``<prefix>-fovt.pi``."""
    return SASCommand(
        program="protonscale",
        cwd=work_dir,
        params={
            "mode": 1,
            "maskfile": f"{prefix}-fovimspdet.fits",
            "specfile": f"{prefix}-fovt.pi",
        },
        expected_outputs=[],
        log_file=f"protonscale_{inst}.log",
    )


def parse_protonscale_log(
    log_path: Path | None,
) -> tuple[float | None, float | None, float | None]:
    """Parse ``(Area, Ave Flux, Scaled Ave Flux)`` from a ``protonscale`` log."""
    if log_path is None or not log_path.is_file():
        return None, None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m_area = _PROTON_AREA_RE.search(text)
    m_flux = _PROTON_FLUX_RE.search(text)
    m_scaled = _PROTON_SCALED_RE.search(text)
    return (
        float(m_area.group(1)) if m_area else None,
        float(m_flux.group(1)) if m_flux else None,
        float(m_scaled.group(1)) if m_scaled else None,
    )
