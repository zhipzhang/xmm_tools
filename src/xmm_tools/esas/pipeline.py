"""End-to-end and selective stage orchestration for ``xmm-esas``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from xmm_tools.esas.config import ESASConfig, SpectrumRegionSpec
from xmm_tools.esas.flare import FlareResult, run_flare
from xmm_tools.esas.quick_image import QuickImageResult, run_quick_image
from xmm_tools.esas.source_detect import SourceMaskResult, run_source_detect
from xmm_tools.esas.spectrum import SpectrumResult, run_spectrum

PIPELINE_STAGES: tuple[str, ...] = (
    "flare",
    "cheese",
    "quick-image",
    "spectrum",
)

_STAGE_ALIASES: dict[str, str] = {
    "flare": "flare",
    "espfilt": "flare",
    "cheese": "cheese",
    "source-detect": "cheese",
    "source_detect": "cheese",
    "quick-image": "quick-image",
    "quick_image": "quick-image",
    "image": "quick-image",
    "spectrum": "spectrum",
    "spectra": "spectrum",
}


def normalize_stage_name(name: str) -> str:
    """Normalize a stage name or alias to its canonical pipeline stage key."""
    key = name.strip().lower()
    canonical = _STAGE_ALIASES.get(key)
    if canonical is None:
        valid = ", ".join(PIPELINE_STAGES)
        raise ValueError(f"Unknown ESAS stage {name!r}. Valid stages: {valid}")
    return canonical


def _parse_stage_list(items: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    if not items:
        return out
    for raw in items:
        for part in raw.split(","):
            part = part.strip()
            if part:
                out.append(normalize_stage_name(part))
    return out


def resolve_stages(
    *,
    stages: Iterable[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    skip: Iterable[str] | None = None,
) -> list[str]:
    """Resolve an ordered list of canonical config-driven ESAS stages to execute.

    Parameters
    ----------
    stages:
        Explicit stage names (comma-separated or iterable). Cannot be combined
        with ``from_stage`` or ``to_stage``.
    from_stage:
        First stage in a contiguous slice of :data:`PIPELINE_STAGES`.
    to_stage:
        Last stage (inclusive) in a contiguous slice of :data:`PIPELINE_STAGES`.
    skip:
        Stage names to exclude from the resolved list.
    """
    explicit = _parse_stage_list(stages)
    if explicit and (from_stage is not None or to_stage is not None):
        raise ValueError("Cannot combine --stages with --from or --to")

    if explicit:
        selected_set = set(explicit)
        selected = [s for s in PIPELINE_STAGES if s in selected_set]
    elif from_stage is not None or to_stage is not None:
        start_name = (
            normalize_stage_name(from_stage)
            if from_stage is not None
            else PIPELINE_STAGES[0]
        )
        end_name = (
            normalize_stage_name(to_stage)
            if to_stage is not None
            else PIPELINE_STAGES[-1]
        )
        start_idx = PIPELINE_STAGES.index(start_name)
        end_idx = PIPELINE_STAGES.index(end_name)
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid stage range: --from {start_name!r} comes after --to {end_name!r}"
            )
        selected = list(PIPELINE_STAGES[start_idx : end_idx + 1])
    else:
        selected = list(PIPELINE_STAGES)

    skip_set = set(_parse_stage_list(skip))
    resolved = [s for s in selected if s not in skip_set]
    if not resolved:
        raise ValueError("No ESAS stages remain after applying stage filters")
    return resolved


@dataclass
class PipelineResult:
    """Aggregated outputs of a multi-stage ``run_pipeline`` execution."""

    config_path: Path
    stages_requested: list[str] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: dict[str, str] = field(default_factory=dict)
    flare: FlareResult | None = None
    cheese: SourceMaskResult | None = None
    quick_image: QuickImageResult | None = None
    spectrum: SpectrumResult | None = None


def run_pipeline(
    config_path: str | Path = "esas.toml",
    *,
    stages: Iterable[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    skip: Iterable[str] | None = None,
    regions: list[SpectrumRegionSpec] | None = None,
    only: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    docker_container: str | None = None,
) -> PipelineResult:
    """Run all or selected config-driven ESAS stages in dependency order.

    Canonical stage order (after ``xmm-esas init`` has created ``esas.toml``):
    ``flare`` -> ``cheese`` -> ``quick-image`` -> ``spectrum``.

    If ``spectrum`` is included automatically (i.e. not explicitly named via
    ``stages``) and neither ``[[spectrum.regions]]`` nor ``regions`` is
    configured, ``spectrum`` is skipped cleanly so the pipeline can run
    ``flare`` -> ``cheese`` -> ``quick-image`` before regions are drawn.
    """
    explicit_stages = set(_parse_stage_list(stages))
    resolved = resolve_stages(
        stages=stages,
        from_stage=from_stage,
        to_stage=to_stage,
        skip=skip,
    )

    cfg_path = Path(config_path)
    cfg = ESASConfig.from_toml(cfg_path)

    result = PipelineResult(
        config_path=cfg_path,
        stages_requested=list(resolved),
    )

    if "flare" in resolved:
        result.flare = run_flare(
            cfg,
            dry_run=dry_run,
            force=force,
            docker_container=docker_container,
        )
        result.stages_run.append("flare")

    if "cheese" in resolved:
        result.cheese = run_source_detect(
            cfg,
            flare=result.flare,
            dry_run=dry_run,
            force=force,
            docker_container=docker_container,
        )
        result.stages_run.append("cheese")

    if "quick-image" in resolved:
        result.quick_image = run_quick_image(
            cfg,
            flare=result.flare,
            dry_run=dry_run,
            force=force,
            docker_container=docker_container,
        )
        result.stages_run.append("quick-image")

    if "spectrum" in resolved:
        target_specs = regions if regions is not None else cfg.spectrum.regions
        if not target_specs and "spectrum" not in explicit_stages:
            result.stages_skipped["spectrum"] = (
                "no [[spectrum.regions]] configured in esas.toml"
            )
        else:
            result.spectrum = run_spectrum(
                cfg,
                regions=regions,
                only=only,
                flare=result.flare,
                mask_res=result.cheese,
                dry_run=dry_run,
                force=force,
                docker_container=docker_container,
            )
            result.stages_run.append("spectrum")

    return result
