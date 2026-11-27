"""General pipeline stage resolution and orchestration for ``xmm-ana``."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xmm_tools.esas.cheese import CheeseResult, run_cheese
from xmm_tools.esas.flare import FlareResult, run_flare
from xmm_tools.esas.spectrum import SpectrumResult, run_spectrum
from xmm_tools.pipeline.config import AnaConfig, SpectrumRegionSpec
from xmm_tools.pipeline.context import PipelineContext
from xmm_tools.pipeline.stage import StageSpec
from xmm_tools.sas.qpb_image import QPBImageResult, run_qpb_image

_STAGE_ALIASES: dict[str, str] = {
    "flare": "flare",
    "espfilt": "flare",
    "cheese": "cheese",
    "qpb-image": "qpb-image",
    "qpb_image": "qpb-image",
    "qpb": "qpb-image",
    "spectrum": "spectrum",
    "spectra": "spectrum",
}


def _build_stage_registry(
    *,
    regions: list[SpectrumRegionSpec] | None = None,
    only: list[str] | None = None,
) -> tuple[StageSpec, ...]:
    def _skip_spectrum(ctx: PipelineContext, explicit: set[str]) -> str | None:
        target_specs = regions if regions is not None else ctx.config.spectrum.regions
        if not target_specs and "spectrum" not in explicit:
            return "no [[esas.spectrum.regions]] configured in ana.toml"
        return None

    return (
        StageSpec(
            name="flare",
            requires=(),
            run=lambda ctx: run_flare(
                ctx.config,
                dry_run=ctx.dry_run,
                force=ctx.force,
                docker_container=ctx.docker_container,
            ),
        ),
        StageSpec(
            name="cheese",
            requires=("flare",),
            run=lambda ctx: run_cheese(
                ctx.config,
                flare=ctx.artifacts.get("flare"),
                dry_run=ctx.dry_run,
                force=ctx.force,
                docker_container=ctx.docker_container,
            ),
        ),
        StageSpec(
            name="qpb-image",
            requires=("flare",),
            run=lambda ctx: run_qpb_image(
                ctx.config,
                flare=ctx.artifacts.get("flare"),
                dry_run=ctx.dry_run,
                force=ctx.force,
                docker_container=ctx.docker_container,
            ),
        ),
        StageSpec(
            name="spectrum",
            requires=("flare", "cheese"),
            run=lambda ctx: run_spectrum(
                ctx.config,
                regions=regions,
                only=only,
                flare=ctx.artifacts.get("flare"),
                mask_res=ctx.artifacts.get("cheese"),
                dry_run=ctx.dry_run,
                force=ctx.force,
                docker_container=ctx.docker_container,
            ),
            should_skip=_skip_spectrum,
        ),
    )


PIPELINE_STAGES: tuple[str, ...] = tuple(
    spec.name for spec in _build_stage_registry()
)


def normalize_stage_name(name: str) -> str:
    """Normalize a stage name or alias to its canonical pipeline stage key."""
    key = name.strip().lower()
    canonical = _STAGE_ALIASES.get(key)
    if canonical is None:
        valid = ", ".join(PIPELINE_STAGES)
        raise ValueError(f"Unknown pipeline stage {name!r}. Valid stages: {valid}")
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
    """Resolve an ordered list of canonical pipeline stages to execute."""
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
        raise ValueError("No pipeline stages remain after applying stage filters")
    return resolved


@dataclass
class PipelineResult:
    """Aggregated outputs of a multi-stage ``run_pipeline`` execution."""

    config_path: Path
    stages_requested: list[str] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def flare(self) -> FlareResult | None:
        return self.artifacts.get("flare")

    @property
    def cheese(self) -> CheeseResult | None:
        return self.artifacts.get("cheese")

    @property
    def qpb_image(self) -> QPBImageResult | None:
        return self.artifacts.get("qpb-image")

    @property
    def spectrum(self) -> SpectrumResult | None:
        return self.artifacts.get("spectrum")


def run_pipeline(
    config_path: str | Path = "ana.toml",
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
    """Run all or selected config-driven stages via ``StageSpec`` registry."""
    explicit_stages = set(_parse_stage_list(stages))
    resolved = resolve_stages(
        stages=stages,
        from_stage=from_stage,
        to_stage=to_stage,
        skip=skip,
    )

    cfg_path = Path(config_path)
    cfg = AnaConfig.from_toml(cfg_path)
    ctx = PipelineContext(
        config=cfg,
        dry_run=dry_run,
        force=force,
        docker_container=docker_container,
    )
    registry = {spec.name: spec for spec in _build_stage_registry(regions=regions, only=only)}

    result = PipelineResult(
        config_path=cfg_path,
        stages_requested=list(resolved),
    )

    for stage_name in resolved:
        spec = registry[stage_name]
        if spec.should_skip is not None:
            reason = spec.should_skip(ctx, explicit_stages)
            if reason is not None:
                result.stages_skipped[stage_name] = reason
                continue
        ctx.artifacts[stage_name] = spec.run(ctx)
        result.stages_run.append(stage_name)

    result.artifacts = dict(ctx.artifacts)
    return result
