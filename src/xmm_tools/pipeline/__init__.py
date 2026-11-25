"""General pipeline configuration, context, orchestration, and ``xmm-ana`` CLI."""

from typing import Any

from xmm_tools.pipeline.config import (
    AnaConfig,
    BandSpec,
    CheeseConfig,
    EspfiltConfig,
    InstrumentsConfig,
    PathsConfig,
    QPBImageConfig,
    SpectrumConfig,
    SpectrumRegionSpec,
    render_template_toml,
)
from xmm_tools.pipeline.context import PipelineContext
from xmm_tools.pipeline.stage import all_nonempty, read_log_tail, write_manifest_json

_ENGINE_EXPORTS = {
    "PIPELINE_STAGES",
    "PipelineResult",
    "normalize_stage_name",
    "resolve_stages",
    "run_pipeline",
}


def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        from xmm_tools.pipeline import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnaConfig",
    "BandSpec",
    "CheeseConfig",
    "EspfiltConfig",
    "InstrumentsConfig",
    "PIPELINE_STAGES",
    "PathsConfig",
    "PipelineContext",
    "PipelineResult",
    "QPBImageConfig",
    "SpectrumConfig",
    "SpectrumRegionSpec",
    "all_nonempty",
    "normalize_stage_name",
    "read_log_tail",
    "render_template_toml",
    "resolve_stages",
    "run_pipeline",
    "write_manifest_json",
]
