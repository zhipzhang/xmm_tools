"""General pipeline configuration, context, orchestration, and ``xmm-ana`` CLI."""

from xmm_tools.config import (
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
from xmm_tools.pipeline.engine import (
    PIPELINE_STAGES,
    PipelineResult,
    normalize_stage_name,
    resolve_stages,
    run_pipeline,
)
from xmm_tools.pipeline.stage import StageSpec, all_nonempty, read_log_tail

__all__ = [
    "PIPELINE_STAGES",
    "AnaConfig",
    "BandSpec",
    "CheeseConfig",
    "EspfiltConfig",
    "InstrumentsConfig",
    "PathsConfig",
    "PipelineContext",
    "PipelineResult",
    "QPBImageConfig",
    "SpectrumConfig",
    "SpectrumRegionSpec",
    "StageSpec",
    "all_nonempty",
    "normalize_stage_name",
    "read_log_tail",
    "render_template_toml",
    "resolve_stages",
    "run_pipeline",
]
