"""XMM-Newton ESAS extended-source analysis configuration and workflow tools."""

from xmm_tools.esas.config import (
    ESASConfig,
    EspfiltConfig,
    ImagingConfig,
    InstrumentsConfig,
    PathsConfig,
    SourceDetectConfig,
    SpectrumConfig,
    render_template_toml,
)
from xmm_tools.esas.flare import FlareResult, load_flare_result, run_flare
from xmm_tools.esas.init import InitResult, initialize_observation, standardize_event_files
from xmm_tools.esas.source_detect import (
    CheeseDetector,
    SourceDetector,
    SourceMaskResult,
    load_source_mask_result,
    run_source_detect,
)

__all__ = [
    "CheeseDetector",
    "ESASConfig",
    "EspfiltConfig",
    "FlareResult",
    "ImagingConfig",
    "InitResult",
    "InstrumentsConfig",
    "PathsConfig",
    "SourceDetectConfig",
    "SourceDetector",
    "SourceMaskResult",
    "SpectrumConfig",
    "initialize_observation",
    "load_flare_result",
    "load_source_mask_result",
    "render_template_toml",
    "run_flare",
    "run_source_detect",
    "standardize_event_files",
]
