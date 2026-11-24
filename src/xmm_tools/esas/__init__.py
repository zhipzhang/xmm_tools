"""XMM-Newton ESAS extended-source analysis configuration and workflow tools."""

from xmm_tools.esas.config import (
    BandSpec,
    ESASConfig,
    EspfiltConfig,
    ImagingConfig,
    InstrumentsConfig,
    PathsConfig,
    QuickImageConfig,
    SourceDetectConfig,
    SpectrumConfig,
    render_template_toml,
)
from xmm_tools.esas.flare import FlareResult, load_flare_result, run_flare
from xmm_tools.esas.init import InitResult, initialize_observation, standardize_event_files
from xmm_tools.esas.quick_image import QuickImageResult, resolve_subbands, run_quick_image
from xmm_tools.esas.source_detect import (
    CheeseDetector,
    SourceDetector,
    SourceMaskResult,
    load_source_mask_result,
    run_source_detect,
)

__all__ = [
    "BandSpec",
    "CheeseDetector",
    "ESASConfig",
    "EspfiltConfig",
    "FlareResult",
    "ImagingConfig",
    "InitResult",
    "InstrumentsConfig",
    "PathsConfig",
    "QuickImageConfig",
    "QuickImageResult",
    "SourceDetectConfig",
    "SourceDetector",
    "SourceMaskResult",
    "SpectrumConfig",
    "initialize_observation",
    "load_flare_result",
    "load_source_mask_result",
    "render_template_toml",
    "resolve_subbands",
    "run_flare",
    "run_quick_image",
    "run_source_detect",
    "standardize_event_files",
]
