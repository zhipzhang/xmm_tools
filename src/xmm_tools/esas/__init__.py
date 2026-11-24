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

__all__ = [
    "ESASConfig",
    "EspfiltConfig",
    "FlareResult",
    "ImagingConfig",
    "InitResult",
    "InstrumentsConfig",
    "PathsConfig",
    "SourceDetectConfig",
    "SpectrumConfig",
    "initialize_observation",
    "load_flare_result",
    "render_template_toml",
    "run_flare",
    "standardize_event_files",
]
