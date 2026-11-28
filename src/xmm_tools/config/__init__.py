"""Configuration schema, single-pass TOML loader, and template renderer."""

from xmm_tools.config.loader import load_ana_config
from xmm_tools.config.schema import (
    VALID_BKGRATE_STYLES,
    VALID_RADIUS_STYLES,
    VALID_REGION_MODES,
    AnaConfig,
    BandSpec,
    CheeseConfig,
    EspfiltConfig,
    FlareConfig,
    InstrumentsConfig,
    MaskConfig,
    PathsConfig,
    QPBImageConfig,
    SpectrumConfig,
    SpectrumRegionSpec,
)
from xmm_tools.config.template import render_template_toml

__all__ = [
    "VALID_BKGRATE_STYLES",
    "VALID_RADIUS_STYLES",
    "VALID_REGION_MODES",
    "AnaConfig",
    "BandSpec",
    "CheeseConfig",
    "EspfiltConfig",
    "FlareConfig",
    "InstrumentsConfig",
    "MaskConfig",
    "PathsConfig",
    "QPBImageConfig",
    "SpectrumConfig",
    "SpectrumRegionSpec",
    "load_ana_config",
    "render_template_toml",
]
