"""ESAS standard extended-source tasks: ``flare``, ``cheese``, ``spectrum``."""

from xmm_tools.esas.cheese import (
    CheeseDetector,
    CheeseResult,
    SourceDetector,
    load_cheese_result,
    run_cheese,
)
from xmm_tools.esas.flare import FlareResult, load_flare_result, run_flare
from xmm_tools.esas.naming import (
    ALLEVC_SUFFIX,
    ALLEVCOOT_SUFFIX,
    FOVT_SUFFIX,
    discover_esas_prefix,
    env_calibration_inputs,
    strip_esas_suffix,
    validate_scalar_extra_params,
)
from xmm_tools.esas.spectrum import (
    DET_SCALE,
    ExpandedRegionJob,
    InstrumentProducts,
    InstrumentSpectrumProducts,
    RegionSpectrumResult,
    SpectrumResult,
    load_spectrum_result,
    run_spectrum,
    sky_region_to_esas_expr,
)

__all__ = [
    "ALLEVCOOT_SUFFIX",
    "ALLEVC_SUFFIX",
    "DET_SCALE",
    "FOVT_SUFFIX",
    "CheeseDetector",
    "CheeseResult",
    "ExpandedRegionJob",
    "FlareResult",
    "InstrumentProducts",
    "InstrumentSpectrumProducts",
    "RegionSpectrumResult",
    "SourceDetector",
    "SpectrumResult",
    "discover_esas_prefix",
    "env_calibration_inputs",
    "load_cheese_result",
    "load_flare_result",
    "load_spectrum_result",
    "run_cheese",
    "run_flare",
    "run_spectrum",
    "sky_region_to_esas_expr",
    "strip_esas_suffix",
    "validate_scalar_extra_params",
]
