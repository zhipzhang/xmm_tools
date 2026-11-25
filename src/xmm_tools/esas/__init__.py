"""ESAS standard extended-source tasks: ``flare``, ``cheese``, ``spectrum``."""

from xmm_tools.esas.cheese import (
    CheeseDetector,
    CheeseResult,
    SourceDetector,
    load_cheese_result,
    run_cheese,
)
from xmm_tools.esas.flare import FlareResult, load_flare_result, run_flare
from xmm_tools.esas.spectrum import (
    InstrumentProducts,
    RegionSpectrumResult,
    SpectrumResult,
    load_spectrum_result,
    run_spectrum,
    sky_region_to_esas_expr,
)

__all__ = [
    "CheeseDetector",
    "CheeseResult",
    "FlareResult",
    "InstrumentProducts",
    "RegionSpectrumResult",
    "SourceDetector",
    "SpectrumResult",
    "load_cheese_result",
    "load_flare_result",
    "load_spectrum_result",
    "run_cheese",
    "run_flare",
    "run_spectrum",
    "sky_region_to_esas_expr",
]
