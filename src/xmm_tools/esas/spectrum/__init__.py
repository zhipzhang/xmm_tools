"""ESAS spectrum extraction subpackage (``mosspectra``/``pnspectra`` + ``mosback``/``pnback``)."""

from xmm_tools.esas.spectrum.detector_coords import DET_SCALE, sky_region_to_esas_expr
from xmm_tools.esas.spectrum.models import (
    InstrumentProducts,
    InstrumentSpectrumProducts,
    RegionSpectrumResult,
    SpectrumResult,
    load_spectrum_result,
)
from xmm_tools.esas.spectrum.regions import ExpandedRegionJob, _import_regions
from xmm_tools.esas.spectrum.workflow import run_spectrum

__all__ = [
    "DET_SCALE",
    "ExpandedRegionJob",
    "InstrumentProducts",
    "InstrumentSpectrumProducts",
    "RegionSpectrumResult",
    "SpectrumResult",
    "_import_regions",
    "load_spectrum_result",
    "run_spectrum",
    "sky_region_to_esas_expr",
]
