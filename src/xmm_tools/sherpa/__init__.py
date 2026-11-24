"""Transparent OOP Sherpa wrappers for XMM-Newton ESAS multi-region/instrument fits."""

from xmm_tools.sherpa.components import ModelBuilder
from xmm_tools.sherpa.dataset import (
    PIXEL_TO_ARCMIN2,
    SpectrumCollection,
    SpectrumDataset,
    make_diagonal_rmf,
)
from xmm_tools.sherpa.plotting import plot_fit
from xmm_tools.sherpa.session import FitReport, JointFitSession

__all__ = [
    "FitReport",
    "JointFitSession",
    "ModelBuilder",
    "PIXEL_TO_ARCMIN2",
    "SpectrumCollection",
    "SpectrumDataset",
    "make_diagonal_rmf",
    "plot_fit",
]
