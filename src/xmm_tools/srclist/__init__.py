"""Multi-epoch XMM-Newton ``emllist.fits`` source-list merging & curation."""

from xmm_tools.srclist.io import (
    export_merged_ds9_reg,
    export_merged_emllist,
    load_cheese_epoch,
    load_emllist,
    load_sky_region_ellipses,
)
from xmm_tools.srclist.merge import (
    angular_separation_arcsec,
    filter_detections,
    merge_catalogs,
)
from xmm_tools.srclist.models import (
    EpochCatalog,
    FilterParams,
    MergedSource,
    OverlayEllipse,
    SourceDetection,
)

__all__ = [
    "EpochCatalog",
    "FilterParams",
    "MergedSource",
    "OverlayEllipse",
    "SourceDetection",
    "angular_separation_arcsec",
    "export_merged_ds9_reg",
    "export_merged_emllist",
    "filter_detections",
    "load_cheese_epoch",
    "load_emllist",
    "load_sky_region_ellipses",
    "merge_catalogs",
]
