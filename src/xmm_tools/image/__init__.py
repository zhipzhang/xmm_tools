"""FITS image loading, smoothing, and interactive DS9 region generator."""

from xmm_tools.image.fits_io import (
    FitsImageData,
    load_fits_image,
    pixel_to_sky,
    sky_to_pixel,
    smooth_image,
    stretch_image,
)
from xmm_tools.image.regions import (
    BoxRegion,
    CircleRegion,
    EllipseRegion,
    PolygonRegion,
    RegionItem,
    StripGridParams,
    box_corners_pixels,
    ellipse_outline_pixels,
    export_ds9_regions,
    generate_strip_grid,
    parse_ds9_regions,
    polygon_pixels,
    resize_box_edge,
    strip_params_from_sky_points,
)

__all__ = [
    "BoxRegion",
    "CircleRegion",
    "EllipseRegion",
    "FitsImageData",
    "PolygonRegion",
    "RegionItem",
    "StripGridParams",
    "box_corners_pixels",
    "ellipse_outline_pixels",
    "export_ds9_regions",
    "generate_strip_grid",
    "load_fits_image",
    "parse_ds9_regions",
    "pixel_to_sky",
    "polygon_pixels",
    "resize_box_edge",
    "sky_to_pixel",
    "smooth_image",
    "stretch_image",
    "strip_params_from_sky_points",
]
