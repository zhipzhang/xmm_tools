"""Backward-compatible re-exports for standalone QPB FITS image wrappers."""

from xmm_tools.sas.qpb_image.products import (
    BandImages,
    Image2D,
    QPBProducts,
    SubbandImages,
)

__all__ = [
    "BandImages",
    "Image2D",
    "QPBProducts",
    "SubbandImages",
]
