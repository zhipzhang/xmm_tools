"""FITS image loading, WCS coordinate helpers, smoothing, and intensity stretching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np


@dataclass
class FitsImageData:
    """Loaded 2D FITS image with optional celestial WCS."""

    path: Path | None
    data: np.ndarray  # 2D float64 (ny, nx)
    header: fits.Header
    wcs: WCS | None
    pixel_scale_arcsec: float = 1.0

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))


def load_fits_image(
    path: str | Path, hdu: int | str | None = None
) -> FitsImageData:
    """Load a 2D FITS image and its celestial WCS (if present)."""
    file_path = Path(path)
    with fits.open(file_path) as hdul:
        target_hdu = None
        if hdu is not None:
            target_hdu = hdul[hdu]
        else:
            for item in hdul:
                if item.data is not None and item.data.ndim >= 2:
                    target_hdu = item
                    break

        if target_hdu is None or target_hdu.data is None:
            raise ValueError(f"No 2D image HDU found in {file_path}")

        raw = np.asarray(target_hdu.data, dtype=np.float64)
        while raw.ndim > 2:
            raw = raw[0]
        header = target_hdu.header.copy()

    wcs: WCS | None = None
    pix_scale = 1.0
    try:
        parsed_wcs = WCS(header, naxis=2)
        if parsed_wcs.has_celestial:
            wcs = parsed_wcs
            scales_deg = proj_plane_pixel_scales(wcs)
            pix_scale = float(np.mean(scales_deg) * 3600.0)
    except Exception:  # noqa: BLE001
        wcs = None

    return FitsImageData(
        path=file_path,
        data=raw,
        header=header,
        wcs=wcs,
        pixel_scale_arcsec=pix_scale,
    )


def smooth_image(data: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Apply 2D Gaussian smoothing (ignoring NaNs) when ``sigma_pixels > 0``."""
    if sigma_pixels <= 0.0:
        return data
    kernel = Gaussian2DKernel(x_stddev=sigma_pixels)
    return np.asarray(
        convolve_fft(data, kernel, normalize_kernel=True, nan_treatment="interpolate"),
        dtype=np.float64,
    )


def stretch_image(
    data: np.ndarray,
    *,
    mode: str = "asinh",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile_low: float = 1.0,
    percentile_high: float = 99.5,
) -> tuple[np.ndarray, float, float]:
    """Compute stretched 2D float64 array and ``(vmin, vmax)`` display bounds."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float64), 0.0, 1.0

    # For XMM images with large zero-padded outside-FOV margins, compute auto-percentiles
    # over non-zero pixels when available so the source contrast stays crisp.
    nonzero = finite[finite > 0.0]
    sample = nonzero if nonzero.size >= max(16, int(finite.size * 0.05)) else finite

    lo = float(np.percentile(sample, percentile_low)) if vmin is None else float(vmin)
    hi = float(np.percentile(sample, percentile_high)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-12

    clipped = np.clip(np.where(np.isfinite(data), data, lo), lo, hi)
    norm = (clipped - lo) / (hi - lo)

    m = mode.lower()
    if m == "linear":
        stretched = norm
    elif m == "sqrt":
        stretched = np.sqrt(norm)
    elif m == "log10":
        stretched = np.log10(1.0 + 1000.0 * norm) / np.log10(1001.0)
    elif m == "asinh":
        stretched = np.arcsinh(10.0 * norm) / np.arcsinh(10.0)
    else:
        raise ValueError(f"Unsupported stretch mode: {mode!r}")

    return np.ascontiguousarray(stretched, dtype=np.float64), lo, hi


def pixel_to_sky(
    img: FitsImageData, x_pix: float, y_pix: float
) -> tuple[float, float] | None:
    """Convert 0-based pixel ``(x, y)`` to ``(ra_deg, dec_deg)``."""
    if img.wcs is None:
        return None
    sky = img.wcs.pixel_to_world(x_pix, y_pix)
    return float(sky.ra.deg), float(sky.dec.deg)


def sky_to_pixel(
    img: FitsImageData, ra_deg: float, dec_deg: float
) -> tuple[float, float] | None:
    """Convert ``(ra_deg, dec_deg)`` to 0-based pixel ``(x, y)``."""
    if img.wcs is None:
        return None
    x_arr, y_arr = img.wcs.wcs_world2pix(ra_deg, dec_deg, 0)
    return float(x_arr), float(y_arr)
