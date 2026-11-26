"""Standalone FITS wrapper & reader for ``qpb_image`` outputs.

Decoupled from the GUI ``xmm_tools.image`` package: provides lightweight 2D
image wrappers (``Image2D``) and band/sub-band accessors (``SubbandImages``,
``BandImages``, ``QPBProducts``) with built-in counts, exposure, mask, and
flux combination helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
import numpy as np

from xmm_tools.sas.qpb_image import BandCombinedProducts, QPBImageResult


@dataclass
class Image2D:
    """Lightweight 2D FITS image wrapper with header preservation and FITS export."""

    data: np.ndarray
    header: fits.Header
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @classmethod
    def from_fits(cls, path: str | Path, hdu: int | str = 0) -> Image2D:
        """Load a 2D FITS image into an ``Image2D`` wrapper."""
        file_path = Path(path)
        with fits.open(file_path) as hdul:
            target = hdul[hdu]
            if target.data is None:
                raise ValueError(f"HDU {hdu!r} has no image data in {file_path}")
            arr = np.asarray(target.data, dtype=np.float64)
            while arr.ndim > 2:
                arr = arr[0]
            hdr = target.header.copy()
        return cls(data=arr, header=hdr, path=file_path)

    def copy(self) -> Image2D:
        return Image2D(
            data=self.data.copy(),
            header=self.header.copy(),
            path=self.path,
        )

    def masked(
        self, mask: np.ndarray | Image2D, *, fill_value: float = np.nan
    ) -> Image2D:
        """Return a copy where pixels outside ``mask > 0`` are set to ``fill_value``."""
        m_arr = mask.data if isinstance(mask, Image2D) else np.asarray(mask)
        valid = np.isfinite(m_arr) & (m_arr > 0)
        out = np.where(valid, self.data, fill_value).astype(np.float64)
        hdr = self.header.copy()
        hdr.add_history("Applied mask in Image2D.masked")
        return Image2D(data=out, header=hdr, path=None)

    def writeto(
        self,
        out_path: str | Path,
        *,
        overwrite: bool = True,
        dtype: type = np.float32,
    ) -> Path:
        """Write this ``Image2D`` to a FITS file."""
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(
            data=self.data.astype(dtype), header=self.header
        ).writeto(dest, overwrite=overwrite)
        return dest


@dataclass
class SubbandImages:
    """Loaded ``eimagecombine`` products for a single energy bin."""

    index: int
    emin: int
    emax: int
    counts: Image2D
    exposure: Image2D
    smoothed_flux: Image2D | None = None

    def rate(
        self,
        *,
        mask: np.ndarray | Image2D | None = None,
        min_exposure: float = 0.0,
        fill_value: float = np.nan,
    ) -> Image2D:
        """Compute ``counts / exposure`` for this sub-band as an ``Image2D``."""
        c = self.counts.data
        e = self.exposure.data
        valid = np.isfinite(c) & np.isfinite(e) & (e > min_exposure)
        if mask is not None:
            m_arr = mask.data if isinstance(mask, Image2D) else np.asarray(mask)
            valid &= np.isfinite(m_arr) & (m_arr > 0)
        out = np.full(c.shape, fill_value, dtype=np.float64)
        out[valid] = c[valid] / e[valid]

        hdr = self.counts.header.copy()
        hdr["BUNIT"] = ("counts/s", f"Exposure-corrected rate ({self.emin}-{self.emax} eV)")
        hdr.add_history(f"Computed counts/exposure for {self.emin}-{self.emax} eV")
        return Image2D(data=out, header=hdr, path=None)


@dataclass
class BandImages:
    """Loaded products and combination wrappers for one target energy band."""

    band_name: str
    emin: int
    emax: int
    subbands: list[SubbandImages] = field(default_factory=list)
    mask: Image2D | None = None
    flux: Image2D | None = None

    @property
    def mask_bool(self) -> np.ndarray | None:
        if self.mask is None:
            return None
        m = self.mask.data
        return np.isfinite(m) & (m > 0)

    @property
    def has_smoothed_flux(self) -> bool:
        return bool(self.subbands) and all(
            sb.smoothed_flux is not None for sb in self.subbands
        )

    def _select_subbands(self, indices: Sequence[int] | None) -> list[SubbandImages]:
        if not self.subbands:
            raise ValueError(f"Band {self.band_name!r} has no sub-bands loaded")
        if indices is None:
            return list(self.subbands)
        by_idx = {sb.index: sb for sb in self.subbands}
        return [by_idx[i] for i in indices]

    def sum_counts(
        self,
        *,
        subband_indices: Sequence[int] | None = None,
        apply_mask: bool = False,
        fill_value: float = 0.0,
    ) -> Image2D:
        """Sum net counts maps across all (or selected) sub-bands."""
        selected = self._select_subbands(subband_indices)
        total = np.zeros_like(selected[0].counts.data, dtype=np.float64)
        for sb in selected:
            c = sb.counts.data
            valid = np.isfinite(c)
            total[valid] += c[valid]
        if apply_mask and self.mask_bool is not None:
            total[~self.mask_bool] = fill_value

        hdr = selected[0].counts.header.copy()
        hdr["BUNIT"] = ("counts", f"Summed net counts ({self.emin}-{self.emax} eV)")
        hdr.add_history(
            "Summed counts across sub-bands: "
            + ", ".join(f"{sb.emin}-{sb.emax}eV" for sb in selected)
        )
        return Image2D(data=total, header=hdr, path=None)

    def sum_exposure(
        self,
        *,
        subband_indices: Sequence[int] | None = None,
        mode: str = "mean",
        apply_mask: bool = False,
        fill_value: float = 0.0,
    ) -> Image2D:
        """Combine exposure maps across sub-bands (``mode='mean'`` or ``'sum'``)."""
        if mode not in ("mean", "sum"):
            raise ValueError(f"Unsupported exposure combine mode: {mode!r}")
        selected = self._select_subbands(subband_indices)
        total = np.zeros_like(selected[0].exposure.data, dtype=np.float64)
        for sb in selected:
            e = sb.exposure.data
            valid = np.isfinite(e)
            total[valid] += e[valid]
        if mode == "mean":
            total /= float(len(selected))
        if apply_mask and self.mask_bool is not None:
            total[~self.mask_bool] = fill_value

        hdr = selected[0].exposure.header.copy()
        hdr["BUNIT"] = ("s", f"Combined exposure ({mode}, {self.emin}-{self.emax} eV)")
        hdr.add_history(
            f"Combined exposure ({mode}) across sub-bands: "
            + ", ".join(f"{sb.emin}-{sb.emax}eV" for sb in selected)
        )
        return Image2D(data=total, header=hdr, path=None)

    def sum_smoothed_flux(
        self,
        *,
        subband_indices: Sequence[int] | None = None,
        apply_mask: bool = True,
        fill_value: float = np.nan,
    ) -> Image2D:
        """Sum ``eimagecombine`` ``asmooth`` flux maps across sub-bands."""
        selected = self._select_subbands(subband_indices)
        if any(sb.smoothed_flux is None for sb in selected):
            raise ValueError(
                f"Band {self.band_name!r} does not have smoothed flux maps for all "
                "selected sub-bands (run qpb-image with with_asmooth = true)"
            )
        ref = selected[0].smoothed_flux
        assert ref is not None
        total = np.zeros_like(ref.data, dtype=np.float64)
        any_valid = np.zeros(total.shape, dtype=bool)
        for sb in selected:
            assert sb.smoothed_flux is not None
            arr = sb.smoothed_flux.data
            valid = np.isfinite(arr)
            total[valid] += arr[valid]
            any_valid |= valid
        if apply_mask and self.mask_bool is not None:
            any_valid &= self.mask_bool
        out = np.full(total.shape, fill_value, dtype=np.float64)
        out[any_valid] = total[any_valid]

        hdr = ref.header.copy()
        hdr["BUNIT"] = ("counts/s", f"Summed smoothed rate ({self.emin}-{self.emax} eV)")
        hdr.add_history(
            "Summed smoothed flux across sub-bands: "
            + ", ".join(f"{sb.emin}-{sb.emax}eV" for sb in selected)
        )
        return Image2D(data=out, header=hdr, path=None)

    def compute_flux(
        self,
        *,
        subband_indices: Sequence[int] | None = None,
        weights: Sequence[float] | None = None,
        apply_mask: bool = True,
        min_exp: float = 0.0,
        min_exp_frac: float = 0.0,
        fill_value: float = np.nan,
    ) -> Image2D:
        """Compute ``sum_i (w_i * counts_i / exposure_i)`` across sub-bands."""
        selected = self._select_subbands(subband_indices)
        if weights is None:
            w_list = [1.0] * len(selected)
        else:
            if len(weights) != len(selected):
                raise ValueError(
                    f"Expected {len(selected)} weights, got {len(weights)}"
                )
            w_list = [float(w) for w in weights]

        shape = selected[0].counts.data.shape
        valid_all = np.ones(shape, dtype=bool)
        if apply_mask and self.mask_bool is not None:
            valid_all &= self.mask_bool

        for sb in selected:
            c = sb.counts.data
            e = sb.exposure.data
            finite_pos = e[np.isfinite(e) & (e > 0)]
            thresh = min_exp
            if min_exp_frac > 0 and finite_pos.size:
                thresh = max(thresh, float(finite_pos.max() * min_exp_frac))
            valid_all &= np.isfinite(c) & np.isfinite(e) & (e > thresh)

        total = np.full(shape, fill_value, dtype=np.float64)
        total[valid_all] = 0.0
        for sb, w in zip(selected, w_list):
            total[valid_all] += (
                w * sb.counts.data[valid_all] / sb.exposure.data[valid_all]
            )

        hdr = selected[0].counts.header.copy()
        hdr["BUNIT"] = ("counts/s", f"Combined rate ({self.emin}-{self.emax} eV)")
        hdr.add_history(
            "Computed weighted sum_i(w_i * counts_i / exposure_i) across: "
            + ", ".join(f"{sb.emin}-{sb.emax}eV" for sb in selected)
        )
        return Image2D(data=total, header=hdr, path=None)

    @classmethod
    def from_band_products(cls, products: BandCombinedProducts) -> BandImages:
        """Load FITS images for a ``BandCombinedProducts`` entry."""
        subbands: list[SubbandImages] = []
        for sb in products.subbands:
            counts = Image2D.from_fits(sb.counts_image)
            exposure = Image2D.from_fits(sb.exposure_image)
            smoothed = (
                Image2D.from_fits(sb.smoothed_flux_image)
                if sb.smoothed_flux_image is not None and sb.smoothed_flux_image.is_file()
                else None
            )
            subbands.append(
                SubbandImages(
                    index=sb.index,
                    emin=sb.emin,
                    emax=sb.emax,
                    counts=counts,
                    exposure=exposure,
                    smoothed_flux=smoothed,
                )
            )
        mask = (
            Image2D.from_fits(products.mask_image)
            if products.mask_image is not None and products.mask_image.is_file()
            else None
        )
        flux = (
            Image2D.from_fits(products.flux_image)
            if products.flux_image.is_file()
            else None
        )
        return cls(
            band_name=products.band_name,
            emin=products.emin,
            emax=products.emax,
            subbands=subbands,
            mask=mask,
            flux=flux,
        )


@dataclass
class QPBProducts:
    """Top-level wrapper for reading and combining ``qpb_image`` outputs."""

    qpb_image_dir: Path
    manifest_path: Path
    bands: dict[str, BandImages] = field(default_factory=dict)

    def __getitem__(self, band_name: str) -> BandImages:
        return self.bands[band_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.bands)

    def __len__(self) -> int:
        return len(self.bands)

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        band_names: Iterable[str] | None = None,
    ) -> QPBProducts:
        """Load ``qpb_image`` products from ``manifest.json`` or its directory."""
        p = Path(path)
        manifest_path = p / "manifest.json" if p.is_dir() else p
        res = QPBImageResult.from_manifest(manifest_path)

        selected = set(band_names) if band_names is not None else None
        loaded_bands: dict[str, BandImages] = {}
        for bname, bprod in res.bands.items():
            if selected is not None and bname not in selected:
                continue
            loaded_bands[bname] = BandImages.from_band_products(bprod)

        return cls(
            qpb_image_dir=res.qpb_image_dir,
            manifest_path=res.manifest_path,
            bands=loaded_bands,
        )
