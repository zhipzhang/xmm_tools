"""Derived product export and atomic flux synthesis for ``qpb_image``."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from xmm_tools.config import BandSpec
from xmm_tools.exec import is_nonempty_file, publish_artifact
from xmm_tools.sas.qpb_image.models import BandProductPaths, SubbandProductPaths
from xmm_tools.sas.qpb_image.products import BandImages


def build_band_product_paths(
    qdir: Path,
    band_name: str,
    band: BandSpec,
    subbands: Sequence[tuple[int, int]],
    flux_path: Path,
    *,
    has_mask: bool,
) -> BandProductPaths:
    """Construct the canonical ``BandProductPaths`` layout under ``qdir / band_name``."""
    export_dir = qdir / band_name
    exported_mask = (
        export_dir / f"{band_name}_{band.emin}_{band.emax}_mask.fits"
        if has_mask
        else None
    )
    exported_subbands = [
        SubbandProductPaths(
            index=idx,
            emin=lo,
            emax=hi,
            counts_image=export_dir / f"{band_name}_{lo}_{hi}_counts.fits",
            exposure_image=export_dir / f"{band_name}_{lo}_{hi}_exp.fits",
            smoothed_flux_image=export_dir / f"{band_name}_{lo}_{hi}_smooth_flux.fits",
        )
        for idx, (lo, hi) in enumerate(subbands)
    ]
    return BandProductPaths(
        band_name=band_name,
        emin=band.emin,
        emax=band.emax,
        flux_image=flux_path,
        mask_image=exported_mask,
        subbands=exported_subbands,
    )


def _atomic_write_flux_fits(flux_image: BandImages, flux_path: Path) -> None:
    flux_path.parent.mkdir(parents=True, exist_ok=True)
    summed = flux_image.sum_smoothed_flux(apply_mask=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=flux_path.parent, prefix=f".{flux_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        summed.writeto(tmp_path, overwrite=True)
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, flux_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def export_band_products(
    qdir: Path,
    band_name: str,
    band: BandSpec,
    subbands: list[tuple[int, int]],
    combine_inputs: Mapping[str, Path],
    flux_path: Path,
) -> tuple[BandProductPaths, dict[str, Path]]:
    """Publish band sub-band FITS products and atomically synthesize ``flux_path``."""
    exported_files: dict[str, Path] = {}

    mask_src = combine_inputs.get("mask")
    has_mask = mask_src is not None and is_nonempty_file(mask_src)
    band_prod = build_band_product_paths(
        qdir,
        band_name,
        band,
        subbands,
        flux_path,
        has_mask=has_mask,
    )

    if has_mask and mask_src is not None and band_prod.mask_image is not None:
        exported_files["mask"] = publish_artifact(
            mask_src, band_prod.mask_image, mode="copy"
        )

    for idx, sb_prod in enumerate(band_prod.subbands):
        exported_files[f"counts_{idx}"] = publish_artifact(
            combine_inputs[f"sub_{idx}"], sb_prod.counts_image, mode="copy"
        )
        exported_files[f"exp_{idx}"] = publish_artifact(
            combine_inputs[f"exp_{idx}"], sb_prod.exposure_image, mode="copy"
        )
        assert sb_prod.smoothed_flux_image is not None
        exported_files[f"smooth_{idx}"] = publish_artifact(
            combine_inputs[f"smooth_{idx}"],
            sb_prod.smoothed_flux_image,
            mode="copy",
        )

    # Ignore any pre-existing flux_path on disk so corrupted files regenerate cleanly.
    band_images = BandImages.from_band_products(band_prod, load_flux=False)
    _atomic_write_flux_fits(band_images, flux_path)
    exported_files["flux"] = flux_path
    return band_prod, exported_files
