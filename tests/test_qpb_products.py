"""Tests for standalone ``xmm_tools.sas.qpb_products`` wrapper and reader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from xmm_tools.sas import Image2D, QPBProducts
from xmm_tools.sas.qpb_image import (
    BandCombinedProducts,
    QPBImageResult,
    SubbandCombinedProducts,
)


def _write_fits(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data.astype(np.float32)).writeto(path, overwrite=True)
    return path


class QPBProductsTests(unittest.TestCase):
    def test_wrapper_loads_and_combines_counts_exposure_mask_flux(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdir = Path(tmp) / "qpb_image"
            bdir = qdir / "soft"

            c0 = _write_fits(bdir / "soft_500_1000_counts.fits", np.full((3, 3), 4.0))
            e0 = _write_fits(bdir / "soft_500_1000_exp.fits", np.full((3, 3), 2.0))
            s0 = _write_fits(bdir / "soft_500_1000_smooth_flux.fits", np.full((3, 3), 2.1))

            c1 = _write_fits(bdir / "soft_1000_2000_counts.fits", np.full((3, 3), 6.0))
            e1_arr = np.full((3, 3), 2.0)
            e1_arr[0, 1] = 0.05  # Low-exposure edge pixel
            e1 = _write_fits(bdir / "soft_1000_2000_exp.fits", e1_arr)
            s1 = _write_fits(bdir / "soft_1000_2000_smooth_flux.fits", np.full((3, 3), 3.1))

            mask_arr = np.ones((3, 3), dtype=np.float32)
            mask_arr[0, 0] = 0.0
            mask = _write_fits(bdir / "soft_500_2000_mask.fits", mask_arr)
            flux = _write_fits(qdir / "soft_500_2000_flux.fits", np.full((3, 3), 5.0))

            res = QPBImageResult(
                qpb_image_dir=qdir,
                manifest_path=qdir / "manifest.json",
                band_flux_images={"soft": flux},
                bands={
                    "soft": BandCombinedProducts(
                        band_name="soft",
                        emin=500,
                        emax=2000,
                        flux_image=flux,
                        mask_image=mask,
                        subbands=[
                            SubbandCombinedProducts(
                                index=0,
                                emin=500,
                                emax=1000,
                                counts_image=c0,
                                exposure_image=e0,
                                smoothed_flux_image=s0,
                            ),
                            SubbandCombinedProducts(
                                index=1,
                                emin=1000,
                                emax=2000,
                                counts_image=c1,
                                exposure_image=e1,
                                smoothed_flux_image=s1,
                            ),
                        ],
                    )
                },
            )
            res.save_manifest()

            products = QPBProducts.from_manifest(qdir)
            self.assertEqual(list(products), ["soft"])
            soft = products["soft"]
            self.assertTrue(soft.has_smoothed_flux)
            self.assertEqual(len(soft.subbands), 2)
            self.assertFalse(soft.mask_bool[0, 0])

            # sum_counts wrapper returns Image2D
            cnt_img = soft.sum_counts(apply_mask=True, fill_value=0.0)
            self.assertIsInstance(cnt_img, Image2D)
            self.assertEqual(cnt_img.data[0, 0], 0.0)
            np.testing.assert_allclose(cnt_img.data[1, 1], 10.0)

            # sum_exposure wrapper (mean and sum)
            exp_mean = soft.sum_exposure(mode="mean")
            exp_sum = soft.sum_exposure(mode="sum")
            np.testing.assert_allclose(exp_mean.data[1, 1], 2.0)
            np.testing.assert_allclose(exp_sum.data[1, 1], 4.0)

            # sum_smoothed_flux applies mask and returns Image2D
            sm_img = soft.sum_smoothed_flux(apply_mask=True, fill_value=np.nan)
            self.assertTrue(np.isnan(sm_img.data[0, 0]))
            np.testing.assert_allclose(sm_img.data[1, 1], 5.2, rtol=1e-5)

            # compute_flux with min_exp_frac and weights, plus writeto
            flux_img = soft.compute_flux(
                weights=[1.0, 2.0], min_exp_frac=0.1, fill_value=0.0
            )
            self.assertEqual(flux_img.data[0, 0], 0.0)
            self.assertEqual(flux_img.data[0, 1], 0.0)
            # 1.0 * (4/2) + 2.0 * (6/2) = 8.0
            np.testing.assert_allclose(flux_img.data[1, 1], 8.0)

            out_path = flux_img.writeto(Path(tmp) / "custom_flux.fits")
            self.assertTrue(out_path.is_file())
            reloaded = Image2D.from_fits(out_path)
            np.testing.assert_allclose(reloaded.data[1, 1], 8.0)


if __name__ == "__main__":
    unittest.main()
