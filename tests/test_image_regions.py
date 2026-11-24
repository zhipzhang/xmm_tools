"""Tests for FITS image loading, smoothing, strip-grid regions, and GUI state."""

from pathlib import Path
import tempfile
import unittest

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import numpy as np

from xmm_tools.image import (
    CircleRegion,
    EllipseRegion,
    PolygonRegion,
    StripGridParams,
    box_corners_pixels,
    ellipse_outline_pixels,
    export_ds9_regions,
    generate_strip_grid,
    load_fits_image,
    parse_ds9_regions,
    pixel_to_sky,
    polygon_pixels,
    resize_box_edge,
    sky_to_pixel,
    smooth_image,
    stretch_image,
    strip_params_from_sky_points,
)
from xmm_tools.image.gui import ImageViewerState


def _write_test_fits(path: Path, ny: int = 64, nx: int = 64) -> Path:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    wcs.wcs.cdelt = [-4.0 / 3600.0, 4.0 / 3600.0]  # 4 arcsec/pix
    wcs.wcs.crval = [287.956, 4.982]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()

    data = np.zeros((ny, nx), dtype=np.float64)
    data[ny // 2, nx // 2] = 100.0
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return path


class TestFitsImageAndRegions(unittest.TestCase):
    def test_load_smooth_stretch_and_wcs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fpath = _write_test_fits(Path(tmp) / "flux.fits")
            img = load_fits_image(fpath)
            self.assertEqual(img.shape, (64, 64))
            self.assertAlmostEqual(img.pixel_scale_arcsec, 4.0, places=3)

            sky = pixel_to_sky(img, 31.0, 31.0)
            self.assertIsNotNone(sky)
            px = sky_to_pixel(img, sky[0], sky[1])
            self.assertIsNotNone(px)
            self.assertAlmostEqual(px[0], 31.0, places=3)
            self.assertAlmostEqual(px[1], 31.0, places=3)

            smoothed = smooth_image(img.data, sigma_pixels=1.5)
            self.assertLess(smoothed.max(), 100.0)
            self.assertGreater(smoothed.max(), 1.0)

            for mode in ("linear", "sqrt", "log10", "asinh"):
                disp, lo, hi = stretch_image(smoothed, mode=mode)
                self.assertEqual(disp.shape, (64, 64))
                self.assertGreater(hi, lo)

    def test_strip_grid_generation_and_ds9_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fpath = _write_test_fits(root / "flux.fits")
            img = load_fits_image(fpath)

            params = StripGridParams(
                start_ra_deg=287.956,
                start_dec_deg=4.982,
                pa_deg=90.0,
                width_arcsec=20.0,
                height_arcsec=60.0,
                count=4,
                gap_arcsec=0.0,
                prefix="e",
            )
            boxes = generate_strip_grid(params)
            self.assertEqual(len(boxes), 4)
            self.assertEqual([b.name for b in boxes], ["e1", "e2", "e3", "e4"])

            # Adjacent box centers should be separated by exactly width_arcsec (20")
            c1 = SkyCoord(ra=boxes[0].ra_deg * u.deg, dec=boxes[0].dec_deg * u.deg, frame="fk5")
            c2 = SkyCoord(ra=boxes[1].ra_deg * u.deg, dec=boxes[1].dec_deg * u.deg, frame="fk5")
            self.assertAlmostEqual(float(c1.separation(c2).arcsec), 20.0, places=2)

            corners = box_corners_pixels(boxes[0], img)
            self.assertIsNotNone(corners)
            self.assertEqual(len(corners), 4)

            # Check strip_params_from_sky_points
            c_end = SkyCoord(ra=287.956 * u.deg, dec=4.982 * u.deg, frame="fk5").directional_offset_by(
                90.0 * u.deg, 80.0 * u.arcsec
            )
            derived = strip_params_from_sky_points(
                287.956,
                4.982,
                float(c_end.ra.deg),
                float(c_end.dec.deg),
                width_arcsec=20.0,
                height_arcsec=60.0,
            )
            self.assertEqual(derived.count, 4)

            reg_path = root / "test.reg"
            all_regs = [
                *boxes,
                CircleRegion("src", 287.956, 4.982, 30.0, include=False),
                EllipseRegion("ell1", 287.96, 4.98, 25.0, 12.0, angle_deg=30.0),
                PolygonRegion(
                    "poly1",
                    [(287.95, 4.97), (287.96, 4.97), (287.96, 4.99)],
                ),
            ]
            export_ds9_regions(all_regs, reg_path)
            parsed = parse_ds9_regions(reg_path, wcs=img.wcs)
            self.assertEqual(len(parsed), 7)
            self.assertEqual(parsed[0].name, "e1")
            self.assertFalse(parsed[4].include)
            self.assertIsInstance(parsed[5], EllipseRegion)
            self.assertAlmostEqual(parsed[5].semi_major_arcsec, 25.0, places=2)
            self.assertIsInstance(parsed[6], PolygonRegion)
            self.assertEqual(len(parsed[6].vertices_deg), 3)
            self.assertEqual(len(ellipse_outline_pixels(parsed[5], img)), 48)
            self.assertEqual(len(polygon_pixels(parsed[6], img)), 3)

    def test_viewer_state_load_and_apply_strip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fpath = _write_test_fits(root / "flux.fits")
            state = ImageViewerState(recent_store_path=root / "recent_fits.json")
            state.open_file(fpath)
            self.assertEqual(state.error_message, "")
            self.assertIsNotNone(state.get_display_flipped())
            state.strip_params.count = 3
            state.apply_strip_grid(replace=True)
            self.assertEqual(len(state.regions), 3)

            # Manual region addition (Circle, Box, Ellipse, Polygon)
            state.add_shape_idx = 0
            state.add_manual_region_at(287.956, 4.982)
            state.add_shape_idx = 1
            state.add_manual_region_at(287.956, 4.982)
            state.add_shape_idx = 2
            state.add_manual_region_at(287.956, 4.982)
            state.poly_draft_vertices = [(287.95, 4.97), (287.96, 4.97), (287.96, 4.99)]
            state.finish_polygon_draft()
            self.assertEqual(len(state.regions), 7)

    def test_resize_box_edge_single_side_stretch(self):
        params = StripGridParams(
            start_ra_deg=287.956,
            start_dec_deg=4.982,
            pa_deg=90.0,
            width_arcsec=20.0,
            height_arcsec=40.0,
            count=1,
            prefix="b",
        )
        box = generate_strip_grid(params)[0]
        # Original top edge is 20" along pa_top = 180 - 90 = 90 deg? Wait:
        # pa_top_deg = (180 - angle_deg) = 180 - 0 = 180 deg (South) or 0 deg.
        # Move drag point 30" further along pa_top_deg from original center -> new height = 50"
        pa_top = (180.0 - box.angle_deg) % 360.0
        orig_center = SkyCoord(ra=box.ra_deg * u.deg, dec=box.dec_deg * u.deg, frame="fk5")
        orig_bottom = orig_center.directional_offset_by(
            ((pa_top + 180.0) % 360.0) * u.deg, 20.0 * u.arcsec
        )
        new_top = orig_center.directional_offset_by(pa_top * u.deg, 40.0 * u.arcsec)
        resize_box_edge(box, float(new_top.ra.deg), float(new_top.dec.deg), edge="top")

        self.assertAlmostEqual(box.height_arcsec, 60.0, places=2)
        # Opposite (bottom) edge should remain fixed
        new_center = SkyCoord(ra=box.ra_deg * u.deg, dec=box.dec_deg * u.deg, frame="fk5")
        new_bottom = new_center.directional_offset_by(
            ((pa_top + 180.0) % 360.0) * u.deg, 30.0 * u.arcsec
        )
        self.assertAlmostEqual(float(orig_bottom.separation(new_bottom).arcsec), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
