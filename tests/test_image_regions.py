"""Tests for FITS image loading, smoothing, strip-grid regions, and GUI state."""

import tempfile
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np
import regions as astropy_regions
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from xmm_tools.image import (
    BoxRegion,
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

try:
    from xmm_tools.image.gui import ImageViewerState
except ModuleNotFoundError as exc:
    if exc.name != "imgui_bundle" and not (exc.name or "").startswith("imgui_bundle."):
        raise
    ImageViewerState = None  # type: ignore[assignment,misc]


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

    @unittest.skipIf(ImageViewerState is None, "imgui_bundle not installed")
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
        with tempfile.TemporaryDirectory() as tmp:
            fpath = _write_test_fits(Path(tmp) / "flux.fits")
            img = load_fits_image(fpath)

        params = StripGridParams(
            start_ra_deg=287.956,
            start_dec_deg=4.982,
            pa_deg=125.0,
            width_arcsec=20.0,
            height_arcsec=40.0,
            count=1,
            prefix="b",
        )
        box = generate_strip_grid(params)[0]
        corners_before = box_corners_pixels(box, img)
        self.assertIsNotNone(corners_before)
        bot_mid_before = np.mean(corners_before[:2], axis=0)

        pa_top = box.angle_deg % 360.0
        orig_center = SkyCoord(ra=box.ra_deg * u.deg, dec=box.dec_deg * u.deg, frame="fk5")
        orig_bottom = orig_center.directional_offset_by(
            ((pa_top + 180.0) % 360.0) * u.deg, 20.0 * u.arcsec
        )
        new_top = orig_center.directional_offset_by(pa_top * u.deg, 40.0 * u.arcsec)
        resize_box_edge(box, float(new_top.ra.deg), float(new_top.dec.deg), edge="top")

        self.assertAlmostEqual(box.height_arcsec, 60.0, places=2)
        # Opposite (bottom) edge and its pixel handle must remain fixed
        new_center = SkyCoord(ra=box.ra_deg * u.deg, dec=box.dec_deg * u.deg, frame="fk5")
        new_bottom = new_center.directional_offset_by(
            ((pa_top + 180.0) % 360.0) * u.deg, 30.0 * u.arcsec
        )
        self.assertAlmostEqual(float(orig_bottom.separation(new_bottom).arcsec), 0.0, places=2)
        corners_after = box_corners_pixels(box, img)
        self.assertIsNotNone(corners_after)
        np.testing.assert_allclose(
            np.mean(corners_after[:2], axis=0), bot_mid_before, atol=1e-2
        )

    def test_ds9_ellipse_and_box_counterclockwise_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fpath = _write_test_fits(root / "flux.fits")
            img = load_fits_image(fpath)
            assert img.wcs is not None

            # 1. FK5 ellipse & box with positive DS9 angle must rotate CCW on sky/image
            # (+major at theta=0 moves toward +x, +y on North-up East-left TAN image)
            for ang in (0.0, 30.0, 45.0, 135.0, 315.0):
                center = SkyCoord(287.956 * u.deg, 4.982 * u.deg, frame="fk5")
                ell = EllipseRegion("ell", 287.956, 4.982, 40.0, 12.0, angle_deg=ang)
                pts = np.asarray(ellipse_outline_pixels(ell, img, n_pts=48))
                expected_ell = (
                    astropy_regions.EllipseSkyRegion(
                        center, 80.0 * u.arcsec, 24.0 * u.arcsec, angle=ang * u.deg
                    )
                    .to_pixel(img.wcs)
                    .to_polygon(n_vertices=48)
                    .vertices
                )
                np.testing.assert_allclose(pts[:, 0], expected_ell.x, atol=0.05)
                np.testing.assert_allclose(pts[:, 1], expected_ell.y, atol=0.05)

                box = BoxRegion("box", 287.956, 4.982, 80.0, 24.0, angle_deg=ang)
                corners = np.asarray(box_corners_pixels(box, img))
                expected_box = (
                    astropy_regions.RectangleSkyRegion(
                        center, 80.0 * u.arcsec, 24.0 * u.arcsec, angle=ang * u.deg
                    )
                    .to_pixel(img.wcs)
                    .corners
                )
                np.testing.assert_allclose(corners, expected_box, atol=0.05)

            # Explicit sign check at 30 deg: +major endpoint is up-right (dx > 0, dy > 0)
            c_px = sky_to_pixel(img, 287.956, 4.982)
            assert c_px is not None
            pts30 = ellipse_outline_pixels(
                EllipseRegion("e30", 287.956, 4.982, 40.0, 12.0, angle_deg=30.0),
                img,
                n_pts=48,
            )
            assert pts30 is not None
            self.assertGreater(pts30[0][0] - c_px[0], 8.0)
            self.assertGreater(pts30[0][1] - c_px[1], 4.5)

            # 2. Image-frame and Galactic-frame DS9 .reg files preserve exact pixel orientation
            img_reg = root / "image_ell.reg"
            img_reg.write_text(
                "image\nellipse(33, 33, 10, 3, 35) # text={img_ell}\n",
                encoding="utf-8",
            )
            parsed_img = parse_ds9_regions(img_reg, wcs=img.wcs)
            self.assertEqual(len(parsed_img), 1)
            assert isinstance(parsed_img[0], EllipseRegion)
            pts_img = np.asarray(ellipse_outline_pixels(parsed_img[0], img, n_pts=48))
            gt_img = (
                astropy_regions.Regions.read(img_reg, format="ds9")[0]
                .to_polygon(n_vertices=48)
                .vertices
            )
            np.testing.assert_allclose(pts_img[:, 0], gt_img.x, atol=0.05)
            np.testing.assert_allclose(pts_img[:, 1], gt_img.y, atol=0.05)

            gal_c = SkyCoord(287.956 * u.deg, 4.982 * u.deg, frame="fk5").galactic
            gal_reg = root / "gal_ell.reg"
            gal_reg.write_text(
                f'galactic\nellipse({gal_c.l.deg:.7f},{gal_c.b.deg:.7f},36",12",25) # text={{gal}}\n',
                encoding="utf-8",
            )
            parsed_gal = parse_ds9_regions(gal_reg, wcs=img.wcs)
            assert isinstance(parsed_gal[0], EllipseRegion)
            pts_gal = np.asarray(ellipse_outline_pixels(parsed_gal[0], img, n_pts=48))
            gt_gal = (
                astropy_regions.Regions.read(gal_reg, format="ds9")[0]
                .to_pixel(img.wcs)
                .to_polygon(n_vertices=48)
                .vertices
            )
            np.testing.assert_allclose(pts_gal[:, 0], gt_gal.x, atol=0.05)
            np.testing.assert_allclose(pts_gal[:, 1], gt_gal.y, atol=0.05)

            # 3. Oblique strip grid exported to DS9 .reg has contiguous abutting edges in DS9
            oblique = generate_strip_grid(
                StripGridParams(
                    start_ra_deg=287.956,
                    start_dec_deg=4.982,
                    pa_deg=125.0,
                    width_arcsec=20.0,
                    height_arcsec=48.0,
                    count=2,
                )
            )
            strip_reg = root / "strip.reg"
            export_ds9_regions(oblique, strip_reg)
            ds9_boxes = [
                r.to_pixel(img.wcs)
                for r in astropy_regions.Regions.read(strip_reg, format="ds9")
            ]
            # Box 0 left (-x_local / East-side) edge must coincide with Box 1 right (+x_local) edge
            edge0 = np.sort(ds9_boxes[0].corners[[0, 3]], axis=0)
            edge1 = np.sort(ds9_boxes[1].corners[[1, 2]], axis=0)
            np.testing.assert_allclose(edge0, edge1, atol=0.05)


if __name__ == "__main__":
    unittest.main()
