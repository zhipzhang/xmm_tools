"""Tests for XMM-Newton cheese source-list (emllist.fits) loading, merging, and GUI state."""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from xmm_tools.image.fits_io import load_fits_image
from xmm_tools.srclist import (
    EpochCatalog,
    FilterParams,
    MergedSource,
    SourceDetection,
    angular_separation_arcsec,
    export_merged_ds9_reg,
    export_merged_emllist,
    filter_detections,
    load_cheese_epoch,
    load_emllist,
    load_sky_region_ellipses,
    merge_catalogs,
)
from xmm_tools.srclist.gui import SrclistViewerState
from xmm_tools.srclist.gui import main as srclist_main
from xmm_tools.srclist.io import STANDARD_SRCLIST_COLUMNS

_REAL_CHEESE_DIR = Path(
    "/Users/zhipz/CodeProjects/XMMNewton_Workdir/xmm_obs/"
    "0923400201_xmm_tools_refactor_test/clean_e2e_run/cheese"
)


def _write_synthetic_emllist(
    path: Path,
    sources: list[dict[str, float]],
    *,
    obs_id: str = "0123456789",
    crval: tuple[float, float] = (200.0, 30.0),
) -> Path:
    """Write a multi-row emldetect-style emllist.fits (ID_INST=0..3 x ID_BAND=0..3)."""
    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(sources, start=1):
        ra = float(spec["ra"])
        dec = float(spec["dec"])
        det_ml = float(spec.get("det_ml", 40.0))
        flux_summary = float(spec.get("flux_summary", 1.0e-14))
        flux_band1_m2 = float(spec.get("flux_band1_m2", flux_summary))
        scts = float(spec.get("scts", 120.0))
        bg_map = float(spec.get("bg_map", 0.5))
        ext = float(spec.get("ext", 0.0))
        ext_ml = float(spec.get("ext_ml", 0.0))
        dist_nn = float(spec.get("dist_nn", 120.0))
        radec_err = float(spec.get("radec_err", 0.5))

        # Summary row (ID_INST=0, ID_BAND=0)
        rows.append(
            {
                "ML_ID_SRC": idx,
                "BOX_ID_SRC": idx,
                "ID_INST": 0,
                "ID_BAND": 0,
                "ID_CLUSTER": idx,
                "SCTS": scts,
                "SCTS_ERR": math.sqrt(max(scts, 1.0)),
                "X_IMA": 450.5,
                "Y_IMA": 450.5,
                "EXT": ext,
                "DET_ML": det_ml,
                "EXT_ML": ext_ml,
                "BG_MAP": bg_map,
                "EXP_MAP": 1.0e4,
                "FLUX": flux_summary,
                "RATE": 0.01,
                "RA": ra,
                "DEC": dec,
                "RADEC_ERR": radec_err,
                "DIST_NN": dist_nn,
                "FLAG": "________",
            }
        )
        # Per-instrument rows mimicking emldetect quirk
        for inst in (1, 2, 3):
            for band in (0, 1, 2, 3):
                flx = flux_band1_m2 if (inst == 2 and band == 1) else 0.0
                bg_inst = 0.0 if inst == 2 else bg_map * 0.4
                rows.append(
                    {
                        "ML_ID_SRC": idx,
                        "BOX_ID_SRC": idx,
                        "ID_INST": inst,
                        "ID_BAND": band,
                        "ID_CLUSTER": idx,
                        "SCTS": scts * 0.5 if inst == 2 else 1.0,
                        "SCTS_ERR": 5.0,
                        "X_IMA": 450.5,
                        "Y_IMA": 450.5,
                        "EXT": ext,
                        "DET_ML": det_ml * 0.5 if inst == 2 else 0.1,
                        "EXT_ML": ext_ml,
                        "BG_MAP": bg_inst,
                        "EXP_MAP": 1.0e4,
                        "FLUX": flx,
                        "RATE": 0.005,
                        "RA": ra,
                        "DEC": dec,
                        "RADEC_ERR": radec_err,
                        "DIST_NN": dist_nn,
                        "FLAG": "________",
                    }
                )

    cols: list[fits.Column] = []
    for name, fmt, unit in STANDARD_SRCLIST_COLUMNS:
        upper = fmt.upper()
        if "A" in upper:
            vals = [str(r.get(name, "")) for r in rows]
            arr = np.asarray(vals, dtype=str)
        elif any(c in upper for c in ("J", "I", "K", "B")):
            vals = [int(r.get(name, 0)) for r in rows]
            arr = np.asarray(vals, dtype=np.int32)
        elif "D" in upper:
            vals = [float(r.get(name, 0.0)) for r in rows]
            arr = np.asarray(vals, dtype=np.float64)
        else:
            vals = [float(r.get(name, 0.0)) for r in rows]
            arr = np.asarray(vals, dtype=np.float32)
        cols.append(fits.Column(name=name, format=fmt, unit=unit, array=arr))

    pri = fits.PrimaryHDU()
    pri.header["OBS_ID"] = obs_id
    pri.header["TCTYPE1"] = "RA---TAN"
    pri.header["TCTYPE2"] = "DEC--TAN"
    pri.header["TCRPIX1"] = 450.5
    pri.header["TCRPIX2"] = 450.5
    pri.header["TCRVAL1"] = crval[0]
    pri.header["TCRVAL2"] = crval[1]
    pri.header["TCDELT1"] = -0.000694444461259991
    pri.header["TCDELT2"] = 0.000694444461259991
    pri.header["TCUNIT1"] = "deg"
    pri.header["TCUNIT2"] = "deg"
    pri.header["RADECSYS"] = "FK5"
    pri.header["EQUINOX"] = 2000.0

    tbl = fits.BinTableHDU.from_columns(cols, name="SRCLIST")
    fits.HDUList([pri, tbl]).writeto(path, overwrite=True)
    return path


def _write_synthetic_skyreg(path: Path, crval: tuple[float, float] = (200.0, 30.0)) -> Path:
    """Write a SAS *-bkgregtsky.fits REGION table with !ELLIPSE and !CIRCLE."""
    shapes = np.asarray(["!ELLIPSE", "!ELLIPSE", "!CIRCLE"], dtype="U16")
    # 0.05"/pix -> 25921 is crpix
    x = np.array(
        [
            [25921.0, 0.0, 0.0, 0.0],
            [26121.0, 0.0, 0.0, 0.0],
            [25721.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array(
        [
            [25921.0, 0.0, 0.0, 0.0],
            [26121.0, 0.0, 0.0, 0.0],
            [25721.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Row 0: a=400 (20"), b=200 (10"); Row 1: R[0]=150 < R[1]=300 (swap + 90 deg)
    r = np.array(
        [
            [400.0, 200.0, 0.0, 0.0],
            [150.0, 300.0, 0.0, 0.0],
            [300.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotang = np.array(
        [
            [30.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    comp = np.array([1, 1, 1], dtype=np.int32)

    tbl = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="SHAPE", format="16A", array=shapes),
            fits.Column(name="X", format="4E", array=x),
            fits.Column(name="Y", format="4E", array=y),
            fits.Column(name="R", format="4E", array=r),
            fits.Column(name="ROTANG", format="4E", array=rotang),
            fits.Column(name="COMPONENT", format="J", array=comp),
        ],
        name="REGION",
    )
    tbl.header["TCTYP2"] = "RA---TAN"
    tbl.header["TCTYP3"] = "DEC--TAN"
    tbl.header["TCRPX2"] = 25921.0
    tbl.header["TCRPX3"] = 25921.0
    tbl.header["TCRVL2"] = crval[0]
    tbl.header["TCRVL3"] = crval[1]
    tbl.header["TCDLT2"] = -1.38888888888889e-5
    tbl.header["TCDLT3"] = 1.38888888888889e-5
    fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(path, overwrite=True)
    return path


def _write_dummy_cheeset(path: Path, crval: tuple[float, float] = (200.0, 30.0)) -> Path:
    hdu = fits.PrimaryHDU(data=np.ones((32, 32), dtype=np.float32))
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRPIX1"] = 16.5
    hdu.header["CRPIX2"] = 16.5
    hdu.header["CRVAL1"] = crval[0]
    hdu.header["CRVAL2"] = crval[1]
    hdu.header["CDELT1"] = -0.000694444461259991
    hdu.header["CDELT2"] = 0.000694444461259991
    hdu.writeto(path, overwrite=True)
    return path


class TestSrclistIOAndMerge(unittest.TestCase):
    def test_load_emllist_band1_flux_quirk_and_bg(self):
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "emllist.fits"
            _write_synthetic_emllist(
                fpath,
                [
                    {
                        "ra": 200.0,
                        "dec": 30.0,
                        "det_ml": 3500.0,
                        "flux_summary": 1.45e-15,
                        "flux_band1_m2": 2.82e-13,
                        "bg_map": 0.75,
                    },
                    {
                        "ra": 200.05,
                        "dec": 30.02,
                        "det_ml": 12.0,
                        "flux_summary": 5.0e-15,
                        "flux_band1_m2": 0.0,
                        "bg_map": 0.40,
                    },
                ],
            )
            cat = load_emllist(fpath)
            self.assertEqual(cat.obs_id, "0123456789")
            self.assertEqual(len(cat.detections), 2)
            # First source picks up band-1 per-instrument max flux instead of summary.
            self.assertAlmostEqual(cat.detections[0].flux, 2.82e-13, places=17)
            self.assertAlmostEqual(cat.detections[0].bg_map, 0.75, places=5)
            # Second source falls back to summary flux when band-1 is zero.
            self.assertAlmostEqual(cat.detections[1].flux, 5.0e-15, places=19)

            kept = filter_detections(cat.detections, FilterParams(min_det_ml=15.0))
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0].ml_id_src, 1)

    def test_load_cheese_epoch_with_manifest_and_sky_ellipses(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs_dir = Path(tmp) / "obs1"
            cdir = obs_dir / "cheese"
            qdir = obs_dir / "qpb_image"
            cdir.mkdir(parents=True)
            qdir.mkdir(parents=True)

            eml = _write_synthetic_emllist(
                cdir / "emllist.fits",
                [{"ra": 200.0, "dec": 30.0, "det_ml": 50.0, "flux_summary": 1e-14}],
            )
            sky_m1 = _write_synthetic_skyreg(cdir / "mos1S001-bkgregtsky.fits")
            cheeset = _write_dummy_cheeset(cdir / "mos1S001-cheeset.fits")
            flux_img = _write_dummy_cheeset(qdir / "soft_500_2000_flux.fits")

            manifest = {
                "schema_version": 2,
                "stage": "cheese",
                "status": "completed",
                "products": {
                    "method": "cheese",
                    "emllist": eml.name,
                    "region_sky": {"mos1": sky_m1.name},
                    "cheese_masks": {"mos1": cheeset.name},
                },
            }
            (cdir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            # Direct load_sky_region_ellipses check
            direct_ells = load_sky_region_ellipses(sky_m1, instrument="mos1")
            self.assertEqual(len(direct_ells), 3)

            cat = load_cheese_epoch(cdir)
            self.assertEqual(cat.cheese_dir, cdir)
            self.assertEqual(len(cat.detections), 1)
            self.assertIn("mos1", cat.sky_ellipses)
            ells = cat.sky_ellipses["mos1"]
            self.assertEqual(len(ells), 3)
            self.assertAlmostEqual(ells[0].ra_deg, 200.0, places=5)
            self.assertAlmostEqual(ells[0].dec_deg, 30.0, places=5)
            self.assertAlmostEqual(ells[0].semi_major_arcsec, 20.0, places=3)
            self.assertAlmostEqual(ells[0].semi_minor_arcsec, 10.0, places=3)
            self.assertAlmostEqual(ells[0].angle_deg, 30.0, places=3)
            # Row 1 swapped axes and rotated by +90 deg.
            self.assertAlmostEqual(ells[1].semi_major_arcsec, 15.0, places=3)
            self.assertAlmostEqual(ells[1].semi_minor_arcsec, 7.5, places=3)
            self.assertAlmostEqual(ells[1].angle_deg, 100.0, places=3)
            # Row 2 circle
            self.assertAlmostEqual(ells[2].semi_major_arcsec, 15.0, places=3)
            self.assertAlmostEqual(ells[2].semi_minor_arcsec, 15.0, places=3)
            self.assertEqual(
                {p.resolve() for p in cat.candidate_images},
                {cheeset.resolve(), flux_img.resolve()},
            )

            # Also test loading via ana.toml
            ana_toml = obs_dir / "ana.toml"
            ana_toml.write_text(
                f'[paths]\nworkdir = "{obs_dir}"\ncheese_dir = "cheese"\n',
                encoding="utf-8",
            )
            cat_toml = load_cheese_epoch(ana_toml)
            self.assertEqual(len(cat_toml.detections), 1)

    def test_merge_catalogs_clustering_antichaining_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cos_dec = math.cos(math.radians(30.0))
            step_4arc = (4.0 / 3600.0) / cos_dec
            step_8arc = (8.0 / 3600.0) / cos_dec

            cat1 = load_emllist(
                _write_synthetic_emllist(
                    root / "ep1.fits",
                    [
                        {
                            "ra": 200.0,
                            "dec": 30.0,
                            "det_ml": 100.0,
                            "flux_summary": 2e-14,
                            "scts": 200.0,
                            "bg_map": 0.5,
                            "radec_err": 0.4,
                        },
                        {
                            "ra": 200.0 + step_8arc,
                            "dec": 30.0,
                            "det_ml": 25.0,
                            "flux_summary": 1e-14,
                            "scts": 40.0,
                            "bg_map": 0.8,
                            "radec_err": 0.8,
                        },
                    ],
                    obs_id="EP1",
                )
            )
            cat2 = load_emllist(
                _write_synthetic_emllist(
                    root / "ep2.fits",
                    [
                        {
                            "ra": 200.0 + step_4arc,
                            "dec": 30.0,
                            "det_ml": 60.0,
                            "flux_summary": 3e-14,
                            "scts": 150.0,
                            "bg_map": 0.0,
                            "radec_err": 1.2,
                        },
                    ],
                    obs_id="EP2",
                )
            )

            # match_radius=0 -> 3 separate merged sources (no cross-matching)
            unmerged = merge_catalogs(
                [cat1, cat2], filter_params=FilterParams(), match_radius_arcsec=0.0
            )
            self.assertEqual(len(unmerged), 3)

            # match_radius=5.0 -> A+B merge, C does NOT chain into A+B
            merged = merge_catalogs(
                [cat1, cat2],
                filter_params=FilterParams(),
                match_radius_arcsec=5.0,
                excluded_sky_coords=[(200.0 + step_8arc, 30.0)],
            )
            self.assertEqual(len(merged), 2)
            s1, s2 = merged[0], merged[1]
            self.assertEqual(len(s1.members), 2)
            self.assertTrue(s1.include)
            # Weighted RA pulls closer to higher-precision EP1 (err=0.4) than EP2 (err=1.2)
            self.assertGreater(s1.ra_deg, 200.0)
            self.assertLess(s1.ra_deg, 200.0 + 0.5 * step_4arc)
            self.assertAlmostEqual(s1.det_ml, 100.0, places=4)
            self.assertAlmostEqual(s1.flux, 3e-14, places=18)
            # BG_MAP prefers positive BG_MAP member (EP1: 0.5) over EP2 (0.0)
            self.assertAlmostEqual(s1.bg_map, 0.5, places=4)
            self.assertAlmostEqual(s1.scts, 200.0, places=3)

            self.assertEqual(len(s2.members), 1)
            self.assertFalse(s2.include)
            self.assertAlmostEqual(
                angular_separation_arcsec(s1.ra_deg, s1.dec_deg, s2.ra_deg, s2.dec_deg),
                s1.dist_nn_arcsec,
                places=3,
            )

    def test_export_emllist_and_ds9_and_cli_headless(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eml_in = _write_synthetic_emllist(
                root / "ep1.fits",
                [
                    {
                        "ra": 200.0,
                        "dec": 30.0,
                        "det_ml": 80.0,
                        "flux_summary": 2e-14,
                        "scts": 250.0,
                        "bg_map": 0.0,  # verify fallback > 0 on export
                    },
                    {
                        "ra": 200.1,
                        "dec": 30.1,
                        "det_ml": 20.0,
                        "flux_summary": 1e-14,
                        "scts": 30.0,
                        "bg_map": 0.25,
                    },
                ],
            )
            cat = load_emllist(eml_in)
            merged = merge_catalogs([cat], filter_params=FilterParams())
            merged[1].include = False

            out_fits = root / "merged_emllist.fits"
            export_merged_emllist(merged, out_fits, template_catalog=cat)
            with fits.open(out_fits) as hdul:
                srclist = hdul["SRCLIST"]
                self.assertEqual(len(srclist.data), 1)
                self.assertEqual(int(srclist.data["ML_ID_SRC"][0]), 1)
                self.assertEqual(int(srclist.data["ID_INST"][0]), 0)
                self.assertEqual(int(srclist.data["ID_BAND"][0]), 0)
                self.assertGreater(float(srclist.data["BG_MAP"][0]), 0.0)
                for key in (
                    "TCTYPE1",
                    "TCTYPE2",
                    "TCRPIX1",
                    "TCRPIX2",
                    "TCRVAL1",
                    "TCRVAL2",
                    "TCDELT1",
                    "TCDELT2",
                    "TCUNIT1",
                    "TCUNIT2",
                    "RADECSYS",
                    "EQUINOX",
                ):
                    self.assertIn(key, srclist.header)
                    self.assertIn(key, hdul[0].header)

            roundtrip = load_emllist(out_fits)
            self.assertEqual(len(roundtrip.detections), 1)
            self.assertAlmostEqual(roundtrip.detections[0].ra_deg, 200.0, places=5)

            out_reg = root / "merged.reg"
            export_merged_ds9_reg(merged, out_reg, include_excluded=True)
            reg_txt = out_reg.read_text(encoding="utf-8")
            self.assertIn("fk5", reg_txt)
            self.assertIn("circle(200.0000000,30.0000000,15.00\")", reg_txt)
            self.assertIn("-circle(200.1000000,30.1000000,15.00\")", reg_txt)

            # Empty export edge case should still write valid 0-row SRCLIST
            empty_fits = root / "empty_merged.fits"
            export_merged_emllist([], empty_fits, template_catalog=cat)
            with fits.open(empty_fits) as hdul:
                self.assertEqual(len(hdul["SRCLIST"].data), 0)

            # Headless CLI test
            cli_fits = root / "cli_merged.fits"
            cli_reg = root / "cli_merged.reg"
            rc = srclist_main(
                [
                    str(eml_in),
                    "--match-radius",
                    "5.0",
                    "--min-ml",
                    "15.0",
                    "--export-fits",
                    str(cli_fits),
                    "--export-reg",
                    str(cli_reg),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(cli_fits.is_file())
            self.assertTrue(cli_reg.is_file())

    def test_srclist_viewer_state_headless(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eml1 = _write_synthetic_emllist(
                root / "ep1.fits",
                [
                    {"ra": 200.0, "dec": 30.0, "det_ml": 90.0, "flux_summary": 2e-14},
                    {"ra": 200.08, "dec": 30.05, "det_ml": 10.0, "flux_summary": 5e-15},
                ],
                obs_id="OBS1",
            )
            eml2 = _write_synthetic_emllist(
                root / "ep2.fits",
                [{"ra": 200.0005, "dec": 30.0, "det_ml": 45.0, "flux_summary": 1e-14}],
                obs_id="OBS2",
            )

            state = SrclistViewerState()
            self.assertIsNone(state.get_active_wcs())

            state.add_epoch(eml1)
            state.add_epoch(eml2)
            self.assertEqual(len(state.epochs), 2)
            self.assertEqual(len(state.merged_sources), 1)

            # Synthetic WCS works with no background image
            geom = state.get_active_wcs()
            self.assertIsNotNone(geom)
            wcs, nx, ny, scale = geom
            self.assertGreaterEqual(nx, 1800)
            self.assertEqual(nx, ny)
            self.assertAlmostEqual(scale, 1.0)
            self.assertLess(wcs.wcs.cdelt[0], 0.0)
            self.assertGreater(wcs.wcs.cdelt[1], 0.0)

            # Toggle exclusion & survive slider change
            state.toggle_source_include(0)
            self.assertFalse(state.merged_sources[0].include)
            state.apply_show_all()
            self.assertEqual(len(state.merged_sources), 2)
            self.assertFalse(state.merged_sources[0].include)
            self.assertTrue(state.merged_sources[1].include)

            # Add and remove manual source
            state.add_manual_source(200.02, 29.98)
            self.assertEqual(len(state.merged_sources), 3)
            man_idx = next(
                i for i, s in enumerate(state.merged_sources) if s.is_manual
            )
            state.remove_manual_source(man_idx)
            self.assertEqual(len(state.merged_sources), 2)

            # Disable an epoch
            state.set_epoch_enabled(0, False)
            self.assertEqual(len(state.merged_sources), 1)

            # Background FITS loading & clearing
            bg = _write_dummy_cheeset(root / "bg_cheeset.fits")
            state.open_background_fits(bg)
            self.assertIsNotNone(state.bg_image)
            disp = state.get_display_flipped()
            self.assertIsNotNone(disp)
            self.assertEqual(disp.shape, (32, 32))
            state.clear_background_fits()
            self.assertIsNone(state.bg_image)

            # Remove epoch
            state.remove_epoch(1)
            self.assertEqual(len(state.epochs), 1)

    @unittest.skipUnless(
        _REAL_CHEESE_DIR.is_dir(),
        "Real 0923400201 cheese dataset not present on this machine",
    )
    def test_real_0923400201_cheese_roundtrip(self):
        cat = load_cheese_epoch(_REAL_CHEESE_DIR)
        self.assertEqual(cat.obs_id, "0923400201")
        self.assertGreater(len(cat.detections), 50)
        self.assertTrue({"mos1", "mos2", "pn"}.issubset(cat.sky_ellipses.keys()))
        self.assertTrue(any("cheeset.fits" in p.name for p in cat.candidate_images))

        # Check SRC 1 band-1 flux quirk is handled properly
        src1 = next(d for d in cat.detections if d.ml_id_src == 1)
        self.assertGreater(src1.flux, 1.0e-13)
        self.assertGreater(src1.bg_map, 0.0)

        merged = merge_catalogs([cat], filter_params=FilterParams())
        self.assertGreater(len(merged), 10)

        with tempfile.TemporaryDirectory() as tmp:
            out_fits = Path(tmp) / "merged_0923400201.fits"
            export_merged_emllist(merged, out_fits, template_catalog=cat)
            with fits.open(out_fits) as hdul:
                src = hdul["SRCLIST"]
                self.assertEqual(len(src.data), len(merged))
                self.assertTrue(np.all(src.data["ID_INST"] == 0))
                self.assertTrue(np.all(src.data["ID_BAND"] == 0))
                self.assertTrue(np.all(src.data["BG_MAP"] > 0.0))
                self.assertIn("TCRVAL1", src.header)

            # Ensure one of the discovered cheeset images loads cleanly via fits_io
            bg_img = load_fits_image(cat.candidate_images[0])
            self.assertIsNotNone(bg_img.wcs)

    def test_ra_wraparound_and_high_dec_merge_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cat1 = load_emllist(
                _write_synthetic_emllist(
                    root / "wrap1.fits",
                    [
                        {
                            "ra": 359.9,
                            "dec": 85.0,
                            "det_ml": 100.0,
                            "flux_summary": 2e-14,
                            "scts": 150.0,
                            "bg_map": 0.4,
                            "radec_err": 0.5,
                        },
                        {
                            "ra": 0.1,
                            "dec": 85.0,
                            "det_ml": 60.0,
                            "flux_summary": 1e-14,
                            "scts": 80.0,
                            "bg_map": 0.3,
                            "radec_err": 0.5,
                        },
                    ],
                    crval=(0.0, 85.0),
                )
            )
            cat2 = load_emllist(
                _write_synthetic_emllist(
                    root / "wrap2.fits",
                    [
                        {
                            "ra": 0.09,
                            "dec": 85.0,
                            "det_ml": 40.0,
                            "flux_summary": 1.2e-14,
                            "scts": 50.0,
                            "bg_map": 0.25,
                            "radec_err": 0.5,
                        },
                    ],
                    crval=(0.0, 85.0),
                )
            )
            merged = merge_catalogs(
                [cat1, cat2],
                filter_params=FilterParams(),
                match_radius_arcsec=5.0,
            )
            # At Dec=+85 deg, 0.2 deg in RA is ~62.8 arcsec -> 359.9 & 0.1 remain distinct,
            # while 0.1 & 0.09 (3.14") merge across Dec=85.
            self.assertEqual(len(merged), 2)
            # Merged RA of 0.1 and 0.09 should stay near 0.095, not jump to 180.
            self.assertLess(abs(merged[1].ra_deg - 0.095), 0.01)

            # Export with template_catalog=None: TCRVAL1 must use spherical mean near 0/360
            # (not arithmetic mean ~180 deg!) and X_IMA/Y_IMA must be finite & distinct.
            out_fits = root / "wrap_no_tpl.fits"
            export_merged_emllist(merged, out_fits, template_catalog=None)
            with fits.open(out_fits) as hdul:
                hdr = hdul["SRCLIST"].header
                data = hdul["SRCLIST"].data
                tcrval1 = float(hdr["TCRVAL1"])
                tcrval2 = float(hdr["TCRVAL2"])
                self.assertTrue(tcrval1 < 5.0 or tcrval1 > 355.0, msg=f"TCRVAL1={tcrval1}")
                self.assertAlmostEqual(tcrval2, 85.0, places=1)
                x_ima = np.asarray(data["X_IMA"], dtype=float)
                y_ima = np.asarray(data["Y_IMA"], dtype=float)
                self.assertTrue(np.all(np.isfinite(x_ima)))
                self.assertTrue(np.all(np.isfinite(y_ima)))
                self.assertNotAlmostEqual(float(x_ima[0]), float(x_ima[1]), places=2)

                # Verify X_IMA/Y_IMA match the output header WCS
                w_out = WCS(naxis=2)
                w_out.wcs.ctype = [hdr["TCTYPE1"], hdr["TCTYPE2"]]
                w_out.wcs.crpix = [hdr["TCRPIX1"], hdr["TCRPIX2"]]
                w_out.wcs.crval = [hdr["TCRVAL1"], hdr["TCRVAL2"]]
                w_out.wcs.cdelt = [hdr["TCDELT1"], hdr["TCDELT2"]]
                wx, wy = w_out.wcs_world2pix(data["RA"], data["DEC"], 1)
                np.testing.assert_allclose(x_ima, wx, atol=1e-2)
                np.testing.assert_allclose(y_ima, wy, atol=1e-2)

            # Export when template tangent point is >90 deg away from a manual source
            far_sources = [
                MergedSource(
                    merged_id=1,
                    ra_deg=20.0,
                    dec_deg=-60.0,
                    radec_err_arcsec=1.0,
                    det_ml=50.0,
                    flux=1e-14,
                    scts=100.0,
                    bg_map=0.5,
                    ext=0.0,
                    ext_ml=0.0,
                    dist_nn_arcsec=999.0,
                    include=True,
                    is_manual=True,
                    manual_id=1,
                ),
                MergedSource(
                    merged_id=2,
                    ra_deg=20.05,
                    dec_deg=-60.02,
                    radec_err_arcsec=1.0,
                    det_ml=40.0,
                    flux=1e-14,
                    scts=80.0,
                    bg_map=0.4,
                    ext=0.0,
                    ext_ml=0.0,
                    dist_nn_arcsec=999.0,
                    include=True,
                    is_manual=True,
                    manual_id=2,
                ),
            ]
            out_far = root / "far_recentered.fits"
            export_merged_emllist(far_sources, out_far, template_catalog=cat1)
            with fits.open(out_far) as hdul:
                hdr = hdul["SRCLIST"].header
                data = hdul["SRCLIST"].data
                self.assertAlmostEqual(float(hdr["TCRVAL1"]), 20.025, places=1)
                self.assertAlmostEqual(float(hdr["TCRVAL2"]), -60.01, places=1)
                self.assertTrue(np.all(np.isfinite(data["X_IMA"])))
                self.assertTrue(np.all(np.isfinite(data["Y_IMA"])))
                self.assertNotAlmostEqual(
                    float(data["X_IMA"][0]), float(data["X_IMA"][1]), places=2
                )

    def test_true_antichaining_with_high_precision_intermediate(self):
        # 0", 4", 8" with DET_ML = 100, 60, 25; intermediate 4" has tiny error so a
        # moving centroid would shift to ~4" and falsely chain 8" into the 0" cluster.
        cos_dec = math.cos(math.radians(30.0))
        step_4arc = (4.0 / 3600.0) / cos_dec
        step_8arc = (8.0 / 3600.0) / cos_dec

        def _det(ep: str, ml_id: int, ra: float, ml: float, err: float) -> SourceDetection:
            return SourceDetection(
                epoch_id=ep,
                epoch_label=ep,
                ml_id_src=ml_id,
                ra_deg=ra,
                dec_deg=30.0,
                radec_err_arcsec=err,
                det_ml=ml,
                flux=1e-14,
                scts=100.0,
                bg_map=0.5,
                ext=0.0,
                ext_ml=0.0,
                dist_nn_arcsec=100.0,
                rate=0.01,
                exp_map=1e4,
            )

        cat = EpochCatalog(
            epoch_id="ep_chain",
            label="ep_chain",
            source_path=Path("ep_chain.fits"),
            detections=[
                _det("ep_chain", 1, 200.0, 100.0, 2.0),
                _det("ep_chain", 2, 200.0 + step_4arc, 60.0, 0.05),
                _det("ep_chain", 3, 200.0 + step_8arc, 25.0, 0.5),
            ],
        )
        merged = merge_catalogs([cat], filter_params=FilterParams(), match_radius_arcsec=5.0)
        self.assertEqual(len(merged), 2)
        self.assertEqual([m.ml_id_src for m in merged[0].members], [1, 2])
        self.assertEqual([m.ml_id_src for m in merged[1].members], [3])

    def test_close_source_exclusion_and_higher_ml_epoch_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cos_dec = math.cos(math.radians(30.0))
            step_1arc = (1.0 / 3600.0) / cos_dec

            eml1 = _write_synthetic_emllist(
                root / "close1.fits",
                [
                    {"ra": 200.0, "dec": 30.0, "det_ml": 80.0, "flux_summary": 2e-14},
                    {
                        "ra": 200.0 + step_1arc,
                        "dec": 30.0,
                        "det_ml": 50.0,
                        "flux_summary": 1.5e-14,
                    },
                ],
                obs_id="OBS1",
            )
            eml2 = _write_synthetic_emllist(
                root / "close2.fits",
                [
                    # Higher DET_ML detection of source 1 in a later epoch
                    {"ra": 200.00005, "dec": 30.0, "det_ml": 150.0, "flux_summary": 3e-14},
                ],
                obs_id="OBS2",
            )

            state = SrclistViewerState(match_radius_arcsec=0.0)
            state.add_epoch(eml1)
            self.assertEqual(len(state.merged_sources), 2)

            # Toggle source 0 OFF; source 1 (only 1.0" away) must remain ON.
            state.toggle_source_include(0)
            self.assertFalse(state.merged_sources[0].include)
            self.assertTrue(state.merged_sources[1].include)

            # Recompute at match_radius=0.0 preserves independent exclusion.
            state.recompute_merged()
            self.assertFalse(state.merged_sources[0].include)
            self.assertTrue(state.merged_sources[1].include)

            # Load a higher-ML epoch matching source 0 at match_radius=0.5";
            # source 0 must remain excluded even though new epoch becomes seed,
            # while source 1 (1.0" away) remains separate and included.
            state.match_radius_arcsec = 0.5
            state.add_epoch(eml2)
            self.assertEqual(len(state.merged_sources), 2)
            # First merged source has ML=150 (from OBS2 + OBS1 #1) and stays excluded.
            self.assertAlmostEqual(state.merged_sources[0].det_ml, 150.0, places=3)
            self.assertFalse(state.merged_sources[0].include)
            self.assertTrue(state.merged_sources[1].include)

            # Toggling merged source 0 back ON clears all member exclusion keys.
            state.toggle_source_include(0)
            self.assertTrue(state.merged_sources[0].include)
            self.assertEqual(len(state.excluded_detection_keys), 0)

    def test_edge_cases_nan_missing_ids_zero_rows_scts_bg_and_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. NaN RA/Dec rejection, missing/NaN/non-positive ML_ID_SRC, SCTS fallback,
            #    and non-standard template column format ('E' for RA/DEC).
            bad_path = root / "malformed.fits"
            cols = [
                fits.Column(
                    name="ML_ID_SRC",
                    format="E",
                    array=np.array([1.0, 1.0, float("nan"), 0.0, 2.0], dtype=np.float32),
                ),
                fits.Column(
                    name="ID_INST",
                    format="J",
                    array=np.array([0, 2, 0, 0, 0], dtype=np.int32),
                ),
                fits.Column(
                    name="ID_BAND",
                    format="J",
                    array=np.array([0, 0, 0, 0, 0], dtype=np.int32),
                ),
                fits.Column(
                    name="RA",
                    format="E",
                    array=np.array([200.0, 200.0, 200.05, 200.1, float("nan")], dtype=np.float32),
                ),
                fits.Column(
                    name="DEC",
                    format="E",
                    array=np.array([30.0, 30.0, 30.02, 30.03, 30.04], dtype=np.float32),
                ),
                fits.Column(
                    name="DET_ML",
                    format="E",
                    array=np.array([40.0, 30.0, 35.0, 25.0, 90.0], dtype=np.float32),
                ),
                fits.Column(
                    name="FLUX",
                    format="E",
                    array=np.array([1e-14, 1e-14, 1e-14, 1e-14, 1e-14], dtype=np.float32),
                ),
                fits.Column(
                    name="SCTS",
                    format="E",
                    array=np.array([0.0, 88.0, -5.0, 0.0, 50.0], dtype=np.float32),
                ),
                fits.Column(
                    name="BG_MAP",
                    format="E",
                    array=np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                ),
                fits.Column(
                    name="RATE",
                    format="E",
                    array=np.array([0.0, 0.0, 0.01, 0.0, 0.0], dtype=np.float32),
                ),
                fits.Column(
                    name="EXP_MAP",
                    format="E",
                    array=np.array([0.0, 0.0, 5000.0, 0.0, 0.0], dtype=np.float32),
                ),
            ]
            tbl = fits.BinTableHDU.from_columns(cols, name="SRCLIST")
            fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(bad_path, overwrite=True)

            cat = load_emllist(bad_path)
            # Source 2 has NaN RA -> rejected; rows with NaN/0 ML_ID_SRC get negative IDs.
            self.assertEqual(len(cat.detections), 3)
            ids = [d.ml_id_src for d in cat.detections]
            self.assertEqual(ids[0], 1)
            self.assertLess(ids[1], 0)
            self.assertLess(ids[2], 0)
            self.assertNotEqual(ids[1], ids[2])
            # Source 1 summary SCTS was 0.0 -> falls back to row 1 SCTS=88.0
            self.assertAlmostEqual(cat.detections[0].scts, 88.0, places=4)
            # Negative/zero SCTS rows stay 0.0 in load_emllist
            self.assertEqual(cat.detections[1].scts, 0.0)
            self.assertEqual(cat.detections[2].scts, 0.0)

            # Singleton automatic source gets DIST_NN == 999.0
            single = merge_catalogs(
                [cat],
                filter_params=FilterParams(min_det_ml=38.0),
            )
            self.assertEqual(len(single), 1)
            self.assertFalse(single[0].is_manual)
            self.assertEqual(single[0].dist_nn_arcsec, 999.0)

            # Merge all 3: source 2 derives SCTS from rate*exp_map=50.0, source 3 uses floor 1.0
            merged = merge_catalogs([cat], filter_params=FilterParams())
            self.assertEqual(len(merged), 3)
            self.assertAlmostEqual(merged[0].scts, 88.0, places=4)
            self.assertAlmostEqual(merged[1].scts, 50.0, places=3)
            self.assertAlmostEqual(merged[2].scts, 1.0, places=4)
            self.assertTrue(all(s.bg_map > 0.0 for s in merged))

            # Force 0 SCTS & BG_MAP before export -> export must enforce > 0
            merged[0].scts = 0.0
            merged[0].bg_map = -1.0
            out_fits = root / "enforced.fits"
            export_merged_emllist(merged, out_fits, template_catalog=cat)
            with fits.open(out_fits) as hdul:
                srclist = hdul["SRCLIST"]
                self.assertTrue(np.all(srclist.data["SCTS"] > 0.0))
                self.assertTrue(np.all(srclist.data["BG_MAP"] > 0.0))
                # Mandatory column formats enforced even though template had 'E' for RA/DEC/ML_ID_SRC
                fmt_by_name = {c.name: c.format for c in srclist.columns}
                self.assertEqual(fmt_by_name["RA"], "D")
                self.assertEqual(fmt_by_name["DEC"], "D")
                self.assertEqual(fmt_by_name["ML_ID_SRC"], "J")
                self.assertEqual(fmt_by_name["ID_INST"], "J")
                self.assertEqual(fmt_by_name["ID_BAND"], "J")

            # Manual source validation for scts > 0 and bg_map > 0
            state = SrclistViewerState()
            msrc = state.add_manual_source(200.0, 30.0, scts=-5.0, bg_map=0.0)
            self.assertGreater(msrc.scts, 0.0)
            self.assertGreater(msrc.bg_map, 0.0)

            # 2. 0-row input emllist.fits
            empty_in = root / "empty_in.fits"
            _write_synthetic_emllist(empty_in, [])
            empty_cat = load_emllist(empty_in)
            self.assertEqual(len(empty_cat.detections), 0)

            # 3. Missing ML_ID_SRC column assigns unique negative IDs
            no_id_path = root / "no_id.fits"
            tbl_no_id = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="D", array=np.array([200.0, 200.1])),
                    fits.Column(name="DEC", format="D", array=np.array([30.0, 30.1])),
                    fits.Column(name="DET_ML", format="E", array=np.array([30.0, 40.0], dtype=np.float32)),
                ],
                name="SRCLIST",
            )
            fits.HDUList([fits.PrimaryHDU(), tbl_no_id]).writeto(no_id_path, overwrite=True)
            no_id_cat = load_emllist(no_id_path)
            self.assertEqual([d.ml_id_src for d in no_id_cat.detections], [-1, -2])

    def test_load_sky_region_ellipses_skips_unsupported_and_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed-bkgregtsky.fits"
            shapes = np.asarray(["!BOX", "!CIRCLE", "!ELLIPSE", "!CIRCLE"], dtype="U16")
            x = np.array(
                [
                    [25921.0, 0.0],
                    [25921.0, 0.0],
                    [float("nan"), 0.0],
                    [25921.0, 0.0],
                ],
                dtype=np.float32,
            )
            y = np.array(
                [
                    [25921.0, 0.0],
                    [25921.0, 0.0],
                    [25921.0, 0.0],
                    [25921.0, 0.0],
                ],
                dtype=np.float32,
            )
            r = np.array(
                [
                    [100.0, 100.0],  # !BOX -> unsupported
                    [0.0, 0.0],  # !CIRCLE with 0 radius -> malformed
                    [200.0, 100.0],  # !ELLIPSE with NaN X -> malformed
                    [300.0, 0.0],  # valid !CIRCLE
                ],
                dtype=np.float32,
            )
            tbl = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="SHAPE", format="16A", array=shapes),
                    fits.Column(name="X", format="2E", array=x),
                    fits.Column(name="Y", format="2E", array=y),
                    fits.Column(name="R", format="2E", array=r),
                ],
                name="REGION",
            )
            tbl.header["TCTYP2"] = "RA---TAN"
            tbl.header["TCTYP3"] = "DEC--TAN"
            tbl.header["TCRPX2"] = 25921.0
            tbl.header["TCRPX3"] = 25921.0
            tbl.header["TCRVL2"] = 200.0
            tbl.header["TCRVL3"] = 30.0
            tbl.header["TCDLT2"] = -1.38888888888889e-5
            tbl.header["TCDLT3"] = 1.38888888888889e-5
            fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(path, overwrite=True)

            ells = load_sky_region_ellipses(path, instrument="pn")
            self.assertEqual(len(ells), 1)
            self.assertAlmostEqual(ells[0].semi_major_arcsec, 15.0, places=3)

            # Single-element R column with ELLIPSE shape -> skipped cleanly (r_arr.size < 2)
            bad_ell = Path(tmp) / "bad_ell-bkgregtsky.fits"
            tbl2 = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="SHAPE", format="16A", array=np.asarray(["!ELLIPSE"], dtype="U16")),
                    fits.Column(name="X", format="E", array=np.array([25921.0], dtype=np.float32)),
                    fits.Column(name="Y", format="E", array=np.array([25921.0], dtype=np.float32)),
                    fits.Column(name="R", format="E", array=np.array([200.0], dtype=np.float32)),
                ],
                name="REGION",
            )
            tbl2.header.update(tbl.header)
            fits.HDUList([fits.PrimaryHDU(), tbl2]).writeto(bad_ell, overwrite=True)
            self.assertEqual(load_sky_region_ellipses(bad_ell, instrument="pn"), [])

    def test_imgui_bundle_signatures_match_gui_calls(self):
        try:
            from imgui_bundle import imgui
        except ImportError:
            self.skipTest("imgui_bundle not installed")

        poly_doc = imgui.ImDrawList.add_polyline.__doc__ or ""
        self.assertIn("thickness: float", poly_doc)
        self.assertIn("flags: int", poly_doc)
        self.assertLess(poly_doc.index("thickness: float"), poly_doc.index("flags: int"))

        key_doc = imgui.is_key_pressed.__doc__ or ""
        self.assertIn("repeat: bool", key_doc)

        gui_src = (
            Path(__file__).resolve().parents[1] / "src/xmm_tools/srclist/gui.py"
        ).read_text(encoding="utf-8")
        # Ensure add_polyline passes thickness float before ImDrawFlags_.closed
        self.assertRegex(
            gui_src,
            re.compile(r"add_polyline\(\s*scr,\s*ell_col,\s*1\.1,\s*imgui\.ImDrawFlags_\.closed"),
        )
        self.assertIn("not imgui.get_io().want_text_input", gui_src)
        self.assertIn("imgui.is_key_pressed(imgui.Key.space, False)", gui_src)


if __name__ == "__main__":
    unittest.main()
