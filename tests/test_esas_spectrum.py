"""Tests for ESAS spectrum extraction (mosspectra/pnspectra + mosback/pnback)."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astropy.io import fits
import numpy as np

from xmm_tools.esas import (
    CheeseResult,
    FlareResult,
    load_spectrum_result,
    run_spectrum,
)
from xmm_tools.exec.runner import RunResult
from xmm_tools.pipeline import AnaConfig, SpectrumRegionSpec, render_template_toml


def _make_dummy_sky_fits(path: Path) -> None:
    hdu = fits.PrimaryHDU(np.zeros((10, 10), dtype=np.float32))
    hdr = hdu.header
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 287.9565
    hdr["CRVAL2"] = 4.9827
    hdr["CRPIX1"] = 5.0
    hdr["CRPIX2"] = 5.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdu.writeto(path, overwrite=True)


def _make_dummy_events_fits(path: Path, n_rows: int) -> None:
    col = fits.Column(name="PI", format="J", array=np.arange(n_rows, dtype=np.int32))
    tbl = fits.BinTableHDU.from_columns([col], name="EVENTS")
    fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(path, overwrite=True)


def _touch(path: Path, content: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestESASSpectrum(unittest.TestCase):
    def _setup_workspace(self, root: Path) -> tuple[AnaConfig, FlareResult, CheeseResult]:
        toml_path = root / "esas.toml"
        toml_path.write_text(
            render_template_toml(
                workdir=".",
                sas_ccf="/data/ccf.cif",
                sas_ccfpath="/ccf/valid",
                sas_odf="/data/SUM.SAS",
            ),
            encoding="utf-8",
        )
        cfg = AnaConfig.from_toml(toml_path)
        cfg.spectrum.elow = 500
        cfg.spectrum.ehigh = 2000

        fdir = cfg.paths.resolve(cfg.paths.flare_dir)
        cdir = cfg.paths.resolve(cfg.paths.cheese_dir)
        clean_events = {
            "mos1": _touch(fdir / "mos1S001-allevc.fits"),
            "mos2": _touch(fdir / "mos2S002-allevc.fits"),
            "pn": _touch(fdir / "pnS003-allevc.fits"),
            "pn_oot": _touch(fdir / "pnS003-allevcoot.fits"),
        }
        flare = FlareResult(
            flare_dir=fdir,
            manifest_path=fdir / "manifest.json",
            clean_events=clean_events,
            flag_filtered=True,
        )
        cheese_masks = {
            "mos1": _touch(cdir / "mos1S001-cheeset.fits"),
            "mos2": _touch(cdir / "mos2S002-cheeset.fits"),
            "pn": _touch(cdir / "pnS003-cheeset.fits"),
        }
        region_det = {
            "mos1": _touch(cdir / "mos1S001-bkgregtdet.fits"),
            "mos2": _touch(cdir / "mos2S002-bkgregtdet.fits"),
            "pn": _touch(cdir / "pnS003-bkgregtdet.fits"),
        }
        region_sky = {
            "mos1": _touch(cdir / "mos1S001-bkgregtsky.fits"),
            "mos2": _touch(cdir / "mos2S002-bkgregtsky.fits"),
            "pn": _touch(cdir / "pnS003-bkgregtsky.fits"),
        }
        mask_res = CheeseResult(
            cheese_dir=cdir,
            manifest_path=cdir / "manifest.json",
            method="cheese",
            cheese_masks=cheese_masks,
            region_det=region_det,
            region_sky=region_sky,
        )
        return cfg, flare, mask_res

    def _fake_runner_factory(
        self,
        calls: list[tuple[str, Path, dict]],
        *,
        zero_pn_for: set[str] | None = None,
    ):
        zero_pn_for = zero_pn_for or set()

        def fake_run(self_runner, cmd, *, force=False):
            calls.append((cmd.program, cmd.cwd_path, dict(cmd.params)))
            cmd.cwd_path.mkdir(parents=True, exist_ok=True)
            log_path = cmd.cwd_path / cmd.log_file if cmd.log_file else None

            if cmd.program == "evselect" and "imageset" in cmd.params:
                _make_dummy_sky_fits(cmd.cwd_path / cmd.params["imageset"])
            elif cmd.program == "ecoordconv":
                log_path.write_text("DETX: DETY: 123.4 -56.7\n", encoding="utf-8")
            elif cmd.program == "evselect" and "filteredset" in cmd.params:
                out_fits = cmd.cwd_path / cmd.params["filteredset"]
                n_rows = 0 if (
                    cmd.cwd_path.name in zero_pn_for and "pn" in cmd.params["table"]
                ) else 42
                _make_dummy_events_fits(out_fits, n_rows)
            elif cmd.program == "mosspectra":
                inst = "mos1S001" if "mos1" in cmd.params["eventfile"] else "mos2S002"
                _touch(cmd.cwd_path / f"{inst}-fovt.pi")
                _touch(cmd.cwd_path / f"{inst}-fovimspdet.fits")
                _touch(cmd.cwd_path / f"{inst}.rmf")
                _touch(cmd.cwd_path / f"{inst}.arf")
            elif cmd.program == "mosback":
                prefix = cmd.params["inspecfile"].replace("-fovt.pi", "")
                _touch(cmd.cwd_path / f"{prefix}-bkg.pi")
                _touch(cmd.cwd_path / f"{prefix}-bkgspec.qdp")
            elif cmd.program == "pnspectra":
                _touch(cmd.cwd_path / "pnS003-fovt.pi")
                _touch(cmd.cwd_path / "pnS003-fovimspdet.fits")
                _touch(cmd.cwd_path / "pnS003-fovtoot.pi")
                _touch(cmd.cwd_path / "pnS003.rmf")
                _touch(cmd.cwd_path / "pnS003.arf")
            elif cmd.program == "pnback":
                _touch(cmd.cwd_path / "pnS003-bkg.pi")
                _touch(cmd.cwd_path / "pnS003-fovtootsub.pi")
                _touch(cmd.cwd_path / "pnS003-bkgspec.qdp")
            elif cmd.program == "protonscale":
                log_path.write_text(
                    "protonscale:-  Area             :    6.25\n"
                    "protonscale:-  Ave Flux         :    1.5E-05\n"
                    "protonscale:-  Scaled Ave Flux  :    10.5\n",
                    encoding="utf-8",
                )

            return RunResult(
                command=cmd, status="completed", returncode=0, log_path=log_path
            )

        return fake_run

    def test_split_union_and_zero_event_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare, mask_res = self._setup_workspace(root)

            # Split region with two boxes; e2 has 0 pn events
            split_reg = root / "east_jet.reg"
            split_reg.write_text(
                "# Region file format: DS9 version 4.1\n"
                "fk5\n"
                "box(287.9565,4.9827,60\",30\",15) # text={e1}\n"
                "box(287.9700,4.9850,60\",30\",15) # text={e2}\n",
                encoding="utf-8",
            )
            # Union region with two ellipses
            union_reg = root / "bkg.reg"
            union_reg.write_text(
                "# Region file format: DS9 version 4.1\n"
                "fk5\n"
                "ellipse(287.90,4.95,120\",80\",30)\n"
                "circle(288.00,5.02,90\")\n",
                encoding="utf-8",
            )

            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="east_jet", file=str(split_reg), mode="split"),
                SpectrumRegionSpec(name="background", file=str(union_reg), mode="union"),
            ]

            calls: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls, zero_pn_for={"e2"}),
            ):
                res = run_spectrum(cfg, flare=flare, mask_res=mask_res)

            self.assertEqual(
                set(res.regions.keys()),
                {"east_jet/e1", "east_jet/e2", "background"},
            )
            # e2 pn was skipped due to 0 events
            e2_pn = res.regions["east_jet/e2"].instruments["pn"]
            self.assertTrue(e2_pn.skipped)
            self.assertEqual(e2_pn.event_count, 0)
            self.assertIsNone(e2_pn.fovt)

            # background union expression contains '||'
            bkg_expr = (res.regions["background"].region_dir / "reg_mos1.txt").read_text()
            self.assertIn("||", bkg_expr)
            self.assertTrue(bkg_expr.startswith("&&("))

            # Verify maskdet / masksky from CheeseResult.region_det / region_sky were passed
            mos_calls = [p for prog, _, p in calls if prog == "mosspectra"]
            pn_calls = [p for prog, _, p in calls if prog == "pnspectra"]
            self.assertTrue(mos_calls and all("maskdet" in p and "masksky" in p for p in mos_calls))
            self.assertTrue(pn_calls and all("maskdet" in p and "masksky" in p for p in pn_calls))

            # Manifest roundtrip
            loaded = load_spectrum_result(cfg)
            self.assertEqual(set(loaded.regions.keys()), set(res.regions.keys()))
            self.assertTrue(
                loaded.regions["east_jet/e1"].instruments["pn"].fovtootsub.exists()
            )
            self.assertAlmostEqual(
                loaded.regions["east_jet/e1"].instruments["mos1"].proton_scale, 10.5
            )

    def test_incremental_upsert_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare, mask_res = self._setup_workspace(root)

            reg1 = root / "r1.reg"
            reg1.write_text("fk5\ncircle(287.95,4.98,60\")\n", encoding="utf-8")
            reg2 = root / "r2.reg"
            reg2.write_text("fk5\ncircle(287.96,4.99,60\")\n", encoding="utf-8")

            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single"),
            ]
            calls1: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls1),
            ):
                res1 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(list(res1.regions.keys()), ["r1"])

            # Add r2, keep r1 overwrite=False -> r1 must be skipped, r2 appended
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single", overwrite=False),
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single", overwrite=False),
            ]
            calls2: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls2),
            ):
                res2 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(set(res2.regions.keys()), {"r1", "r2"})
            r1_dirs = [cwd for _, cwd, _ in calls2 if cwd.name == "r1"]
            self.assertEqual(r1_dirs, [])

            # Set r1 overwrite=True -> r1 re-runs while r2 stays in manifest
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single", overwrite=True),
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single", overwrite=False),
            ]
            calls3: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls3),
            ):
                res3 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(set(res3.regions.keys()), {"r1", "r2"})
            r1_dirs_after = [cwd for _, cwd, _ in calls3 if cwd.name == "r1"]
            r2_dirs_after = [cwd for _, cwd, _ in calls3 if cwd.name == "r2"]
            self.assertTrue(len(r1_dirs_after) > 0)
            self.assertEqual(r2_dirs_after, [])

    def test_dry_run_skips_disk_writes_and_handles_ellipse_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare, mask_res = self._setup_workspace(root)

            reg = root / "mixed.reg"
            reg.write_text(
                "# Region file format: DS9 version 4.1\n"
                "fk5\n"
                "ellipse(287.90,4.95,120\",80\",30) # text={ell}\n"
                "box(287.9565,4.9827,60\",30\",15) # text={box1}\n",
                encoding="utf-8",
            )
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="mixed", file=str(reg), mode="split"),
            ]

            res = run_spectrum(cfg, flare=flare, mask_res=mask_res, dry_run=True)
            self.assertEqual(set(res.regions.keys()), {"mixed/ell", "mixed/box1"})
            for r in res.regions.values():
                self.assertFalse(r.region_dir.exists())
            self.assertFalse(res.manifest_path.exists())

    def test_regions_path_fallback_on_import_error(self):
        import builtins
        import sys
        from xmm_tools.esas.spectrum import _import_regions

        with tempfile.TemporaryDirectory() as tmp:
            dep_dir = Path(tmp) / "deps"
            pkg_dir = dep_dir / "regions"
            pkg_dir.mkdir(parents=True)
            (pkg_dir / "__init__.py").write_text(
                "CircleSkyRegion = object\n"
                "EllipseSkyRegion = object\n"
                "PolygonSkyRegion = object\n"
                "RectangleSkyRegion = object\n"
                "Regions = object\n",
                encoding="utf-8",
            )

            real_import = builtins.__import__
            saved_regions = sys.modules.pop("regions", None)
            saved_path = list(sys.path)
            fail_once = {"flag": True}

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "regions" and fail_once["flag"]:
                    fail_once["flag"] = False
                    raise ImportError("simulated missing regions")
                return real_import(name, globals, locals, fromlist, level)

            try:
                with patch("builtins.__import__", side_effect=guarded_import):
                    res = _import_regions(dep_dir)
                self.assertEqual(len(res), 5)
                self.assertIn(str(dep_dir), sys.path)
            finally:
                sys.path[:] = saved_path
                sys.modules.pop("regions", None)
                if saved_regions is not None:
                    sys.modules["regions"] = saved_regions
