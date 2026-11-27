"""Tests for ESAS spectrum extraction (mosspectra/pnspectra + mosback/pnback)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from xmm_tools.config import AnaConfig, SpectrumRegionSpec, render_template_toml
from xmm_tools.esas import (
    CheeseResult,
    FlareResult,
    load_spectrum_result,
    run_spectrum,
)
from xmm_tools.exec import MANIFEST_SCHEMA_VERSION, RunResult


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
            workspace_root=root,
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
            workspace_root=root,
        )
        return cfg, flare, mask_res

    def _fake_runner_factory(
        self,
        calls: list[tuple[str, Path, dict]],
        *,
        zero_pn_for: set[str] | None = None,
        fail_region: str | None = None,
    ):
        zero_pn_for = zero_pn_for or set()

        def fake_run(self_runner, cmd, *, force=False):
            calls.append((cmd.program, cmd.cwd_path, dict(cmd.params)))
            if fail_region and fail_region in cmd.cwd_path.parts and cmd.program == "mosspectra":
                raise RuntimeError(f"boom in {fail_region}")
            cmd.cwd_path.mkdir(parents=True, exist_ok=True)
            log_path = cmd.cwd_path / cmd.log_file if cmd.log_file else None

            if cmd.program == "evselect" and "imageset" in cmd.params:
                _make_dummy_sky_fits(cmd.cwd_path / cmd.params["imageset"])
            elif cmd.program == "ecoordconv":
                log_path.write_text("DETX: DETY: 123.4 -56.7\n", encoding="utf-8")
            elif cmd.program == "evselect" and "filteredset" in cmd.params:
                out_fits = cmd.cwd_path / cmd.params["filteredset"]
                n_rows = 0 if (
                    cmd.cwd_path.parent.name in zero_pn_for and "pn" in cmd.params["table"]
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

            split_reg = root / "east_jet.reg"
            split_reg.write_text(
                "# Region file format: DS9 version 4.1\n"
                "fk5\n"
                "box(287.9565,4.9827,60\",30\",15) # text={e1}\n"
                "box(287.9700,4.9850,60\",30\",15) # text={e2}\n",
                encoding="utf-8",
            )
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
            e2_pn = res.regions["east_jet/e2"].instruments["pn"]
            self.assertTrue(e2_pn.skipped)
            self.assertEqual(e2_pn.event_count, 0)
            self.assertIsNone(e2_pn.fovt)

            bkg_expr = (res.regions["background"].region_dir / "reg_mos1.txt").read_text()
            self.assertIn("||", bkg_expr)
            self.assertTrue(bkg_expr.startswith("&&("))

            mos_calls = [p for prog, _, p in calls if prog == "mosspectra"]
            pn_calls = [p for prog, _, p in calls if prog == "pnspectra"]
            self.assertTrue(mos_calls and all("maskdet" in p and "masksky" in p for p in mos_calls))
            self.assertTrue(pn_calls and all("maskdet" in p and "masksky" in p for p in pn_calls))

            # Verify v2 manifest records commands and zero host path leakage
            raw_manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(raw_manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(raw_manifest["stage"], "spectrum")
            self.assertIn("ref_sky_image:mos1", raw_manifest["steps"])
            self.assertIn("region:east_jet/e1:spectra:mos1", raw_manifest["steps"])
            self.assertIn("region:east_jet/e2:count_events:pn", raw_manifest["steps"])
            for step_raw in raw_manifest["steps"].values():
                self.assertNotIn(str(root), json.dumps({k: v for k, v in step_raw.items() if k != "shell"}))
            self.assertEqual(raw_manifest["products"]["regions"]["background"]["source_file"], "../bkg.reg")

            # Re-running unchanged reuses 0-event checks & spectra with 0 SAS calls
            calls.clear()
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls, zero_pn_for={"e2"}),
            ):
                res_reused = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(calls, [])
            self.assertEqual(res_reused.manifest.status, "reused")

            loaded = load_spectrum_result(cfg)
            self.assertEqual(set(loaded.regions.keys()), set(res.regions.keys()))
            self.assertTrue(loaded.regions["east_jet/e1"].instruments["pn"].fovtootsub.exists())
            self.assertAlmostEqual(loaded.regions["east_jet/e1"].instruments["mos1"].proton_scale, 10.5)

    def test_incremental_upsert_overwrite_reg_invalidation_and_active_tracking(self):
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
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._fake_runner_factory(calls1)):
                res1 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(list(res1.regions.keys()), ["r1"])

            # Add r2, keep r1 overwrite=False -> r1 reused (0 SAS calls), r2 extracted
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single", overwrite=False),
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single", overwrite=False),
            ]
            calls2: list[tuple[str, Path, dict]] = []
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._fake_runner_factory(calls2)):
                res2 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(set(res2.regions.keys()), {"r1", "r2"})
            r1_calls = [cwd for _, cwd, _ in calls2 if cwd.parent.name == "r1"]
            self.assertEqual(r1_calls, [])

            # Set r1 overwrite=True -> r1 re-runs while r2 stays reused
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single", overwrite=True),
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single", overwrite=False),
            ]
            calls3: list[tuple[str, Path, dict]] = []
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._fake_runner_factory(calls3)):
                res3 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertEqual(set(res3.regions.keys()), {"r1", "r2"})
            r1_calls_after = [cwd for _, cwd, _ in calls3 if cwd.parent.name == "r1"]
            r2_calls_after = [cwd for _, cwd, _ in calls3 if cwd.parent.name == "r2"]
            self.assertTrue(len(r1_calls_after) > 0)
            self.assertEqual(r2_calls_after, [])

            # Modifying r1.reg content automatically invalidates r1 even with overwrite=False
            reg1.write_text("fk5\ncircle(287.95,4.98,90\")\n", encoding="utf-8")
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single", overwrite=False),
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single", overwrite=False),
            ]
            calls4: list[tuple[str, Path, dict]] = []
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._fake_runner_factory(calls4)):
                run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertTrue(any(cwd.parent.name == "r1" for _, cwd, _ in calls4))
            self.assertFalse(any(cwd.parent.name == "r2" for _, cwd, _ in calls4))

            # Removing r1 from ana.toml marks r1 active=False while preserving r2 active=True
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single", overwrite=False),
            ]
            calls5: list[tuple[str, Path, dict]] = []
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._fake_runner_factory(calls5)):
                res5 = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertFalse(res5.regions["r1"].active)
            self.assertTrue(res5.regions["r2"].active)

            # Subset (--only r2) preserves r1 active=False; ad-hoc run keeps existing active flags
            reloaded = load_spectrum_result(cfg)
            self.assertFalse(reloaded.regions["r1"].active)
            self.assertTrue(reloaded.regions["r2"].active)

    def test_extra_params_validation_and_partial_failure_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare, mask_res = self._setup_workspace(root)
            reg1 = root / "r1.reg"
            reg1.write_text("fk5\ncircle(287.95,4.98,60\")\n", encoding="utf-8")
            reg2 = root / "r2.reg"
            reg2.write_text("fk5\ncircle(287.96,4.99,60\")\n", encoding="utf-8")
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single"),
                SpectrumRegionSpec(name="r2", file=str(reg2), mode="single"),
            ]

            for attr, bad in (
                ("spectra_extra_params", {"eventfile": "x.fits"}),
                ("spectra_extra_params", {"bad": "/abs/path"}),
                ("back_extra_params", {"inspecfile": "x.pi"}),
                ("back_extra_params", {"bad": Path("/tmp/x")}),
            ):
                bad_cfg = AnaConfig.from_toml(root / "esas.toml")
                bad_cfg.spectrum.regions = list(cfg.spectrum.regions)
                setattr(bad_cfg.spectrum, attr, bad)
                with self.assertRaises(ValueError):
                    run_spectrum(bad_cfg, flare=flare, mask_res=mask_res, dry_run=True)

            # Partial failure persists succeeded region with status='partial'
            calls: list[tuple[str, Path, dict]] = []
            with (
                patch(
                    "xmm_tools.exec.runner.SASRunner.run",
                    new=self._fake_runner_factory(calls, fail_region="r2"),
                ),
                self.assertRaises(RuntimeError),
            ):
                run_spectrum(cfg, flare=flare, mask_res=mask_res)

            partial = load_spectrum_result(cfg)
            self.assertEqual(partial.manifest.status, "partial")
            self.assertIn("r1", partial.regions)

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

    def test_withsrcrem_requires_nonempty_cheese_masks(self):
        from xmm_tools.esas.spectrum.region_runner import _resolve_region_masks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, mask_res = self._setup_workspace(root)
            with self.assertRaises(ValueError):
                _resolve_region_masks(None, "mos1", withsrcrem=True)

            # Empty mask file fails
            mask_res.region_det["mos1"].write_text("", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                _resolve_region_masks(mask_res, "mos1", withsrcrem=True)

            # Missing region_sky fails
            mask_res.region_det["mos1"].write_text("ok", encoding="utf-8")
            mask_res.region_sky.pop("mos1")
            with self.assertRaises(FileNotFoundError):
                _resolve_region_masks(mask_res, "mos1", withsrcrem=True)

            # withsrcrem=False ignores missing masks cleanly
            self.assertEqual(_resolve_region_masks(None, "mos1", withsrcrem=False), (None, None))

    def test_expand_region_specs_rejects_traversal_and_cross_mode_collisions(self):
        from xmm_tools.esas.spectrum.regions import expand_region_specs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _, _ = self._setup_workspace(root)
            sdir = cfg.paths.resolve(cfg.paths.spectrum_dir)

            reg_split = root / "a.reg"
            reg_split.write_text(
                "fk5\ncircle(287.95,4.98,60\") # text={b}\n", encoding="utf-8"
            )
            reg_single = root / "ab.reg"
            reg_single.write_text(
                "fk5\ncircle(287.96,4.99,60\")\n", encoding="utf-8"
            )

            # Collision: split group="a" + label="b" -> "a/b" vs single name="a/b"
            with self.assertRaises(ValueError):
                expand_region_specs(
                    [
                        SpectrumRegionSpec(name="a", file=str(reg_split), mode="split"),
                        SpectrumRegionSpec(name="a/b", file=str(reg_single), mode="single"),
                    ],
                    cfg,
                    sdir,
                )

            # Path traversal / containment violations
            for bad_name in ("../escape", ".", "/tmp/abs_reg"):
                with self.assertRaises(ValueError):
                    expand_region_specs(
                        [SpectrumRegionSpec(name=bad_name, file=str(reg_single), mode="single")],
                        cfg,
                        sdir,
                    )

    def test_protonscale_regex_and_validation_and_qdp_cleanup(self):
        from xmm_tools.esas.spectrum.commands import parse_protonscale_log
        from xmm_tools.esas.spectrum.region_runner import run_single_region_job
        from xmm_tools.esas.spectrum.regions import expand_region_specs
        from xmm_tools.exec import SASCommandError, SASRunner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "ps.log"
            # Missing Ave Flux line must NOT match Scaled Ave Flux into ave_flux
            log.write_text(
                "protonscale:-  Area             :    6.25\n"
                "protonscale:-  Scaled Ave Flux  :    10.5\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_protonscale_log(log), (6.25, None, 10.5))

            cfg, flare, mask_res = self._setup_workspace(root)
            reg1 = root / "r1.reg"
            reg1.write_text("fk5\ncircle(287.95,4.98,60\")\n", encoding="utf-8")
            cfg.spectrum.regions = [SpectrumRegionSpec(name="r1", file=str(reg1), mode="single")]
            sdir = cfg.paths.resolve(cfg.paths.spectrum_dir)
            job = expand_region_specs(cfg.spectrum.regions, cfg, sdir)[0]

            # Unparsable protonscale log raises SASCommandError
            def bad_ps_run(self_runner, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                log_path = cmd.cwd_path / cmd.log_file
                if cmd.program == "evselect" and "imageset" in cmd.params:
                    _make_dummy_sky_fits(cmd.cwd_path / cmd.params["imageset"])
                elif cmd.program == "ecoordconv":
                    log_path.write_text("DETX: DETY: 123.4 -56.7\n", encoding="utf-8")
                elif cmd.program == "evselect":
                    _make_dummy_events_fits(cmd.cwd_path / cmd.params["filteredset"], 10)
                elif cmd.program == "mosspectra":
                    inst = "mos1S001" if "mos1" in cmd.params["eventfile"] else "mos2S002"
                    for suf in ("-fovt.pi", "-fovimspdet.fits", ".rmf", ".arf"):
                        _touch(cmd.cwd_path / f"{inst}{suf}")
                elif cmd.program == "mosback":
                    prefix = cmd.params["inspecfile"].replace("-fovt.pi", "")
                    _touch(cmd.cwd_path / f"{prefix}-bkg.pi")
                elif cmd.program == "protonscale":
                    log_path.write_text("corrupted protonscale output\n", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0, log_path=log_path)

            ref_img = sdir / "_work" / "ref_sky" / "mos1" / "ref_mos1.fits"
            ref_img.parent.mkdir(parents=True, exist_ok=True)
            _make_dummy_sky_fits(ref_img)

            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=bad_ps_run),
                self.assertRaises(SASCommandError),
            ):
                run_single_region_job(
                    job,
                    workspace_root=root,
                    spec_cfg=cfg.spectrum,
                    spectra_extra={},
                    back_extra={},
                    clean_events={"mos1": flare.clean_events["mos1"]},
                    mask_res=mask_res,
                    ref_images={"mos1": ref_img},
                    cal_inputs={},
                    runner=SASRunner(env=cfg.env),
                    dry_run=False,
                    force=False,
                )

            # Missing fovimspdet with keepinterfiles=True raises SASCommandError,
            # while keepinterfiles=False skips protonscale cleanly
            def no_fovimspdet_run(self_runner, cmd, *, force=False):
                res = bad_ps_run(self_runner, cmd, force=force)
                if cmd.program == "mosspectra":
                    for f in cmd.cwd_path.glob("*-fovimspdet.fits"):
                        f.unlink()
                return res

            cfg.spectrum.keepinterfiles = True
            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=no_fovimspdet_run),
                self.assertRaises(SASCommandError),
            ):
                run_single_region_job(
                    job,
                    workspace_root=root,
                    spec_cfg=cfg.spectrum,
                    spectra_extra={},
                    back_extra={},
                    clean_events={"mos1": flare.clean_events["mos1"]},
                    mask_res=mask_res,
                    ref_images={"mos1": sdir / "_work" / "ref_sky" / "mos1" / "ref_mos1.fits"},
                    cal_inputs={},
                    runner=SASRunner(env=cfg.env),
                    dry_run=False,
                    force=True,
                )

            cfg.spectrum.keepinterfiles = False
            with patch("xmm_tools.exec.runner.SASRunner.run", new=no_fovimspdet_run):
                reg_no_ps, _ = run_single_region_job(
                    job,
                    workspace_root=root,
                    spec_cfg=cfg.spectrum,
                    spectra_extra={},
                    back_extra={},
                    clean_events={"mos1": flare.clean_events["mos1"]},
                    mask_res=mask_res,
                    ref_images={"mos1": sdir / "_work" / "ref_sky" / "mos1" / "ref_mos1.fits"},
                    cal_inputs={},
                    runner=SASRunner(env=cfg.env),
                    dry_run=False,
                    force=True,
                )
            self.assertIsNone(reg_no_ps.instruments["mos1"].proton_scale)
            cfg.spectrum.keepinterfiles = True

            # Toggling withplotfiles=True -> False cleans stale QDP files & symlinks
            calls: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls),
            ):
                res_with_qdp = run_spectrum(cfg, flare=flare, mask_res=mask_res, force=True)
            r1_dir = res_with_qdp.regions["r1"].region_dir
            self.assertTrue((r1_dir / "mos1S001-bkgspec.qdp").is_symlink())

            cfg.spectrum.withplotfiles = False

            def no_qdp_back_run(self_runner, cmd, *, force=False):
                res = self._fake_runner_factory(calls)(self_runner, cmd, force=force)
                if cmd.program in ("mosback", "pnback"):
                    for qdp in cmd.cwd_path.glob("*-bkgspec.qdp"):
                        qdp.unlink()
                return res

            with patch("xmm_tools.exec.runner.SASRunner.run", new=no_qdp_back_run):
                res_no_qdp = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertFalse((r1_dir / "mos1S001-bkgspec.qdp").exists())
            self.assertFalse((r1_dir / "mos1S001-bkgspec.qdp").is_symlink())
            self.assertEqual(res_no_qdp.regions["r1"].instruments["mos1"].qdp_plots, {})

    def test_failed_rerun_deactivates_stale_region_and_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare, mask_res = self._setup_workspace(root)
            reg1 = root / "r1.reg"
            reg1.write_text("fk5\ncircle(287.95,4.98,60\")\n", encoding="utf-8")
            cfg.spectrum.regions = [SpectrumRegionSpec(name="r1", file=str(reg1), mode="single")]

            calls: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls),
            ):
                ok = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertTrue(ok.regions["r1"].active)

            # Force rerun that fails for r1 -> manifest must be status='failed' & r1.active=False
            with (
                patch(
                    "xmm_tools.exec.runner.SASRunner.run",
                    new=self._fake_runner_factory(calls, fail_region="r1"),
                ),
                self.assertRaises(RuntimeError),
            ):
                run_spectrum(cfg, flare=flare, mask_res=mask_res, force=True)

            failed = load_spectrum_result(cfg)
            self.assertEqual(failed.manifest.status, "failed")
            self.assertIn("r1", failed.regions)
            self.assertFalse(failed.regions["r1"].active)

    def test_unmatched_only_raises_and_preserves_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare, mask_res = self._setup_workspace(root)
            reg1 = root / "r1.reg"
            reg1.write_text("fk5\ncircle(287.95,4.98,60\")\n", encoding="utf-8")
            cfg.spectrum.regions = [
                SpectrumRegionSpec(name="r1", file=str(reg1), mode="single")
            ]

            sdir = cfg.paths.resolve(cfg.paths.spectrum_dir)
            manifest_path = sdir / "manifest.json"
            self.assertFalse(manifest_path.exists())

            # 1. Fresh workspace: raises ValueError and does not create spectrum/manifest.json
            with self.assertRaises(ValueError) as ctx:
                run_spectrum(
                    cfg,
                    flare=flare,
                    mask_res=mask_res,
                    only=["nonexistent_typo"],
                )
            self.assertIn("nonexistent_typo", str(ctx.exception))
            self.assertFalse(manifest_path.exists())
            self.assertFalse(sdir.exists())

            # 2. Existing valid manifest on disk: raises ValueError and leaves it unmodified
            calls: list[tuple[str, Path, dict]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run",
                new=self._fake_runner_factory(calls),
            ):
                res = run_spectrum(cfg, flare=flare, mask_res=mask_res)
            self.assertTrue(res.manifest_path.is_file())
            before_text = res.manifest_path.read_text(encoding="utf-8")

            with self.assertRaises(ValueError):
                run_spectrum(
                    cfg,
                    flare=flare,
                    mask_res=mask_res,
                    only=["nonexistent_typo"],
                )
            self.assertEqual(res.manifest_path.read_text(encoding="utf-8"), before_text)
