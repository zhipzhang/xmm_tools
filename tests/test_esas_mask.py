"""Tests for ESAS custom ``region`` masking, CLI, and ``--mask-dir`` integration."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from xmm_tools.config import AnaConfig, MaskConfig, render_template_toml
from xmm_tools.esas import (
    CheeseResult,
    FlareResult,
    SpectrumResult,
    load_cheese_result,
    run_region_mask,
)
from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    RunResult,
    SASCommandError,
    StageManifest,
)
from xmm_tools.pipeline import run_pipeline
from xmm_tools.pipeline.cli import main as cli_main


def _touch_dummy_fits(path: Path, payload: bytes = b"A" * 2880) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _make_event_fits_with_ref_wcs(path: Path) -> Path:
    col = fits.Column(name="PI", format="J", array=np.array([1000, 1500], dtype=np.int32))
    tbl = fits.BinTableHDU.from_columns([col], name="EVENTS")
    tbl.header["REFXCRPX"] = 12800.5
    tbl.header["REFXCRVL"] = 287.9565
    tbl.header["REFXCDLT"] = -1.388889e-5
    tbl.header["REFXCTYP"] = "RA---TAN"
    tbl.header["REFXCUNI"] = "deg"
    tbl.header["REFYCRPX"] = 12800.5
    tbl.header["REFYCRVL"] = 4.9827
    tbl.header["REFYCDLT"] = 1.388889e-5
    tbl.header["REFYCTYP"] = "DEC--TAN"
    tbl.header["REFYCUNI"] = "deg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(path, overwrite=True)
    return path


def _make_srclist_fits(path: Path, n_rows: int) -> Path:
    ra = np.linspace(287.9, 288.0, n_rows, dtype=np.float64) if n_rows else np.empty(0, dtype=np.float64)
    dec = np.linspace(4.9, 5.0, n_rows, dtype=np.float64) if n_rows else np.empty(0, dtype=np.float64)
    tbl = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="DEC", format="D", array=dec),
        ],
        name="SRCLIST",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), tbl]).writeto(path, overwrite=True)
    return path


class TestESASMask(unittest.TestCase):
    def _prepare_flare(
        self, root: Path, *, with_real_evt_wcs: bool = False
    ) -> tuple[AnaConfig, FlareResult]:
        toml_path = root / "ana.toml"
        toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
        cfg = AnaConfig.from_toml(toml_path)
        flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
        flare_dir.mkdir(parents=True, exist_ok=True)

        clean_events: dict[str, Path] = {}
        for key, fname in (
            ("mos1", "mos1_evt-allevc.fits"),
            ("mos2", "mos2_evt-allevc.fits"),
            ("pn", "pn_evt-allevc.fits"),
        ):
            p = flare_dir / fname
            if with_real_evt_wcs:
                _make_event_fits_with_ref_wcs(p)
            else:
                _touch_dummy_fits(p)
            clean_events[key] = p

        flare_res = FlareResult(
            flare_dir=flare_dir,
            manifest_path=flare_dir / "manifest.json",
            flag_filtered=True,
            clean_events=clean_events,
            workspace_root=root,
        )
        flare_res.save_manifest()
        return cfg, flare_res

    def test_run_region_mask_styles_ledger_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = self._prepare_flare(root)
            srclist = _touch_dummy_fits(root / "merged_emllist.fits", b"S" * 2880)

            seen_cmds: list = []

            def fake_run(self_runner, cmd, *, force=False):
                seen_cmds.append(cmd)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                for out in cmd.expected_outputs:
                    (cmd.cwd_path / out).write_bytes(b"R" * 2880)
                return RunResult(command=cmd, status="completed", returncode=0)

            # 1. Default contour (col)
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_region_mask(cfg, srclist=srclist, out_dir="mask_contour")

            self.assertEqual(len(seen_cmds), 6)  # det + sky for mos1, mos2, pn
            self.assertEqual(res.method, "region_mask")
            self.assertEqual(res.cheese_masks, {})
            self.assertEqual(set(res.region_det.keys()), {"mos1", "mos2", "pn"})
            self.assertEqual(set(res.region_sky.keys()), {"mos1", "mos2", "pn"})

            det_cmd = seen_cmds[0]
            sky_cmd = seen_cmds[1]
            self.assertEqual(det_cmd.program, "region")
            self.assertEqual(det_cmd.params["outunit"], "detxy")
            self.assertEqual(sky_cmd.params["outunit"], "xy")
            self.assertEqual(det_cmd.params["operationstyle"], "global")
            self.assertEqual(det_cmd.params["srclisttab"], "srclist.fits:SRCLIST")
            self.assertEqual(det_cmd.params["radiusstyle"], "contour")
            self.assertEqual(det_cmd.params["bkgratestyle"], "col")
            self.assertEqual(det_cmd.params["bkgratecol"], "BG_MAP")
            self.assertEqual(det_cmd.params["bkgfraction"], 0.4)
            self.assertEqual(det_cmd.params["nosrcellipse"], "no")
            self.assertEqual(det_cmd.params["withboresightfudge"], "yes")
            self.assertNotIn("bkgrateuser", det_cmd.params)
            self.assertNotIn("energyfraction", det_cmd.params)
            self.assertNotIn("fixedradius", det_cmd.params)

            # Published relative symlinks inside out_dir
            self.assertTrue((res.cheese_dir / "emllist.fits").is_symlink())
            self.assertFalse(Path(os.readlink(res.cheese_dir / "emllist.fits")).is_absolute())
            self.assertTrue((res.cheese_dir / "mos1_evt-bkgregtdet.fits").is_symlink())
            self.assertTrue((res.cheese_dir / "mos1_evt-bkgregtsky.fits").is_symlink())

            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["stage"], "mask")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["products"]["method"], "region_mask")

            # Re-running unchanged reuses all 6 steps
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res_reused = run_region_mask(cfg, srclist=srclist, out_dir="mask_contour")
            self.assertEqual(seen_cmds, [])
            self.assertEqual(res_reused.manifest.status, "reused")

            # Mutating srclist with identical byte size invalidates via SHA-256
            srclist.write_bytes(b"T" * 2880)
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res_rebuilt = run_region_mask(cfg, srclist=srclist, out_dir="mask_contour")
            self.assertEqual(len(seen_cmds), 6)
            self.assertEqual(res_rebuilt.manifest.status, "completed")

            # 2. contour + user + nosrcellipse
            seen_cmds.clear()
            user_cfg = MaskConfig(
                radius_style="contour",
                bkgrate_style="user",
                bkgrate_user=2.5,
                nosrcellipse=True,
                bkg_fraction=0.3,
                expression="DET_ML > 15",
            )
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                run_region_mask(
                    cfg, srclist=srclist, out_dir="mask_user", mask_cfg=user_cfg
                )
            p_user = seen_cmds[0].params
            self.assertEqual(p_user["bkgratestyle"], "user")
            self.assertEqual(p_user["bkgrateuser"], 2.5)
            self.assertEqual(p_user["nosrcellipse"], "yes")
            self.assertEqual(p_user["bkgfraction"], 0.3)
            self.assertEqual(p_user["expression"], "DET_ML > 15")

            # 3. enfrac
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                run_region_mask(
                    cfg,
                    srclist=srclist,
                    out_dir="mask_enfrac",
                    mask_cfg=MaskConfig(radius_style="enfrac", energy_fraction=0.9),
                )
            p_eef = seen_cmds[0].params
            self.assertEqual(p_eef["radiusstyle"], "enfrac")
            self.assertEqual(p_eef["energyfraction"], 0.9)
            self.assertNotIn("bkgfraction", p_eef)
            self.assertNotIn("fixedradius", p_eef)

            # 4. userfixed
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                run_region_mask(
                    cfg,
                    srclist=srclist,
                    out_dir="mask_fixed",
                    mask_cfg=MaskConfig(radius_style="userfixed", fixed_radius=30.0),
                )
            p_fix = seen_cmds[0].params
            self.assertEqual(p_fix["radiusstyle"], "userfixed")
            self.assertEqual(p_fix["fixedradius"], 30.0)
            self.assertNotIn("bkgfraction", p_fix)
            self.assertNotIn("energyfraction", p_fix)

    def test_guards_dry_run_and_reserved_extra_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare = self._prepare_flare(root)
            srclist = _touch_dummy_fits(root / "merged_emllist.fits")

            # Refusal to overwrite paths.cheese_dir
            with self.assertRaises(ValueError):
                run_region_mask(cfg, srclist=srclist, out_dir=cfg.paths.cheese_dir)

            # Missing srclist raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                run_region_mask(cfg, srclist=root / "missing.fits", out_dir="mask_out")

            # Reserved or path-valued extra_params rejected
            for bad_extra in (
                {"eventset": "evil.fits"},
                {"outunit": "xy"},
                {"radiusstyle": "enfrac"},
                {"some_path": "/abs/path.fits"},
            ):
                with self.assertRaises(ValueError):
                    run_region_mask(
                        cfg,
                        srclist=srclist,
                        out_dir="mask_bad",
                        mask_cfg=MaskConfig(extra_params=bad_extra),
                        flare=flare,
                        dry_run=True,
                    )

            # dry_run performs zero disk writes and invokes no subprocesses
            dry_out = root / "mask_dry"
            with patch("xmm_tools.exec.runner.subprocess.run") as mock_subproc:
                res_dry = run_region_mask(
                    cfg, srclist=srclist, out_dir=dry_out, dry_run=True
                )
            mock_subproc.assert_not_called()
            self.assertEqual(res_dry.manifest.status, "dry_run")
            self.assertFalse(dry_out.exists())

    def test_zero_row_srclist_writes_valid_empty_region_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = self._prepare_flare(root, with_real_evt_wcs=True)
            empty_src = _make_srclist_fits(root / "empty_emllist.fits", n_rows=0)

            with patch("xmm_tools.exec.runner.SASRunner.run") as mock_run:
                res = run_region_mask(cfg, srclist=empty_src, out_dir="mask_empty")
            mock_run.assert_not_called()
            self.assertEqual(res.manifest.status, "completed")
            self.assertIn("empty_region:mos1", res.manifest.steps)

            with fits.open(res.region_det["mos1"]) as hdul:
                self.assertIn("REGION", hdul)
                reg_hdu = hdul["REGION"]
                self.assertEqual(len(reg_hdu.data), 0)
                self.assertEqual(
                    list(reg_hdu.columns.names),
                    ["SHAPE", "DETX", "DETY", "R", "ROTANG", "COMPONENT"],
                )

            with fits.open(res.region_sky["pn"]) as hdul:
                self.assertIn("REGION", hdul)
                reg_hdu = hdul["REGION"]
                self.assertEqual(len(reg_hdu.data), 0)
                self.assertEqual(
                    list(reg_hdu.columns.names),
                    ["SHAPE", "X", "Y", "R", "ROTANG", "COMPONENT"],
                )
                self.assertEqual(reg_hdu.header["TCTYP2"], "RA---TAN")
                self.assertEqual(reg_hdu.header["TCTYP3"], "DEC--TAN")
                self.assertAlmostEqual(reg_hdu.header["TCRVL2"], 287.9565)
                self.assertAlmostEqual(reg_hdu.header["TCRVL3"], 4.9827)

            # Second run reuses empty_region steps
            with patch("xmm_tools.exec.runner.SASRunner.run") as mock_run:
                res2 = run_region_mask(cfg, srclist=empty_src, out_dir="mask_empty")
            mock_run.assert_not_called()
            self.assertEqual(res2.manifest.status, "reused")

    def test_cli_mask_preserves_toml_defaults_and_applies_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "ana.toml"
            toml_text = (
                render_template_toml(workdir=".")
                .replace('radius_style = "contour"', 'radius_style = "enfrac"')
                .replace("energy_fraction = 0.85", "energy_fraction = 0.91")
                .replace("nosrcellipse = false", "nosrcellipse = true")
            )
            toml_path.write_text(toml_text, encoding="utf-8")

            captured: list[MaskConfig] = []

            def fake_run_region_mask(cfg, *, srclist, out_dir, mask_cfg=None, **kw):
                captured.append(mask_cfg)
                return CheeseResult(
                    cheese_dir=root / out_dir,
                    manifest_path=root / out_dir / "manifest.json",
                    method="region_mask",
                    emllist=root / out_dir / "emllist.fits",
                    region_det={"mos1": root / out_dir / "mos1_evt-bkgregtdet.fits"},
                    region_sky={"mos1": root / out_dir / "mos1_evt-bkgregtsky.fits"},
                    manifest=StageManifest(stage="mask", status="completed"),
                )

            with (
                patch("xmm_tools.pipeline.cli.run_region_mask", side_effect=fake_run_region_mask),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc1 = cli_main(
                    [
                        "mask",
                        "--config",
                        str(toml_path),
                        "--srclist",
                        "merged.fits",
                        "--out-dir",
                        "mask_out",
                    ]
                )
            self.assertEqual(rc1, 0)
            self.assertEqual(captured[0].radius_style, "enfrac")
            self.assertAlmostEqual(captured[0].energy_fraction, 0.91)
            self.assertTrue(captured[0].nosrcellipse)

            buf = io.StringIO()
            with (
                patch("xmm_tools.pipeline.cli.run_region_mask", side_effect=fake_run_region_mask),
                contextlib.redirect_stdout(buf),
            ):
                rc2 = cli_main(
                    [
                        "mask",
                        "--config",
                        str(toml_path),
                        "--srclist",
                        "merged.fits",
                        "--out-dir",
                        "mask_out2",
                        "--radius-style",
                        "contour",
                        "--no-nosrcellipse",
                        "--bkgrate-style",
                        "user",
                        "--bkgrate-user",
                        "3.0",
                        "--bkg-fraction",
                        "0.25",
                        "--expression",
                        "RATE > 0.1",
                    ]
                )
            self.assertEqual(rc2, 0)
            self.assertEqual(captured[1].radius_style, "contour")
            self.assertFalse(captured[1].nosrcellipse)
            self.assertEqual(captured[1].bkgrate_style, "user")
            self.assertAlmostEqual(captured[1].bkgrate_user, 3.0)
            self.assertAlmostEqual(captured[1].bkg_fraction, 0.25)
            self.assertEqual(captured[1].expression, "RATE > 0.1")
            self.assertIn("Mask dir:", buf.getvalue())
            self.assertIn("[status=completed]", buf.getvalue())

    def test_load_cheese_result_with_mask_dir_manifest_and_manifestless(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = self._prepare_flare(root)

            # 1. With manifest.json
            mask_with_manifest = root / "mask_m"
            mask_with_manifest.mkdir()
            det = _touch_dummy_fits(mask_with_manifest / "mos1_evt-bkgregtdet.fits")
            sky = _touch_dummy_fits(mask_with_manifest / "mos1_evt-bkgregtsky.fits")
            CheeseResult(
                cheese_dir=mask_with_manifest,
                manifest_path=mask_with_manifest / "manifest.json",
                method="region_mask",
                region_det={"mos1": det},
                region_sky={"mos1": sky},
                workspace_root=root,
            ).save_manifest()

            loaded = load_cheese_result(cfg, mask_dir="mask_m")
            self.assertEqual(loaded.method, "region_mask")
            self.assertEqual(loaded.region_det["mos1"], det)

            # 2. Manifest-less standalone directory (even when flare manifest is absent)
            (cfg.paths.resolve(cfg.paths.flare_dir) / "manifest.json").unlink()
            standalone = root / "standalone_masks"
            s_det = _touch_dummy_fits(standalone / "mos1S001-bkgregtdet.fits")
            s_sky = _touch_dummy_fits(standalone / "mos1S001-bkgregtsky.fits")
            s_chs = _touch_dummy_fits(standalone / "pnS003-cheeset.fits")

            loaded2 = load_cheese_result(cfg, mask_dir=standalone)
            self.assertEqual(loaded2.region_det["mos1"], s_det)
            self.assertEqual(loaded2.region_sky["mos1"], s_sky)
            self.assertEqual(loaded2.cheese_masks["pn"], s_chs)

            # Ambiguous files raise ValueError
            _touch_dummy_fits(standalone / "mos1S002-bkgregtdet.fits")
            with self.assertRaises(ValueError):
                load_cheese_result(cfg, mask_dir=standalone)

    def test_spectrum_and_run_pipeline_respect_mask_dir_over_cheese(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare = self._prepare_flare(root)
            reg_file = root / "bg.reg"
            reg_file.write_text('fk5\ncircle(287.95,4.98,60")\n', encoding="utf-8")

            toml_path = root / "ana.toml"
            toml_path.write_text(
                render_template_toml(workdir=".")
                + '\n[[esas.spectrum.regions]]\nname = "bg"\nfile = "bg.reg"\nmode = "single"\n',
                encoding="utf-8",
            )
            cfg = AnaConfig.from_toml(toml_path)

            # Populate both native cheese/ and custom mask_merged/
            cdir = cfg.paths.resolve(cfg.paths.cheese_dir)
            mdir = cfg.paths.resolve("mask_merged")
            for inst, pfx in (("mos1", "mos1_evt"), ("mos2", "mos2_evt"), ("pn", "pn_evt")):
                _touch_dummy_fits(cdir / f"{pfx}-bkgregtdet.fits")
                _touch_dummy_fits(cdir / f"{pfx}-bkgregtsky.fits")
                _touch_dummy_fits(mdir / f"{pfx}-bkgregtdet.fits")
                _touch_dummy_fits(mdir / f"{pfx}-bkgregtsky.fits")

            # Test run_spectrum(..., mask_dir="mask_merged") via CLI
            seen_mask_dirs: list[Path] = []

            def fake_run_single_region_job(job, **kwargs):
                mres = kwargs["mask_res"]
                seen_mask_dirs.append(mres.cheese_dir)
                raise RuntimeError("stop after checking mask_res")

            with (
                patch(
                    "xmm_tools.esas.spectrum.workflow.run_single_region_job",
                    side_effect=fake_run_single_region_job,
                ),
                patch(
                    "xmm_tools.esas.spectrum.workflow.ensure_ref_sky_images",
                    return_value={"mos1": root / "r1.fits", "mos2": root / "r2.fits", "pn": root / "r3.fits"},
                ),
            ):
                with self.assertRaises(RuntimeError):
                    cli_main(
                        [
                            "spectrum",
                            "--config",
                            str(toml_path),
                            "--mask-dir",
                            "mask_merged",
                        ]
                    )
                self.assertEqual(seen_mask_dirs, [mdir])

            # Test run_pipeline even when cheese stage ALSO runs: spectrum must still get mask_merged, NOT cheese!
            captured_spec_calls: list[dict] = []

            def fake_cheese(c, **kw):
                return CheeseResult(
                    cheese_dir=cdir,
                    manifest_path=cdir / "manifest.json",
                    method="cheese",
                    manifest=StageManifest(stage="cheese", status="completed"),
                )

            def fake_spectrum_res(c, **kw):
                captured_spec_calls.append(dict(kw))
                return SpectrumResult(
                    spectrum_dir=root / "spectrum",
                    manifest_path=root / "spectrum" / "manifest.json",
                    manifest=StageManifest(stage="spectrum", status="completed"),
                )

            with (
                patch("xmm_tools.pipeline.engine.run_flare", return_value=flare),
                patch("xmm_tools.pipeline.engine.run_cheese", side_effect=fake_cheese),
                patch("xmm_tools.pipeline.engine.run_spectrum", side_effect=fake_spectrum_res),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc = cli_main(
                    [
                        "run",
                        "--config",
                        str(toml_path),
                        "--stages",
                        "flare,cheese,spectrum",
                        "--mask-dir",
                        "mask_merged",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(len(captured_spec_calls), 1)
            self.assertIsNone(captured_spec_calls[0]["mask_res"])
            self.assertEqual(captured_spec_calls[0]["mask_dir"], "mask_merged")

            # End-to-end check: run_pipeline forwarding mask_dir directly to real run_spectrum
            seen_mask_dirs.clear()
            with (
                patch("xmm_tools.pipeline.engine.run_flare", return_value=flare),
                patch("xmm_tools.pipeline.engine.run_cheese", side_effect=fake_cheese),
                patch(
                    "xmm_tools.esas.spectrum.workflow.run_single_region_job",
                    side_effect=fake_run_single_region_job,
                ),
                patch(
                    "xmm_tools.esas.spectrum.workflow.ensure_ref_sky_images",
                    return_value={
                        "mos1": root / "r1.fits",
                        "mos2": root / "r2.fits",
                        "pn": root / "r3.fits",
                    },
                ),
                self.assertRaises(RuntimeError),
            ):
                run_pipeline(
                    toml_path,
                    stages=["flare", "cheese", "spectrum"],
                    mask_dir="mask_merged",
                )
            self.assertEqual(seen_mask_dirs, [mdir])

    def test_symlinked_out_dir_and_parent_clobber_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_obs = root / "real_obs"
            real_obs.mkdir()
            cfg, flare = self._prepare_flare(real_obs)
            srclist = _touch_dummy_fits(real_obs / "merged_emllist.fits")

            # 1. Symlinked output directory: cheese_alias -> cheese
            cheese_alias = real_obs / "cheese_alias"
            cheese_alias.symlink_to(real_obs / "cheese")
            with self.assertRaises(ValueError):
                run_region_mask(
                    cfg, srclist=srclist, out_dir=cheese_alias, flare=flare
                )

            # 2. Symlinked parent directory: obs_alias -> real_obs, out_dir = obs_alias/cheese
            obs_alias = root / "obs_alias"
            obs_alias.symlink_to(real_obs)
            with self.assertRaises(ValueError):
                run_region_mask(
                    cfg, srclist=srclist, out_dir=obs_alias / "cheese", flare=flare
                )

    def test_external_srclist_staged_before_steps_and_docker_mount_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workdir = tmp_path / "obs1"
            workdir.mkdir()
            cfg, _ = self._prepare_flare(workdir)
            ext_srclist = _touch_dummy_fits(
                tmp_path / "external_merged.fits", b"E" * 2880
            )

            seen_cmds: list = []

            def fake_run(self_runner, cmd, *, force=False):
                seen_cmds.append(cmd)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                for out in cmd.expected_outputs:
                    (cmd.cwd_path / out).write_bytes(b"R" * 2880)
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_region_mask(
                    cfg, srclist=ext_srclist, out_dir="mask_ext"
                )

            self.assertEqual(len(seen_cmds), 6)
            staged_srclist = res.cheese_dir / "_work" / "srclist" / "emllist.fits"
            self.assertTrue(staged_srclist.is_file())
            self.assertEqual(staged_srclist.read_bytes(), b"E" * 2880)
            self.assertEqual(seen_cmds[0].symlinks["srclist.fits"], staged_srclist)

            manifest_data = json.loads(
                res.manifest_path.read_text(encoding="utf-8")
            )
            # Strip "shell" before checking for any host path leak
            manifest_no_shell = json.loads(json.dumps(manifest_data))
            for step_rec in manifest_no_shell["steps"].values():
                step_rec.pop("shell", None)
            serialized_no_shell = json.dumps(manifest_no_shell)
            self.assertNotIn(str(tmp_path), serialized_no_shell)
            self.assertNotIn(os.path.realpath(tmp_path), serialized_no_shell)
            self.assertEqual(
                manifest_data["steps"]["region_det:mos1"]["symlinks"]["srclist.fits"],
                "../../srclist/emllist.fits",
            )
            self.assertEqual(
                manifest_data["steps"]["region_det:mos1"]["inputs"]["srclist"]["path"],
                "_work/srclist/emllist.fits",
            )

            # Mutating external srclist invalidates StepLedger and re-runs region
            ext_srclist.write_bytes(b"F" * 2880)
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res2 = run_region_mask(
                    cfg, srclist=ext_srclist, out_dir="mask_ext"
                )
            self.assertEqual(len(seen_cmds), 6)
            self.assertEqual(res2.manifest.status, "completed")
            self.assertEqual(staged_srclist.read_bytes(), b"F" * 2880)

            # Docker mount validation rejects unmounted --out-dir
            mounted_dir = tmp_path / "mounted"
            mounted_dir.mkdir()
            with (
                patch(
                    "xmm_tools.exec.runner.SASRunner._resolve_docker_mounts",
                    return_value=[(mounted_dir, Path("/work"))],
                ),
                self.assertRaisesRegex(
                    ValueError, "not mounted inside Docker container"
                ),
            ):
                run_region_mask(
                    cfg,
                    srclist=ext_srclist,
                    out_dir="mask_unmounted",
                    docker_container="xmmsas_box",
                )

            # Symlinked output parent pointing into mounted_dir translates docker -w cleanly
            alias_parent = tmp_path / "mounted_alias"
            alias_parent.symlink_to(mounted_dir)
            seen_exec_argvs: list[list[str]] = []

            def fake_subproc(argv, **kwargs):
                seen_exec_argvs.append(list(argv))
                w_idx = argv.index("-w")
                c_cwd = Path(argv[w_idx + 1])
                rel_cwd = c_cwd.relative_to("/work")
                host_cwd = mounted_dir / rel_cwd
                host_cwd.mkdir(parents=True, exist_ok=True)
                for arg in argv:
                    if arg.startswith("bkgregionset="):
                        (host_cwd / arg.split("=", 1)[1]).write_bytes(b"R" * 2880)

                class _Proc:
                    returncode = 0
                    stdout = "ok\n"
                    stderr = ""

                return _Proc()

            with (
                patch(
                    "xmm_tools.exec.runner.SASRunner._resolve_docker_mounts",
                    return_value=[(Path(os.path.realpath(mounted_dir)), Path("/work"))],
                ),
                patch("xmm_tools.exec.runner.subprocess.run", side_effect=fake_subproc),
            ):
                res_alias = run_region_mask(
                    cfg,
                    srclist=ext_srclist,
                    out_dir=alias_parent / "mask_via_alias",
                    docker_container="xmmsas_box",
                )
            self.assertEqual(len(seen_exec_argvs), 6)
            for argv in seen_exec_argvs:
                w_idx = argv.index("-w")
                self.assertTrue(argv[w_idx + 1].startswith("/work/mask_via_alias/_work/"))
            self.assertEqual(res_alias.manifest.status, "completed")

    def test_expression_no_sources_selected_falls_back_to_empty_region_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = self._prepare_flare(root, with_real_evt_wcs=True)
            srclist = _make_srclist_fits(root / "merged_emllist.fits", n_rows=2)

            def fake_run_no_sources(self_runner, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                log_path = cmd.cwd_path / cmd.log_file
                log_path.write_text(
                    "** region: error (noSourcesSelected), No sources passed expression\n",
                    encoding="utf-8",
                )
                fail_res = RunResult(
                    command=cmd,
                    status="failed",
                    returncode=1,
                    log_path=log_path,
                    error="region failed with exit code 1",
                )
                raise SASCommandError(
                    "region failed with exit code 1", result=fail_res
                )

            mcfg = MaskConfig(expression="DET_ML > 99999")
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_no_sources):
                res = run_region_mask(
                    cfg, srclist=srclist, out_dir="mask_no_src", mask_cfg=mcfg
                )

            self.assertEqual(res.manifest.status, "completed")
            for inst in ("mos1", "mos2", "pn"):
                self.assertIn(f"empty_region:{inst}", res.manifest.steps)
                self.assertNotIn(f"region_det:{inst}", res.manifest.steps)
                self.assertNotIn(f"region_sky:{inst}", res.manifest.steps)

            with fits.open(res.region_det["mos1"]) as hdul:
                reg = hdul["REGION"]
                self.assertEqual(len(reg.data), 0)
                self.assertEqual(reg.header["HDUVERS"], "1.0.0")
                self.assertEqual(reg.header["HDUCLASS"], "ASC")
                self.assertEqual(reg.header["HDUCLAS1"], "REGION")
                self.assertEqual(reg.header["HDUCLAS2"], "STANDARD")
                self.assertEqual(reg.header["MTYPE1"], "pos")
                self.assertEqual(reg.header["MFORM1"], "DETX,DETY")

            with fits.open(res.region_sky["pn"]) as hdul:
                reg = hdul["REGION"]
                self.assertEqual(len(reg.data), 0)
                self.assertEqual(reg.header["HDUCLASS"], "ASC")
                self.assertEqual(reg.header["MFORM1"], "X,Y")

            # Re-running with identical expression reuses empty_region steps cleanly
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_no_sources):
                res2 = run_region_mask(
                    cfg, srclist=srclist, out_dir="mask_no_src", mask_cfg=mcfg
                )
            self.assertEqual(res2.manifest.status, "reused")

            # Empty expression must NOT swallow noSourcesSelected
            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_no_sources),
                self.assertRaises(SASCommandError),
            ):
                run_region_mask(
                    cfg,
                    srclist=srclist,
                    out_dir="mask_empty_expr_fail",
                    mask_cfg=MaskConfig(expression=""),
                )

    def test_legacy_failed_manifest_rejected_and_mos10_ignored_in_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = self._prepare_flare(root)

            # 1. Legacy manifest with "status": "failed" must be rejected
            legacy_failed_dir = root / "legacy_failed"
            legacy_failed_dir.mkdir()
            (legacy_failed_dir / "manifest.json").write_text(
                json.dumps({"status": "failed", "method": "cheese"}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_cheese_result(cfg, mask_dir=legacy_failed_dir)

            # Legacy manifest omitting "status" still loads
            legacy_ok_dir = root / "legacy_ok"
            legacy_ok_dir.mkdir()
            (legacy_ok_dir / "manifest.json").write_text(
                json.dumps({"method": "cheese", "cheese_masks": {}}),
                encoding="utf-8",
            )
            loaded_legacy = load_cheese_result(cfg, mask_dir=legacy_ok_dir)
            self.assertEqual(loaded_legacy.method, "cheese")

            # 2. mos10foo-bkgregtdet.fits must NOT match mos1
            (cfg.paths.resolve(cfg.paths.flare_dir) / "manifest.json").unlink()
            scan_dir = root / "scan_mos10"
            _touch_dummy_fits(scan_dir / "mos10foo-bkgregtdet.fits")
            loaded_none = load_cheese_result(cfg, mask_dir=scan_dir)
            self.assertEqual(loaded_none.region_det, {})

            mos1_det = _touch_dummy_fits(scan_dir / "mos1S001-bkgregtdet.fits")
            loaded_mos1 = load_cheese_result(cfg, mask_dir=scan_dir)
            self.assertEqual(loaded_mos1.region_det, {"mos1": mos1_det})


if __name__ == "__main__":
    unittest.main()
