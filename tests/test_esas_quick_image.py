"""Tests for qpb-image stage (evqpb + eimageget + eimagecombine) and incremental ledger."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from xmm_tools.esas import FlareResult
from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    SASCommandError,
    SASRunner,
    atomic_write_json,
)
from xmm_tools.exec.runner import RunResult
from xmm_tools.pipeline import AnaConfig, BandSpec, render_template_toml
from xmm_tools.sas import QPBImageResult, resolve_subbands, run_qpb_image


class TestESASQuickImage(unittest.TestCase):
    def _write_raw_events(self, root: Path) -> None:
        for name in ("mos1_evt.fits", "mos2_evt.fits", "pn_evt.fits", "pn_oot_evt.fits"):
            (root / name).write_text("raw_evt", encoding="utf-8")

    def _write_flare_with_gti(self, cfg: AnaConfig) -> FlareResult:
        flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
        flare_dir.mkdir(parents=True, exist_ok=True)
        clean_events = {}
        gti_files = {}
        for key, prefix in (("mos1", "mos1S001"), ("mos2", "mos2S002"), ("pn", "pnS003")):
            evt = flare_dir / f"{prefix}-allevc.fits"
            evt.write_text("allevc", encoding="utf-8")
            clean_events[key] = evt
            gti = flare_dir / f"{prefix}-gti.fits"
            gti.write_text("gti", encoding="utf-8")
            gti_files[key] = gti
        pn_oot = flare_dir / "pnS003-allevcoot.fits"
        pn_oot.write_text("allevcoot", encoding="utf-8")
        clean_events["pn_oot"] = pn_oot

        flare = FlareResult(
            flare_dir=flare_dir,
            manifest_path=flare_dir / "manifest.json",
            flag_filtered=True,
            clean_events=clean_events,
            gti_files=gti_files,
        )
        flare.save_manifest()
        return flare

    def _make_fake_run(self, calls: list[tuple[str, dict, dict, Path]]):
        inst_tags = {
            "mos1_evt.fits": "P0923400301M1S001",
            "mos2_evt.fits": "P0923400301M2S002",
            "pn_evt.fits": "P0923400301PNS003",
        }

        def fake_run(runner_self, cmd, *, force=False):
            calls.append(
                (cmd.program, dict(cmd.params), dict(cmd.symlinks), cmd.cwd_path)
            )
            cmd.cwd_path.mkdir(parents=True, exist_ok=True)
            if cmd.program == "eimageget":
                tag = inst_tags[Path(cmd.params["evtfile"]).name]
                for idx in range(len(cmd.params["pimin"])):
                    (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text(
                        f"ima_{tag}_{idx}", encoding="utf-8"
                    )
                    (cmd.cwd_path / f"{tag}_ima_{idx}exp.fits").write_text(
                        f"exp_{tag}_{idx}", encoding="utf-8"
                    )
                    (cmd.cwd_path / f"{tag}_ima_{idx}fwc.fits").write_text(
                        f"fwc_{tag}_{idx}", encoding="utf-8"
                    )
            elif cmd.program == "eimagecombine":
                prefix = cmd.params["prefix"]
                n_sub = len(cmd.params["epn_weight"])
                width_scale = float(cmd.params.get("width", 5.0)) / 5.0
                for idx in range(n_sub):
                    cnt_val = 3.0 + 2.0 * idx
                    exp_val = 2.0
                    sm_val = (cnt_val / exp_val) * width_scale
                    fits.PrimaryHDU(
                        data=np.full((4, 4), cnt_val, dtype=np.float32)
                    ).writeto(
                        cmd.cwd_path / f"{prefix}ima_{idx}_sub.fits",
                        overwrite=True,
                    )
                    fits.PrimaryHDU(
                        data=np.full((4, 4), exp_val, dtype=np.float32)
                    ).writeto(
                        cmd.cwd_path / f"{prefix}ima_{idx}exp.fits",
                        overwrite=True,
                    )
                    sm_arr = np.full((4, 4), sm_val, dtype=np.float32)
                    sm_arr[0, 0] = np.nan
                    fits.PrimaryHDU(data=sm_arr).writeto(
                        cmd.cwd_path / f"{prefix}ima_{idx}_subdiv_smooth.fits",
                        overwrite=True,
                    )
                mask = np.ones((4, 4), dtype=np.float32)
                mask[0, 0] = 0.0
                fits.PrimaryHDU(data=mask).writeto(
                    cmd.cwd_path / f"{prefix}ima_mask.fits", overwrite=True
                )
            else:
                for out in cmd.expected_outputs:
                    (cmd.cwd_path / out).write_text(
                        f"fits_{cmd.program}_{out}", encoding="utf-8"
                    )
            return RunResult(command=cmd, status="completed", returncode=0)

        return fake_run

    def test_resolve_subbands_explicit_and_auto_split(self):
        edges = [350, 500, 1000, 2000, 4500, 7000]
        explicit = BandSpec(
            name="soft", emin=500, emax=2000, subbands=[[500, 1200], [1200, 2000]]
        )
        self.assertEqual(resolve_subbands(explicit, edges), [(500, 1200), (1200, 2000)])

        auto = BandSpec(name="full", emin=500, emax=7000)
        self.assertEqual(
            resolve_subbands(auto, edges),
            [(500, 1000), (1000, 2000), (2000, 4500), (4500, 7000)],
        )

    def test_run_qpb_image_uses_symlinks_merges_subbands_and_reuses_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)
            flare = self._write_flare_with_gti(cfg)

            calls: list[tuple[str, dict, dict, Path]] = []
            fake_run = self._make_fake_run(calls)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_qpb_image(cfg, flare=flare)

            progs = [p for p, _, _, _ in calls]
            # 1 atthkgen + 3 evqpb + 2 bands * 3 eimageget + 2 eimagecombine
            self.assertEqual(progs.count("atthkgen"), 1)
            self.assertEqual(progs.count("evqpb"), 3)
            self.assertEqual(progs.count("eimageget"), 6)
            self.assertEqual(progs.count("eimagecombine"), 2)

            # Verify evqpb and eimageget pass ONLY local symlink filenames in params
            evqpb_calls = [(params, syms) for p, params, syms, _ in calls if p == "evqpb"]
            self.assertEqual(
                [c["table"] for c, _ in evqpb_calls],
                ["mos1_evt.fits", "mos2_evt.fits", "pn_evt.fits"],
            )
            self.assertEqual(
                [Path(s["mos1_evt.fits"]).name for _, s in evqpb_calls[:1]],
                ["mos1_evt.fits"],
            )

            eimg_calls = [(params, syms) for p, params, syms, _ in calls if p == "eimageget"]
            self.assertEqual(eimg_calls[0][0]["evtfile"], "mos1_evt.fits")
            self.assertEqual(eimg_calls[0][0]["fwcfile"], "mos1_qpb.fits")
            self.assertEqual(eimg_calls[0][0]["gtifile"], "mos1_gti.fits")
            self.assertEqual(eimg_calls[0][0]["attfile"], "atthk.fits")
            self.assertEqual(
                Path(eimg_calls[0][1]["mos1_gti.fits"]).name, "mos1S001-gti.fits"
            )
            self.assertEqual(eimg_calls[0][0]["pimin"], [500, 1000])
            self.assertEqual(eimg_calls[0][0]["pimax"], [1000, 2000])
            self.assertNotIn("ootfile", eimg_calls[0][0])
            self.assertEqual(eimg_calls[2][0]["ootfile"], "pn_oot_evt.fits")
            self.assertEqual(
                Path(eimg_calls[2][1]["pn_oot_evt.fits"]).name, "pn_oot_evt.fits"
            )

            # Verify summed smoothed flux image (1.5 + 2.5 = 4.0, masked [0,0] = NaN)
            self.assertEqual(res.band_flux_images["soft"].name, "soft_500_2000_flux.fits")
            self.assertEqual(res.band_flux_images["hard"].name, "hard_2000_7000_flux.fits")
            with fits.open(res.band_flux_images["soft"]) as hdul:
                expected = np.full((4, 4), 4.0, dtype=np.float32)
                expected[0, 0] = np.nan
                np.testing.assert_allclose(hdul[0].data, expected, equal_nan=True)
                self.assertEqual(hdul[0].header.get("BUNIT"), "counts/s")

            # Verify v2 schema manifest and zero host path leakage in portable fields
            manifest_text = res.manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["stage"], "qpb_image")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["products"]["band_flux_images"]["soft"],
                "soft_500_2000_flux.fits",
            )
            for step_raw in manifest["steps"].values():
                portable_json = json.dumps(
                    {k: v for k, v in step_raw.items() if k != "shell"}
                )
                self.assertNotIn(str(root), portable_json)

            self.assertIn("soft", res.bands)
            self.assertEqual(len(res.bands["soft"].subbands), 2)
            self.assertTrue(res.bands["soft"].mask_image.is_file())
            self.assertEqual(
                res.bands["soft"].subbands[0].counts_image.name,
                "soft_500_1000_counts.fits",
            )
            self.assertEqual(
                res.bands["soft"].subbands[0].exposure_image.name,
                "soft_500_1000_exp.fits",
            )
            reloaded = QPBImageResult.from_manifest(res.manifest_path)
            self.assertEqual(len(reloaded.bands["soft"].subbands), 2)
            self.assertEqual(reloaded.bands["soft"].subbands[1].emin, 1000)
            self.assertEqual(reloaded.bands["soft"].subbands[1].emax, 2000)

            # 1) Re-running without changes reuses every step (0 subprocess calls)
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                reused_res = run_qpb_image(cfg, flare=flare)
            self.assertEqual(calls, [])
            self.assertEqual(reused_res.manifest.status, "reused")

            # 2) Changing smoothing width reuses atthkgen, evqpb, eimageget; re-runs ONLY eimagecombine + export_band
            cfg.qpb_image.width = 3.0
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res_smooth = run_qpb_image(cfg, flare=flare)
            self.assertEqual(
                [p for p, _, _, _ in calls], ["eimagecombine", "eimagecombine"]
            )
            self.assertEqual(res_smooth.manifest.steps["atthkgen"].status, "reused")
            self.assertEqual(res_smooth.manifest.steps["evqpb:mos1"].status, "reused")
            self.assertEqual(
                res_smooth.manifest.steps["eimageget:soft:mos1"].status, "reused"
            )
            self.assertEqual(
                res_smooth.manifest.steps["eimagecombine:soft"].status, "completed"
            )
            self.assertEqual(
                res_smooth.manifest.steps["export_band:soft"].status, "completed"
            )

            # 3) Adding a 3rd band reuses soft & hard completely
            cfg.qpb_image.bands["broad"] = BandSpec(
                name="broad", emin=500, emax=7000, subbands=[[500, 2000], [2000, 7000]]
            )
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res_broad = run_qpb_image(cfg, flare=flare)
            self.assertEqual(
                [p for p, _, _, _ in calls],
                ["eimageget", "eimageget", "eimageget", "eimagecombine"],
            )
            self.assertEqual(
                res_broad.manifest.steps["eimagecombine:soft"].status, "reused"
            )
            self.assertEqual(
                res_broad.manifest.steps["export_band:hard"].status, "reused"
            )
            self.assertEqual(
                res_broad.manifest.steps["eimagecombine:broad"].status, "completed"
            )
            self.assertEqual(
                res_broad.band_flux_images["broad"].name, "broad_500_7000_flux.fits"
            )

            # 4) Re-running with force=True cleans stale files in eimageget & combine dirs
            stale_eimg = (
                res.qpb_image_dir
                / "_work"
                / "bands"
                / "soft"
                / "eimageget_mos1"
                / "stale_old_bin_ima_0.fits"
            )
            stale_comb = (
                res.qpb_image_dir
                / "_work"
                / "bands"
                / "soft"
                / "combine"
                / "soft_500_2000_stale.fits"
            )
            stale_eimg.write_text("stale", encoding="utf-8")
            stale_comb.write_text("stale", encoding="utf-8")
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                run_qpb_image(cfg, flare=flare, force=True)
            self.assertFalse(stale_eimg.exists())
            self.assertFalse(stale_comb.exists())

    def test_dry_run_creates_zero_directories_or_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)
            self._write_flare_with_gti(cfg)

            res = run_qpb_image(cfg, dry_run=True)
            self.assertFalse((root / "qpb_image").exists())
            self.assertIsNotNone(res.manifest)
            self.assertEqual(res.manifest.status, "dry_run")

    def test_run_qpb_image_raises_when_gti_missing_and_validates_v2_flare_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)

            with self.assertRaises(FileNotFoundError):
                run_qpb_image(cfg)

            # Failed v2 flare manifest must raise RuntimeError
            flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
            atomic_write_json(
                flare_dir / "manifest.json",
                {
                    "schema_version": 2,
                    "stage": "flare",
                    "status": "failed",
                    "products": {"gti_files": {}},
                },
            )
            with self.assertRaises(RuntimeError):
                run_qpb_image(cfg)

            # Completed v2 flare manifest resolves GTIs cleanly
            for inst in ("mos1", "mos2", "pn"):
                (flare_dir / f"{inst}S001-gti.fits").write_text("gti", encoding="utf-8")
            atomic_write_json(
                flare_dir / "manifest.json",
                {
                    "schema_version": 2,
                    "stage": "flare",
                    "status": "completed",
                    "products": {
                        "gti_files": {
                            inst: f"{inst}S001-gti.fits"
                            for inst in ("mos1", "mos2", "pn")
                        }
                    },
                },
            )
            calls: list[tuple[str, dict, dict, Path]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)
            ):
                res = run_qpb_image(cfg)
            self.assertTrue(res.band_flux_images["soft"].is_file())

    def test_quick_image_gti_override_in_toml_and_explicit_gti_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            custom_dir = root / "custom_gti"
            custom_dir.mkdir()
            for inst in ("mos1", "mos2", "pn"):
                (custom_dir / f"{inst}.gti").write_text("gti", encoding="utf-8")

            toml_text = (
                render_template_toml(workdir=".")
                + "\n[sas.qpb_image.gti_files]\n"
                + 'mos1 = "custom_gti/mos1.gti"\n'
                + 'mos2 = "custom_gti/mos2.gti"\n'
                + 'pn = "custom_gti/pn.gti"\n'
            )
            toml_path = root / "esas.toml"
            toml_path.write_text(toml_text, encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)

            seen_gti_targets: list[str] = []

            def fake_run(runner_self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    seen_gti_targets.append(Path(cmd.symlinks[cmd.params["gtifile"]]).name)
                    tag = Path(cmd.params["evtfile"]).stem
                    (cmd.cwd_path / f"{tag}_ima_0.fits").write_text("ima", encoding="utf-8")
                elif cmd.program == "eimagecombine":
                    prefix = cmd.params["prefix"]
                    for idx in (0, 1):
                        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                            cmd.cwd_path / f"{prefix}ima_{idx}_sub.fits", overwrite=True
                        )
                        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                            cmd.cwd_path / f"{prefix}ima_{idx}exp.fits", overwrite=True
                        )
                        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                            cmd.cwd_path / f"{prefix}ima_{idx}_subdiv_smooth.fits",
                            overwrite=True,
                        )
                else:
                    for out in cmd.expected_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_qpb_image(cfg)

            self.assertEqual(seen_gti_targets[:3], ["mos1.gti", "mos2.gti", "pn.gti"])
            self.assertTrue(res.band_flux_images["soft"].exists())

    def test_eimagecombine_isolated_dir_docker_symlinks_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)
            flare = self._write_flare_with_gti(cfg)

            combine_cwds: list[Path] = []

            def fake_run_incomplete(runner_self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    tag = Path(cmd.params["evtfile"]).stem
                    for idx in (0, 1):
                        (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text(
                            "ima", encoding="utf-8"
                        )
                elif cmd.program == "eimagecombine":
                    combine_cwds.append(cmd.cwd_path)
                    prefix = cmd.params["prefix"]
                    # Only write sub-band 1 (missing sub-band 0)
                    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                        cmd.cwd_path / f"{prefix}ima_1_sub.fits", overwrite=True
                    )
                    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                        cmd.cwd_path / f"{prefix}ima_1exp.fits", overwrite=True
                    )
                else:
                    for out in cmd.expected_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_incomplete),
                self.assertRaises(SASCommandError) as ctx,
            ):
                run_qpb_image(cfg, flare=flare)
            self.assertIn("incomplete sub-bands", str(ctx.exception))
            self.assertEqual(len(combine_cwds), 1)
            self.assertNotEqual(
                combine_cwds[0],
                cfg.paths.resolve("qpb_image/_work/bands/soft/combine"),
            )

            # Fail fast when combine_log reports cp skipping file
            def fake_run_log_error(runner_self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    tag = Path(cmd.params["evtfile"]).stem
                    for idx in (0, 1):
                        (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text(
                            "ima", encoding="utf-8"
                        )
                elif cmd.program == "eimagecombine":
                    combine_cwds.append(cmd.cwd_path)
                    (cmd.cwd_path / cmd.log_file).write_text(
                        "cp: skipping file 'x_PN_ima_0_sub.fits', as it was replaced while being copied\n",
                        encoding="utf-8",
                    )
                else:
                    for out in cmd.expected_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_log_error),
                self.assertRaises(SASCommandError) as ctx2,
            ):
                run_qpb_image(cfg, flare=flare, force=True)
            self.assertIn("skipping file", str(ctx2.exception))
            self.assertEqual(len(combine_cwds), 2)
            self.assertNotEqual(combine_cwds[0], combine_cwds[1])

            # Docker mode runs eimagecombine inside /tmp/xmm_eimagecombine in container script
            docker_combine_inputs: list[list[str]] = []
            docker_combine_argvs: list[list[str]] = []

            def fake_run_docker(runner_self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    tag = Path(cmd.params["evtfile"]).stem
                    for idx in (0, 1):
                        (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text(
                            "ima", encoding="utf-8"
                        )
                elif cmd.program == "eimagecombine":
                    docker_combine_argvs.append(cmd.argv())
                    exec_argv = SASRunner(
                        env=runner_self.env,
                        docker_container="xmmsas_box",
                        docker_mounts=[(root, "/home/xmmsas/work")],
                    )._build_exec_argv(cmd)
                    self.assertIn("/tmp/xmm_eimagecombine", " ".join(exec_argv))
                    ima_links = sorted(
                        p.name for p in cmd.cwd_path.glob("*_ima_*.fits") if p.is_symlink()
                    )
                    docker_combine_inputs.append(ima_links)
                    for link_name in ima_links:
                        self.assertFalse(
                            Path(os.readlink(cmd.cwd_path / link_name)).is_absolute()
                        )
                    prefix = cmd.params["prefix"]
                    for idx in (0, 1):
                        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                            cmd.cwd_path / f"{prefix}ima_{idx}_sub.fits", overwrite=True
                        )
                        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                            cmd.cwd_path / f"{prefix}ima_{idx}exp.fits", overwrite=True
                        )
                        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(
                            cmd.cwd_path / f"{prefix}ima_{idx}_subdiv_smooth.fits",
                            overwrite=True,
                        )
                else:
                    for out in cmd.expected_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_docker):
                res_docker = run_qpb_image(
                    cfg, flare=flare, force=True, docker_container="xmmsas_box"
                )
            self.assertEqual(len(docker_combine_inputs), 2)
            self.assertTrue(len(docker_combine_inputs[0]) > 0)
            self.assertEqual(
                res_docker.manifest.steps["eimagecombine:soft"].program,
                "eimagecombine",
            )
            self.assertTrue(
                all("/tmp/xmm_eimagecombine" in " ".join(a) for a in docker_combine_argvs)
            )
            self.assertTrue(res_docker.band_flux_images["soft"].is_file())

    def test_corrupted_flux_regeneration_and_manifest_roundtrip_portability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            ccf = root / "ccf.cif"
            odf_sum = root / "1234_0204870101_SCX00000SUM.SAS"
            ccf.write_text("ccf_v1", encoding="utf-8")
            odf_sum.write_text("sum_v1", encoding="utf-8")
            toml_path = root / "esas.toml"
            toml_path.write_text(
                render_template_toml(
                    workdir=".",
                    sas_ccf="ccf.cif",
                    sas_odf="1234_0204870101_SCX00000SUM.SAS",
                ),
                encoding="utf-8",
            )
            cfg = AnaConfig.from_toml(toml_path)
            flare = self._write_flare_with_gti(cfg)

            calls: list[tuple[str, dict, dict, Path]] = []
            with patch(
                "xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)
            ):
                res = run_qpb_image(cfg, flare=flare)

            # Corrupt soft_500_2000_flux.fits with non-FITS bytes
            soft_flux = res.band_flux_images["soft"]
            soft_flux.write_bytes(b"corrupted bytes")

            calls.clear()
            with patch(
                "xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)
            ):
                regen_res = run_qpb_image(cfg, flare=flare)

            # No SAS subprocesses re-run; only export_band:soft regenerates the flux FITS
            self.assertEqual(calls, [])
            self.assertEqual(
                regen_res.manifest.steps["export_band:soft"].status, "completed"
            )
            self.assertEqual(
                regen_res.manifest.steps["export_band:hard"].status, "reused"
            )
            with fits.open(soft_flux) as hdul:
                self.assertEqual(hdul[0].data.shape, (4, 4))

            # Round-trip from_manifest -> save_manifest preserves ../ relative paths
            reloaded = QPBImageResult.from_manifest(regen_res.manifest_path)
            reloaded.save_manifest()
            raw_text = regen_res.manifest_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), raw_text)
            raw_json = json.loads(raw_text)
            gti_in_path = raw_json["steps"]["eimageget:soft:mos1"]["inputs"]["gti_file"]["path"]
            self.assertTrue(gti_in_path.startswith("../soft_proton_flare/"))

            # Changing ccf.cif invalidates atthkgen, evqpb:*, and eimageget:*
            ccf.write_text("ccf_v2", encoding="utf-8")
            calls.clear()
            with patch(
                "xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)
            ):
                run_qpb_image(cfg, flare=flare)
            self.assertIn("atthkgen", [p for p, _, _, _ in calls])
            self.assertEqual([p for p, _, _, _ in calls].count("evqpb"), 3)
            self.assertEqual([p for p, _, _, _ in calls].count("eimageget"), 6)

    def test_extra_params_reject_reserved_and_path_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)
            flare = self._write_flare_with_gti(cfg)

            for attr, bad_dict in (
                ("evqpb_extra_params", {"outset": "bad.fits"}),
                ("evqpb_extra_params", {"dir_opt": "/abs/path"}),
                ("eimageget_extra_params", {"pimin": [500]}),
                ("eimageget_extra_params", {"bad_path": Path("/tmp/foo")}),
                ("eimagecombine_extra_params", {"withasmooth": False}),
                ("eimagecombine_extra_params", {"win_path": r"foo\bar"}),
            ):
                bad_cfg = AnaConfig.from_toml(toml_path)
                setattr(bad_cfg.qpb_image, attr, bad_dict)
                with self.assertRaises(ValueError):
                    run_qpb_image(bad_cfg, flare=flare, dry_run=True)


if __name__ == "__main__":
    unittest.main()
