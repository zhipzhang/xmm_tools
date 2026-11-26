"""Tests for quick_image stage (evqpb + eimageget + eimagecombine)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astropy.io import fits
import numpy as np

from xmm_tools.esas import FlareResult
from xmm_tools.exec.runner import RunResult
from xmm_tools.pipeline import AnaConfig, BandSpec, render_template_toml
from xmm_tools.sas import resolve_subbands, run_qpb_image


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

    def test_resolve_subbands_explicit_and_auto_split(self):
        edges = [350, 500, 1000, 2000, 4500, 7000]
        explicit = BandSpec(name="soft", emin=500, emax=2000, subbands=[[500, 1200], [1200, 2000]])
        self.assertEqual(resolve_subbands(explicit, edges), [(500, 1200), (1200, 2000)])

        auto = BandSpec(name="full", emin=500, emax=7000)
        self.assertEqual(
            resolve_subbands(auto, edges),
            [(500, 1000), (1000, 2000), (2000, 4500), (4500, 7000)],
        )

    def test_run_qpb_image_uses_raw_events_and_gti_and_merges_subbands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)
            flare = self._write_flare_with_gti(cfg)

            calls: list[tuple[str, dict, Path]] = []
            inst_tags = {
                "mos1_evt.fits": "P0923400301M1S001",
                "mos2_evt.fits": "P0923400301M2S002",
                "pn_evt.fits": "P0923400301PNS003",
            }

            def fake_run(self, cmd, *, force=False):
                calls.append((cmd.program, dict(cmd.params), cmd.cwd_path))
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    tag = inst_tags[Path(cmd.params["evtfile"]).name]
                    for idx in range(len(cmd.params["pimin"])):
                        (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text("ima", encoding="utf-8")
                        (cmd.cwd_path / f"{tag}_ima_{idx}exp.fits").write_text("exp", encoding="utf-8")
                        (cmd.cwd_path / f"{tag}_ima_{idx}fwc.fits").write_text("fwc", encoding="utf-8")
                elif cmd.program == "eimagecombine":
                    prefix = cmd.params["prefix"]
                    # Subband 0: counts=3.0 / exp=2.0 -> 1.5; Subband 1: counts=5.0 / exp=2.0 -> 2.5
                    for idx, (cnt_val, exp_val) in enumerate(((3.0, 2.0), (5.0, 2.0))):
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
                    mask = np.ones((4, 4), dtype=np.float32)
                    mask[0, 0] = 0.0
                    fits.PrimaryHDU(data=mask).writeto(
                        cmd.cwd_path / f"{prefix}ima_mask.fits", overwrite=True
                    )
                else:
                    for out in cmd.core_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_qpb_image(cfg, flare=flare)

            progs = [p for p, _, _ in calls]
            # 1 atthkgen + 3 evqpb + 2 bands * 3 eimageget + 2 eimagecombine
            self.assertEqual(progs.count("atthkgen"), 1)
            self.assertEqual(progs.count("evqpb"), 3)
            self.assertEqual(progs.count("eimageget"), 6)
            self.assertEqual(progs.count("eimagecombine"), 2)

            # Verify evqpb and eimageget use raw event files + fwcfile + gtifile
            evqpb_calls = [params for p, params, _ in calls if p == "evqpb"]
            self.assertEqual(
                [Path(c["table"]).name for c in evqpb_calls],
                ["mos1_evt.fits", "mos2_evt.fits", "pn_evt.fits"],
            )

            eimg_calls = [params for p, params, _ in calls if p == "eimageget"]
            self.assertEqual(Path(eimg_calls[0]["evtfile"]).name, "mos1_evt.fits")
            self.assertEqual(Path(eimg_calls[0]["fwcfile"]).name, "mos1_qpb.fits")
            self.assertEqual(Path(eimg_calls[0]["gtifile"]).name, "mos1S001-gti.fits")
            self.assertEqual(eimg_calls[0]["pimin"], [500, 1000])
            self.assertEqual(eimg_calls[0]["pimax"], [1000, 2000])
            self.assertNotIn("ootfile", eimg_calls[0])
            self.assertEqual(Path(eimg_calls[2]["ootfile"]).name, "pn_oot_evt.fits")

            # Verify summed subband flux image has expected values (1.5 + 2.5 = 4.0, masked [0,0] = 0.0)
            self.assertEqual(res.band_flux_images["soft"].name, "soft_500_2000_flux.fits")
            self.assertEqual(res.band_flux_images["hard"].name, "hard_2000_7000_flux.fits")
            with fits.open(res.band_flux_images["soft"]) as hdul:
                expected = np.full((4, 4), 4.0, dtype=np.float32)
                expected[0, 0] = 0.0
                np.testing.assert_allclose(hdul[0].data, expected)
                self.assertEqual(hdul[0].header.get("BUNIT"), "counts/s")

            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "qpb_image")
            self.assertEqual(manifest["band_flux_images"]["soft"], "soft_500_2000_flux.fits")
            self.assertIn("soft", res.bands)
            self.assertEqual(len(res.bands["soft"].subbands), 2)
            self.assertTrue(res.bands["soft"].mask_image.is_file())
            self.assertEqual(res.bands["soft"].subbands[0].counts_image.name, "soft_500_1000_counts.fits")
            self.assertEqual(res.bands["soft"].subbands[0].exposure_image.name, "soft_500_1000_exp.fits")
            reloaded = type(res).from_manifest(res.manifest_path)
            self.assertEqual(len(reloaded.bands["soft"].subbands), 2)
            self.assertEqual(reloaded.bands["soft"].subbands[1].emin, 1000)
            self.assertEqual(reloaded.bands["soft"].subbands[1].emax, 2000)

            # Re-running without force skips via manifest.json
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                run_qpb_image(cfg, flare=flare)
            self.assertEqual(calls, [])

            # Re-running with force=True cleans stale files in band_dir first
            stale_file = res.qpb_image_dir / "_work" / "soft" / "stale_old_bin_ima_0.fits"
            stale_file.write_text("stale", encoding="utf-8")
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                run_qpb_image(cfg, flare=flare, force=True)
            self.assertFalse(stale_file.exists())

    def test_run_qpb_image_raises_when_gti_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)

            with self.assertRaises(FileNotFoundError):
                run_qpb_image(cfg)

    def test_quick_image_gti_override_in_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            custom_dir = root / "custom_gti"
            custom_dir.mkdir()
            for inst in ("mos1", "mos2", "pn"):
                (custom_dir / f"{inst}.gti").write_text("gti", encoding="utf-8")

            toml_text = (
                render_template_toml(workdir=".")
                + '\n[sas.qpb_image.gti_files]\n'
                + 'mos1 = "custom_gti/mos1.gti"\n'
                + 'mos2 = "custom_gti/mos2.gti"\n'
                + 'pn = "custom_gti/pn.gti"\n'
            )
            toml_path = root / "esas.toml"
            toml_path.write_text(toml_text, encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)

            seen_gtis: list[str] = []

            def fake_run(self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    seen_gtis.append(Path(cmd.params["gtifile"]).name)
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
                else:
                    for out in cmd.core_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_qpb_image(cfg)

            self.assertEqual(seen_gtis[:3], ["mos1.gti", "mos2.gti", "pn.gti"])
            self.assertTrue(res.band_flux_images["soft"].exists())

    def test_eimagecombine_isolated_dir_and_validation(self):
        from xmm_tools.exec.runner import SASCommandError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_events(root)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = AnaConfig.from_toml(toml_path)
            flare = self._write_flare_with_gti(cfg)

            combine_cwds: list[Path] = []

            def fake_run_incomplete(self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    tag = Path(cmd.params["evtfile"]).stem
                    for idx in (0, 1):
                        (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text("ima", encoding="utf-8")
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
                    for out in cmd.core_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_incomplete):
                with self.assertRaises(SASCommandError) as ctx:
                    run_qpb_image(cfg, flare=flare)
            self.assertIn("incomplete sub-bands", str(ctx.exception))
            self.assertEqual(len(combine_cwds), 1)
            self.assertNotEqual(combine_cwds[0], cfg.paths.resolve("qpb_image/_work/soft"))

            # Also fail fast when combine_log reports cp skipping file
            def fake_run_log_error(self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "eimageget":
                    tag = Path(cmd.params["evtfile"]).stem
                    for idx in (0, 1):
                        (cmd.cwd_path / f"{tag}_ima_{idx}.fits").write_text("ima", encoding="utf-8")
                elif cmd.program == "eimagecombine":
                    combine_cwds.append(cmd.cwd_path)
                    (cmd.cwd_path / cmd.log_file).write_text(
                        "cp: skipping file 'x_PN_ima_0_sub.fits', as it was replaced while being copied\n",
                        encoding="utf-8",
                    )
                else:
                    for out in cmd.core_outputs:
                        (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_log_error):
                with self.assertRaises(SASCommandError) as ctx2:
                    run_qpb_image(cfg, flare=flare, force=True)
            self.assertIn("skipping file", str(ctx2.exception))
            self.assertEqual(len(combine_cwds), 2)
            self.assertNotEqual(combine_cwds[0], combine_cwds[1])


if __name__ == "__main__":
    unittest.main()
