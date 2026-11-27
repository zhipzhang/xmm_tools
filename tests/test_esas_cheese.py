"""Tests for the ESAS cheese stage, SHA-256 invalidation, and SourceDetector."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xmm_tools.config import AnaConfig, render_template_toml
from xmm_tools.esas import (
    CheeseResult,
    FlareResult,
    load_cheese_result,
    run_cheese,
)
from xmm_tools.exec import MANIFEST_SCHEMA_VERSION, RunResult, atomic_write_json


class TestESASCheese(unittest.TestCase):
    def _prepare_flare(self, root: Path) -> tuple[AnaConfig, FlareResult]:
        toml_path = root / "ana.toml"
        toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
        cfg = AnaConfig.from_toml(toml_path)
        flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
        flare_dir.mkdir(parents=True)

        clean_events = {}
        for key, fname in (
            ("mos1", "mos1_evt-allevc.fits"),
            ("mos2", "mos2_evt-allevc.fits"),
            ("pn", "pn_evt-allevc.fits"),
            ("pn_oot", "pn_evt-allevcoot.fits"),
        ):
            p = flare_dir / fname
            p.write_bytes(b"A" * 2880)
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

    def test_cheese_detector_writes_v2_manifest_reuses_and_invalidates_on_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare = self._prepare_flare(root)
            cfg.cheese.extra_params = {"clobber": True}
            seen_cmds: list = []

            def fake_run(self, cmd, *, force=False):
                seen_cmds.append(cmd)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                for out in cmd.expected_outputs:
                    (cmd.cwd_path / out).write_text("fits_data", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = run_cheese(cfg)

            self.assertEqual(len(seen_cmds), 1)
            cmd = seen_cmds[0]
            self.assertEqual(cmd.program, "cheese")
            self.assertEqual(cmd.cwd_path, res.cheese_dir / "_work" / "cheese")
            self.assertEqual(cmd.params["mos1file"], "mos1_evt-allevc.fits")
            self.assertEqual(cmd.params["pnootfile"], "pn_evt-allevcoot.fits")
            self.assertEqual(cmd.params["elowlist"], [350, 2500])
            self.assertTrue(cmd.params["clobber"])
            self.assertIn("emllist.fits", cmd.expected_outputs)
            self.assertIn("mos1_evt-cheeset.fits", cmd.expected_outputs)
            self.assertIn("pn_evt-bkgregtdet.fits", cmd.expected_outputs)

            # Published symlink inside cheese_dir points to _work/cheese/
            self.assertTrue((res.cheese_dir / "emllist.fits").is_symlink())
            self.assertFalse(Path(os.readlink(res.cheese_dir / "emllist.fits")).is_absolute())

            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["stage"], "cheese")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["products"]["method"], "cheese")
            self.assertEqual(manifest["products"]["emllist"], "emllist.fits")
            self.assertNotIn(
                str(root),
                json.dumps({k: v for k, v in manifest["steps"]["cheese"].items() if k != "shell"}),
            )

            # Re-running without force reuses via StepLedger
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res2 = run_cheese(cfg)
            self.assertEqual(seen_cmds, [])
            self.assertEqual(res2.manifest.status, "reused")
            self.assertEqual(res2.cheese_masks["mos1"].name, "mos1_evt-cheeset.fits")

            # Mutating input -allevc.fits with identical 2880-byte size must invalidate cheese!
            flare.clean_events["mos1"].write_bytes(b"B" * 2880)
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res3 = run_cheese(cfg)
            self.assertEqual(len(seen_cmds), 1)
            self.assertEqual(res3.manifest.status, "completed")

    def test_cheese_rejects_reserved_and_path_extra_params_and_validates_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare = self._prepare_flare(root)

            for bad in (
                {"mos1file": "other.fits"},
                {"bad_dir": "/abs/path"},
                {"bad_path": Path("/tmp/foo")},
                {"win_path": r"a\b"},
            ):
                cfg.cheese.extra_params = bad
                with self.assertRaises(ValueError):
                    run_cheese(cfg, flare=flare, dry_run=True)

            # Failed v2 manifest rejected by load_cheese_result
            cheese_dir = cfg.paths.resolve(cfg.paths.cheese_dir)
            atomic_write_json(
                cheese_dir / "manifest.json",
                {"schema_version": 2, "stage": "cheese", "status": "failed", "products": {}},
            )
            with self.assertRaises(RuntimeError):
                load_cheese_result(cfg)

    def test_custom_source_detector_plugin_and_fallback_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, flare = self._prepare_flare(root)

            class CustomDetector:
                def detect(
                    self,
                    cfg,
                    flare,
                    *,
                    dry_run=False,
                    force=False,
                    docker_container=None,
                ) -> CheeseResult:
                    cdir = cfg.paths.resolve(cfg.paths.cheese_dir)
                    cdir.mkdir(parents=True, exist_ok=True)
                    mask = cdir / "mos1_evt-cheeset.fits"
                    mask.write_text("custom_mask", encoding="utf-8")
                    res = CheeseResult(
                        cheese_dir=cdir,
                        manifest_path=cdir / "manifest.json",
                        method="custom_photutils",
                        cheese_masks={"mos1": mask},
                        workspace_root=root,
                    )
                    res.save_manifest()
                    return res

            res = run_cheese(cfg, flare=flare, detector=CustomDetector())
            self.assertEqual(res.method, "custom_photutils")
            loaded = load_cheese_result(cfg)
            self.assertEqual(loaded.method, "custom_photutils")
            self.assertEqual(loaded.cheese_masks["mos1"].name, "mos1_evt-cheeset.fits")


if __name__ == "__main__":
    unittest.main()
