"""Tests for the ESAS flare stage (espfilt + non-destructive evselect FLAG + v2 manifest)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xmm_tools.config import AnaConfig, render_template_toml
from xmm_tools.esas import FlareResult, load_flare_result, run_flare
from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    RunResult,
    SASCommandError,
    atomic_write_json,
)


class TestESASFlare(unittest.TestCase):
    def _write_workspace(self, root: Path) -> AnaConfig:
        for name in ("mos1_evt.fits", "mos2_evt.fits", "pn_evt.fits", "pn_oot_evt.fits"):
            (root / name).write_text("evt", encoding="utf-8")
        ccf = root / "ccf.cif"
        odf_sum = root / "1234_0204870101_SCX00000SUM.SAS"
        ccf.write_text("ccf_v1", encoding="utf-8")
        odf_sum.write_text("sum_v1", encoding="utf-8")
        toml_path = root / "ana.toml"
        toml_path.write_text(
            render_template_toml(
                workdir=".",
                sas_ccf="ccf.cif",
                sas_odf="1234_0204870101_SCX00000SUM.SAS",
            ),
            encoding="utf-8",
        )
        return AnaConfig.from_toml(toml_path)

    def _make_fake_run(self, calls: list[tuple[str, Path, dict]]):
        prefix_map = {
            "mos1_evt.fits": "mos1S001",
            "mos2_evt.fits": "mos2S002",
            "pn_evt.fits": "pnS003",
        }

        def fake_run(runner_self, cmd, *, force=False):
            calls.append((cmd.program, cmd.cwd_path, dict(cmd.params)))
            cmd.cwd_path.mkdir(parents=True, exist_ok=True)
            if cmd.program == "espfilt":
                prefix = prefix_map[cmd.params["eventfile"]]
                for suffix in ("-allevc.fits", "-hist.qdp", "-gti.fits", "-fovlc.fits"):
                    (cmd.cwd_path / f"{prefix}{suffix}").write_text(
                        f"unflagged_{prefix}", encoding="utf-8"
                    )
                if cmd.params.get("withoot") == "Y":
                    (cmd.cwd_path / f"{prefix}-allevcoot.fits").write_text(
                        f"unflagged_{prefix}_oot", encoding="utf-8"
                    )
            else:
                for out in cmd.expected_outputs:
                    (cmd.cwd_path / out).write_text(
                        f"flagged_{cmd.cwd_path.name}", encoding="utf-8"
                    )
            return RunResult(command=cmd, status="completed", returncode=0)

        return fake_run

    def test_non_destructive_flag_filtering_and_partial_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)
            calls: list[tuple[str, Path, dict]] = []

            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                res = run_flare(cfg)

            esp_calls = [c for c in calls if c[0] == "espfilt"]
            ev_calls = [c for c in calls if c[0] == "evselect"]
            self.assertEqual(len(esp_calls), 3)
            self.assertEqual(len(ev_calls), 4)
            self.assertEqual(esp_calls[2][2].get("withoot"), "Y")
            self.assertEqual(esp_calls[2][2].get("ootfile"), "pn_oot_evt.fits")
            self.assertEqual([c[2]["table"] for c in ev_calls], ["unflagged_events.fits"] * 4)
            self.assertEqual([c[2]["filteredset"] for c in ev_calls], ["clean_events.fits"] * 4)

            # Raw unflagged espfilt output is preserved untouched in _work/espfilt/<inst>/
            raw_m1 = res.flare_dir / "_work" / "espfilt" / "mos1" / "mos1S001-allevc.fits"
            self.assertTrue(raw_m1.is_file())
            self.assertEqual(raw_m1.read_text(encoding="utf-8"), "unflagged_mos1S001")

            # Published flare_dir symlink points to flag_filter/mos1/clean_events.fits
            pub_m1 = res.flare_dir / "mos1S001-allevc.fits"
            self.assertTrue(pub_m1.is_symlink())
            self.assertFalse(Path(os.readlink(pub_m1)).is_absolute())
            self.assertEqual(pub_m1.read_text(encoding="utf-8"), "flagged_mos1")

            # Verify v2 manifest & zero host path leakage
            self.assertTrue(res.flag_filtered)
            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["stage"], "flare")
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["products"]["flag_filtered"])
            self.assertEqual(manifest["products"]["clean_events"]["mos1"], "mos1S001-allevc.fits")
            self.assertEqual(manifest["products"]["clean_events"]["pn_oot"], "pnS003-allevcoot.fits")
            for step_raw in manifest["steps"].values():
                self.assertNotIn(str(root), json.dumps({k: v for k, v in step_raw.items() if k != "shell"}))

            # 1) Unchanged re-run reuses all 7 steps with 0 SAS calls
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                res2 = run_flare(cfg)
            self.assertEqual(calls, [])
            self.assertEqual(res2.manifest.status, "reused")
            self.assertTrue(res2.flag_filtered)

            # 2) Changing mos_flag_expr re-runs ONLY MOS1 + MOS2 evselect (espfilt x3 and PN reused!)
            cfg.flare.mos_flag_expr = "((FLAG & 0x766aa000)==0) && (PI>200)"
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                res3 = run_flare(cfg)
            self.assertEqual([(p, cwd.name) for p, cwd, _ in calls], [("evselect", "mos1"), ("evselect", "mos2")])
            self.assertEqual(res3.manifest.steps["espfilt:mos1"].status, "reused")
            self.assertEqual(res3.manifest.steps["flag_filter:pn"].status, "reused")
            self.assertEqual(res3.manifest.steps["flag_filter:pn_oot"].status, "reused")
            self.assertEqual(res3.manifest.steps["flag_filter:mos1"].status, "completed")
            self.assertEqual(res3.manifest.steps["flag_filter:mos2"].status, "completed")

            # 3) Toggling apply_flag_filter=False switches published links to unflagged events with 0 SAS calls!
            cfg.flare.apply_flag_filter = False
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                res4 = run_flare(cfg)
            self.assertEqual(calls, [])
            self.assertFalse(res4.flag_filtered)
            self.assertEqual(pub_m1.read_text(encoding="utf-8"), "unflagged_mos1S001")
            self.assertEqual(res4.manifest.status, "reused")

            # 4) Toggling apply_flag_filter=True back reuses cached flag_filter:* steps with 0 SAS calls!
            cfg.flare.apply_flag_filter = True
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                res5 = run_flare(cfg)
            self.assertEqual(calls, [])
            self.assertTrue(res5.flag_filtered)
            self.assertEqual(pub_m1.read_text(encoding="utf-8"), "flagged_mos1")
            self.assertEqual(res5.manifest.status, "reused")

            loaded = load_flare_result(cfg)
            self.assertEqual(loaded.clean_events["mos1"].name, "mos1S001-allevc.fits")

    def test_calibration_change_invalidates_espfilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)
            calls: list[tuple[str, Path, dict]] = []

            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                run_flare(cfg)

            (root / "ccf.cif").write_text("ccf_v2", encoding="utf-8")
            calls.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=self._make_fake_run(calls)):
                run_flare(cfg)
            self.assertEqual([c[0] for c in calls].count("espfilt"), 3)

    def test_run_flare_raises_when_espfilt_outputs_missing_and_validates_v2_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)

            def fake_run(self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                return RunResult(command=cmd, status="completed", returncode=0)

            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run),
                self.assertRaises(SASCommandError),
            ):
                run_flare(cfg)

            # load_flare_result rejects failed v2 manifest
            atomic_write_json(
                cfg.paths.resolve(cfg.paths.flare_dir) / "manifest.json",
                {"schema_version": 2, "stage": "flare", "status": "failed", "products": {}},
            )
            with self.assertRaises(RuntimeError):
                load_flare_result(cfg)

    def test_load_flare_result_fallback_without_manifest_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)
            res_dry = run_flare(cfg, dry_run=True)
            self.assertFalse((root / "soft_proton_flare").exists())
            self.assertEqual(res_dry.manifest.status, "dry_run")

            flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
            flare_dir.mkdir(parents=True)
            (flare_dir / "mos1S001-allevc.fits").write_text("m1", encoding="utf-8")
            (flare_dir / "mos1S001-hist.qdp").write_text("1 2\n", encoding="utf-8")

            loaded = load_flare_result(cfg)
            self.assertFalse(loaded.flag_filtered)
            self.assertEqual(loaded.clean_events["mos1"], flare_dir / "mos1S001-allevc.fits")
            self.assertEqual(loaded.qdp_files["mos1"], flare_dir / "mos1S001-hist.qdp")

            # Roundtrip FlareResult.save_manifest preserves workspace_root
            FlareResult(
                flare_dir=flare_dir,
                manifest_path=flare_dir / "manifest.json",
                flag_filtered=True,
                clean_events=loaded.clean_events,
                workspace_root=root,
            ).save_manifest()
            reloaded = load_flare_result(cfg)
            self.assertTrue(reloaded.flag_filtered)


if __name__ == "__main__":
    unittest.main()
