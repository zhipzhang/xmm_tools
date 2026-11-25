"""Tests for the ESAS flare stage (espfilt + evselect FLAG + manifest.json)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from xmm_tools.esas import (
    ESASConfig,
    load_flare_result,
    render_template_toml,
    run_flare,
)
from xmm_tools.sas.runner import RunResult, SASCommandError


class TestESASFlare(unittest.TestCase):
    def _write_workspace(self, root: Path) -> ESASConfig:
        for name in ("mos1_evt.fits", "mos2_evt.fits", "pn_evt.fits", "pn_oot_evt.fits"):
            (root / name).write_text("evt", encoding="utf-8")
        toml_path = root / "esas.toml"
        toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
        return ESASConfig.from_toml(toml_path)

    def test_run_flare_discovers_real_esas_prefixes_and_isolates_intermediates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)
            calls: list[tuple[str, dict]] = []
            prefix_map = {
                "mos1_evt.fits": "mos1S001",
                "mos2_evt.fits": "mos2S002",
                "pn_evt.fits": "pnS003",
            }

            def fake_run(self, cmd, *, force=False):
                calls.append((cmd.program, dict(cmd.params)))
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "espfilt":
                    prefix = prefix_map[cmd.params["eventfile"]]
                    for suffix in ("-allevc.fits", "-hist.qdp", "-gti.fits", "-fovlc.fits"):
                        (cmd.cwd_path / f"{prefix}{suffix}").write_text(
                            "fits_data", encoding="utf-8"
                        )
                    if cmd.params.get("withoot") == "Y":
                        (cmd.cwd_path / f"{prefix}-allevcoot.fits").write_text(
                            "fits_data", encoding="utf-8"
                        )
                else:
                    for out in cmd.core_outputs:
                        (cmd.cwd_path / out).write_text("fits_data", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                res = run_flare(cfg)

            # 3 espfilt calls (mos1, mos2, pn) + 4 evselect calls (mos1, mos2, pn, pn_oot)
            esp_calls = [c for c in calls if c[0] == "espfilt"]
            ev_calls = [c for c in calls if c[0] == "evselect"]
            self.assertEqual(len(esp_calls), 3)
            self.assertEqual(len(ev_calls), 4)
            self.assertEqual(esp_calls[2][1].get("withoot"), "Y")
            self.assertEqual(esp_calls[2][1].get("ootfile"), "pn_oot_evt.fits")
            self.assertEqual(
                [c[1]["table"] for c in ev_calls],
                [
                    "mos1S001-allevc.fits",
                    "mos2S002-allevc.fits",
                    "pnS003-allevc.fits",
                    "pnS003-allevcoot.fits",
                ],
            )

            # Intermediate espfilt products moved to _intermediate/, core outputs kept
            self.assertTrue((res.flare_dir / "mos1S001-allevc.fits").exists())
            self.assertFalse((res.flare_dir / "mos1S001-fovlc.fits").exists())
            self.assertTrue(
                (res.flare_dir / "_intermediate" / "mos1S001-fovlc.fits").exists()
            )

            self.assertTrue(res.flag_filtered)
            self.assertTrue(res.manifest_path.exists())
            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "flare")
            self.assertTrue(manifest["flag_filtered"])
            self.assertEqual(
                manifest["clean_events"]["mos1"], "mos1S001-allevc.fits"
            )
            self.assertEqual(
                manifest["clean_events"]["pn_oot"], "pnS003-allevcoot.fits"
            )

            # Re-running without force should short-circuit via manifest.json
            calls.clear()
            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                res2 = run_flare(cfg)
            self.assertEqual(calls, [])
            self.assertTrue(res2.flag_filtered)

            loaded = load_flare_result(cfg)
            self.assertEqual(loaded.clean_events["mos1"].name, "mos1S001-allevc.fits")

    def test_run_flare_reuses_existing_instrument_outputs_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)
            flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
            flare_dir.mkdir(parents=True)
            for suffix in ("-allevc.fits", "-hist.qdp", "-gti.fits"):
                (flare_dir / f"mos1S001{suffix}").write_text("m1", encoding="utf-8")

            esp_eventfiles: list[str] = []

            def fake_run(self, cmd, *, force=False):
                if cmd.program == "espfilt":
                    esp_eventfiles.append(cmd.params["eventfile"])
                    prefix = "mos2U002" if cmd.params["eventfile"] == "mos2_evt.fits" else "pnS003"
                    for suffix in ("-allevc.fits", "-hist.qdp", "-gti.fits"):
                        (cmd.cwd_path / f"{prefix}{suffix}").write_text("ok", encoding="utf-8")
                    if cmd.params.get("withoot") == "Y":
                        (cmd.cwd_path / f"{prefix}-allevcoot.fits").write_text("ok", encoding="utf-8")
                else:
                    for out in cmd.core_outputs:
                        (cmd.cwd_path / out).write_text("ok", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                res = run_flare(cfg)

            self.assertEqual(esp_eventfiles, ["mos2_evt.fits", "pn_evt.fits"])
            self.assertEqual(res.clean_events["mos1"].name, "mos1S001-allevc.fits")
            self.assertEqual(res.clean_events["mos2"].name, "mos2U002-allevc.fits")
            self.assertEqual(res.clean_events["pn"].name, "pnS003-allevc.fits")

    def test_run_flare_raises_when_espfilt_outputs_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)

            def fake_run(self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                with self.assertRaises(SASCommandError):
                    run_flare(cfg)

    def test_load_flare_result_fallback_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._write_workspace(root)
            flare_dir = cfg.paths.resolve(cfg.paths.flare_dir)
            flare_dir.mkdir(parents=True)
            (flare_dir / "mos1S001-allevc.fits").write_text("m1", encoding="utf-8")
            (flare_dir / "mos1S001-hist.qdp").write_text("1 2\n", encoding="utf-8")

            loaded = load_flare_result(cfg)
            self.assertFalse(loaded.flag_filtered)
            self.assertEqual(loaded.clean_events["mos1"], flare_dir / "mos1S001-allevc.fits")
            self.assertEqual(loaded.qdp_files["mos1"], flare_dir / "mos1S001-hist.qdp")


if __name__ == "__main__":
    unittest.main()
