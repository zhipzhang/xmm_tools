"""Tests for quick_image stage (evqpb + eimageget + eimagecombine)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from xmm_tools.esas import (
    BandSpec,
    ESASConfig,
    FlareResult,
    render_template_toml,
    resolve_subbands,
    run_quick_image,
)
from xmm_tools.sas.runner import RunResult


class TestESASQuickImage(unittest.TestCase):
    def test_resolve_subbands_explicit_and_auto_split(self):
        edges = [350, 500, 1000, 2000, 4500, 7000]
        explicit = BandSpec(name="soft", emin=500, emax=2000, subbands=[[500, 1200], [1200, 2000]])
        self.assertEqual(resolve_subbands(explicit, edges), [(500, 1200), (1200, 2000)])

        auto = BandSpec(name="full", emin=500, emax=7000)
        self.assertEqual(
            resolve_subbands(auto, edges),
            [(500, 1000), (1000, 2000), (2000, 4500), (4500, 7000)],
        )

    def test_run_quick_image_splits_subbands_and_combines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "esas.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")
            cfg = ESASConfig.from_toml(toml_path)

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
                p.write_text("allevc", encoding="utf-8")
                clean_events[key] = p

            flare = FlareResult(
                flare_dir=flare_dir,
                manifest_path=flare_dir / "manifest.json",
                flag_filtered=True,
                clean_events=clean_events,
            )
            flare.save_manifest()

            calls: list[tuple[str, dict]] = []

            def fake_run(self, cmd, *, force=False):
                calls.append((cmd.program, dict(cmd.params)))
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                for out in cmd.core_outputs:
                    (cmd.cwd_path / out).write_text("fits", encoding="utf-8")
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                res = run_quick_image(cfg, flare=flare)

            progs = [p for p, _ in calls]
            # 1 atthkgen + 3 evqpb + (2+2)*3=12 eimageget + 2 eimagecombine
            self.assertEqual(progs.count("atthkgen"), 1)
            self.assertEqual(progs.count("evqpb"), 3)
            self.assertEqual(progs.count("eimageget"), 12)
            self.assertEqual(progs.count("eimagecombine"), 2)

            self.assertEqual(res.band_flux_images["soft"].name, "soft_500_2000_flux.fits")
            self.assertEqual(res.band_flux_images["hard"].name, "hard_2000_7000_flux.fits")

            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "quick_image")
            self.assertEqual(manifest["band_flux_images"]["soft"], "soft_500_2000_flux.fits")

            # Re-running without force skips via manifest.json
            calls.clear()
            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                run_quick_image(cfg, flare=flare)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
