"""Tests for ``xmm_tools.pipeline.engine`` and ``xmm-ana run`` CLI."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from xmm_tools.esas import CheeseResult, FlareResult, SpectrumResult
from xmm_tools.pipeline import (
    PIPELINE_STAGES,
    render_template_toml,
    resolve_stages,
    run_pipeline,
)
from xmm_tools.pipeline.cli import main as cli_main
from xmm_tools.sas import QPBImageResult


class TestResolveStages(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(resolve_stages(), list(PIPELINE_STAGES))
        self.assertEqual(
            PIPELINE_STAGES, ("flare", "cheese", "qpb-image", "spectrum")
        )

    def test_explicit_stages_sorted_canonically_with_aliases(self):
        self.assertEqual(
            resolve_stages(stages=["spectrum,espfilt", "qpb_image"]),
            ["flare", "qpb-image", "spectrum"],
        )

    def test_from_to_and_skip(self):
        self.assertEqual(
            resolve_stages(from_stage="cheese", to_stage="spectrum"),
            ["cheese", "qpb-image", "spectrum"],
        )
        self.assertEqual(
            resolve_stages(skip=["qpb-image"]),
            ["flare", "cheese", "spectrum"],
        )

    def test_invalid_combinations_raise(self):
        with self.assertRaises(ValueError):
            resolve_stages(stages=["flare"], from_stage="cheese")
        with self.assertRaises(ValueError):
            resolve_stages(from_stage="spectrum", to_stage="flare")
        with self.assertRaises(ValueError):
            resolve_stages(stages=["init"])
        with self.assertRaises(ValueError):
            resolve_stages(stages=["unknown_stage"])
        with self.assertRaises(ValueError):
            resolve_stages(stages=["flare"], skip=["flare"])


class TestRunPipeline(unittest.TestCase):
    def _write_config(self, root: Path, *, with_regions: bool = False) -> Path:
        toml_path = root / "ana.toml"
        text = render_template_toml(workdir=".")
        if with_regions:
            text += (
                '\n[[esas.spectrum.regions]]\n'
                'name = "bg"\n'
                'file = "regions/bg.reg"\n'
                'mode = "single"\n'
            )
        toml_path.write_text(text, encoding="utf-8")
        return toml_path

    def test_default_pipeline_skips_spectrum_when_no_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root, with_regions=False)
            order: list[str] = []

            def fake_flare(cfg, **kw):
                order.append("flare")
                return FlareResult(
                    flare_dir=root / "soft_proton_flare",
                    manifest_path=root / "soft_proton_flare" / "manifest.json",
                    flag_filtered=True,
                )

            def fake_cheese(cfg, **kw):
                order.append("cheese")
                return CheeseResult(
                    cheese_dir=root / "cheese",
                    manifest_path=root / "cheese" / "manifest.json",
                    method="cheese",
                )

            def fake_qi(cfg, **kw):
                order.append("qpb-image")
                return QPBImageResult(
                    qpb_image_dir=root / "qpb_image",
                    manifest_path=root / "qpb_image" / "manifest.json",
                )

            with (
                patch("xmm_tools.pipeline.engine.run_flare", side_effect=fake_flare),
                patch("xmm_tools.pipeline.engine.run_cheese", side_effect=fake_cheese),
                patch("xmm_tools.pipeline.engine.run_qpb_image", side_effect=fake_qi),
            ):
                res = run_pipeline(cfg_path)

            self.assertEqual(order, ["flare", "cheese", "qpb-image"])
            self.assertEqual(res.stages_run, ["flare", "cheese", "qpb-image"])
            self.assertIn("spectrum", res.stages_skipped)

    def test_explicit_spectrum_without_regions_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root, with_regions=False)
            with self.assertRaises(ValueError):
                run_pipeline(cfg_path, stages=["spectrum"])

    def test_full_pipeline_via_cli_and_selective_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root, with_regions=True)
            order: list[str] = []

            def fake_flare(cfg, **kw):
                order.append("flare")
                return FlareResult(
                    flare_dir=root / "soft_proton_flare",
                    manifest_path=root / "soft_proton_flare" / "manifest.json",
                    flag_filtered=True,
                )

            def fake_cheese(cfg, **kw):
                order.append("cheese")
                return CheeseResult(
                    cheese_dir=root / "cheese",
                    manifest_path=root / "cheese" / "manifest.json",
                    method="cheese",
                )

            def fake_qi(cfg, **kw):
                order.append("qpb-image")
                return QPBImageResult(
                    qpb_image_dir=root / "qpb_image",
                    manifest_path=root / "qpb_image" / "manifest.json",
                )

            def fake_spec(cfg, **kw):
                order.append("spectrum")
                return SpectrumResult(
                    spectrum_dir=root / "spectrum",
                    manifest_path=root / "spectrum" / "manifest.json",
                )

            with (
                patch("xmm_tools.pipeline.engine.run_flare", side_effect=fake_flare),
                patch("xmm_tools.pipeline.engine.run_cheese", side_effect=fake_cheese),
                patch("xmm_tools.pipeline.engine.run_qpb_image", side_effect=fake_qi),
                patch("xmm_tools.pipeline.engine.run_spectrum", side_effect=fake_spec),
            ):
                rc = cli_main(["run", "--config", str(cfg_path), "--dry-run"])
                self.assertEqual(rc, 0)
                self.assertEqual(
                    order, ["flare", "cheese", "qpb-image", "spectrum"]
                )

                order.clear()
                rc2 = cli_main(
                    [
                        "run",
                        "--config",
                        str(cfg_path),
                        "--stages",
                        "cheese,spectrum",
                        "--only",
                        "bg",
                    ]
                )
                self.assertEqual(rc2, 0)
                self.assertEqual(order, ["cheese", "spectrum"])


if __name__ == "__main__":
    unittest.main()
