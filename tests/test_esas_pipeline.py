"""Tests for ``xmm_tools.pipeline.engine`` and ``xmm-ana`` CLI."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xmm_tools.esas import CheeseResult, FlareResult, SpectrumResult
from xmm_tools.exec import StageManifest
from xmm_tools.pipeline import (
    PIPELINE_STAGES,
    AnaConfig,
    render_template_toml,
    resolve_stages,
    run_pipeline,
)
from xmm_tools.pipeline.cli import main as cli_main
from xmm_tools.pipeline.engine import _build_stage_registry
from xmm_tools.sas import ODFInitResult, QPBImageResult


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

    def test_spectrum_requires_depends_on_withsrcrem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "ana.toml"
            cfg_path.write_text(
                render_template_toml(workdir=".").replace(
                    "withsrcrem = true", "withsrcrem = false"
                ),
                encoding="utf-8",
            )
            cfg = AnaConfig.from_toml(cfg_path)
            reg_default = {s.name: s for s in _build_stage_registry()}
            reg_no_mask = {s.name: s for s in _build_stage_registry(config=cfg)}
            self.assertEqual(reg_default["spectrum"].requires, ("flare", "cheese"))
            self.assertEqual(reg_no_mask["spectrum"].requires, ("flare",))


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
                    manifest=StageManifest(stage="flare", status="reused"),
                )

            def fake_cheese(cfg, **kw):
                order.append("cheese")
                return CheeseResult(
                    cheese_dir=root / "cheese",
                    manifest_path=root / "cheese" / "manifest.json",
                    method="cheese",
                    manifest=StageManifest(stage="cheese", status="completed"),
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
            self.assertEqual(
                res.stage_statuses,
                {"flare": "reused", "cheese": "completed", "qpb-image": "completed"},
            )
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
                    manifest=StageManifest(stage="flare", status="dry_run"),
                )

            def fake_cheese(cfg, **kw):
                order.append("cheese")
                return CheeseResult(
                    cheese_dir=root / "cheese",
                    manifest_path=root / "cheese" / "manifest.json",
                    method="cheese",
                    manifest=StageManifest(stage="cheese", status="reused"),
                )

            def fake_qi(cfg, **kw):
                order.append("qpb-image")
                return QPBImageResult(
                    qpb_image_dir=root / "qpb_image",
                    manifest_path=root / "qpb_image" / "manifest.json",
                    manifest=StageManifest(stage="qpb-image", status="dry_run"),
                )

            def fake_spec(cfg, **kw):
                order.append("spectrum")
                return SpectrumResult(
                    spectrum_dir=root / "spectrum",
                    manifest_path=root / "spectrum" / "manifest.json",
                    manifest=StageManifest(stage="spectrum", status="completed"),
                )

            with (
                patch("xmm_tools.pipeline.engine.run_flare", side_effect=fake_flare),
                patch("xmm_tools.pipeline.engine.run_cheese", side_effect=fake_cheese),
                patch("xmm_tools.pipeline.engine.run_qpb_image", side_effect=fake_qi),
                patch("xmm_tools.pipeline.engine.run_spectrum", side_effect=fake_spec),
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = cli_main(["run", "--config", str(cfg_path), "--dry-run"])
                self.assertEqual(rc, 0)
                self.assertEqual(
                    order, ["flare", "cheese", "qpb-image", "spectrum"]
                )
                self.assertIn("flare (dry_run)", buf.getvalue())
                self.assertIn("cheese (reused)", buf.getvalue())

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

    def test_cli_init_select_valid_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odf_dir = root / "ODF"
            work_dir = root / "work"
            odf_dir.mkdir()

            captured: dict[str, object] = {}

            def fake_init(**kwargs):
                captured.update(kwargs)
                return ODFInitResult(
                    workdir=work_dir,
                    config_path=work_dir / "ana.toml",
                    manifest_path=work_dir / "init_manifest.json",
                    ccf_path=work_dir / "ccf.cif",
                    odf_sum_path=work_dir / "SUM.SAS",
                    event_files={"mos1": work_dir / "mos1_evt.fits"},
                    manifest=StageManifest(stage="init", status="completed"),
                )

            buf = io.StringIO()
            with (
                patch("xmm_tools.pipeline.cli.initialize_observation", side_effect=fake_init),
                contextlib.redirect_stdout(buf),
            ):
                rc = cli_main(
                    [
                        "init",
                        "--odf-dir",
                        str(odf_dir),
                        "--workdir",
                        str(work_dir),
                        "--select",
                        "mos1=P0923400201M1S001MIEVLI0000.FIT",
                        "--select",
                        "PN=P0923400201PNS003PIEVLI0000.FIT",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(
                captured["exposure_selection"],
                {
                    "mos1": "P0923400201M1S001MIEVLI0000.FIT",
                    "pn": "P0923400201PNS003PIEVLI0000.FIT",
                },
            )
            self.assertIn("[status=completed]", buf.getvalue())

            invalid_cases = [
                ["init", "--odf-dir", str(odf_dir), "--select", "mos1"],
                ["init", "--odf-dir", str(odf_dir), "--select", "mos1="],
                ["init", "--odf-dir", str(odf_dir), "--select", "rgs1=foo.fits"],
                [
                    "init",
                    "--odf-dir",
                    str(odf_dir),
                    "--select",
                    "mos1=a.fit",
                    "--select",
                    "MOS1=b.fit",
                ],
            ]
            for argv in invalid_cases:
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as ctx,
                ):
                    cli_main(argv)
                self.assertEqual(ctx.exception.code, 2)

    def test_cli_individual_stage_subcommands_report_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root, with_regions=True)

            flare_res = FlareResult(
                flare_dir=root / "soft_proton_flare",
                manifest_path=root / "soft_proton_flare" / "manifest.json",
                flag_filtered=True,
                manifest=StageManifest(stage="flare", status="reused"),
            )
            cheese_res = CheeseResult(
                cheese_dir=root / "cheese",
                manifest_path=root / "cheese" / "manifest.json",
                method="cheese",
                manifest=StageManifest(stage="cheese", status="completed"),
            )
            qi_res = QPBImageResult(
                qpb_image_dir=root / "qpb_image",
                manifest_path=root / "qpb_image" / "manifest.json",
                manifest=StageManifest(stage="qpb-image", status="reused"),
            )
            spec_res = SpectrumResult(
                spectrum_dir=root / "spectrum",
                manifest_path=root / "spectrum" / "manifest.json",
                manifest=StageManifest(stage="spectrum", status="completed"),
            )

            with (
                patch("xmm_tools.pipeline.cli.run_flare", return_value=flare_res),
                patch("xmm_tools.pipeline.cli.run_cheese", return_value=cheese_res),
                patch("xmm_tools.pipeline.cli.run_qpb_image", return_value=qi_res),
                patch("xmm_tools.pipeline.cli.run_spectrum", return_value=spec_res),
            ):
                for subcmd, expected in (
                    ("flare", "[status=reused]"),
                    ("cheese", "[status=completed]"),
                    ("qpb-image", "[status=reused]"),
                    ("spectrum", "[status=completed]"),
                ):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        rc = cli_main([subcmd, "--config", str(cfg_path)])
                    self.assertEqual(rc, 0)
                    self.assertIn(expected, buf.getvalue())


if __name__ == "__main__":
    unittest.main()
