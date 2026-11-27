"""Unit tests for ``xmm_tools.config`` schema, single-pass loader, and validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tomllib

from xmm_tools.config import (
    AnaConfig,
    BandSpec,
    EspfiltConfig,
    FlareConfig,
    SpectrumConfig,
    SpectrumRegionSpec,
    load_ana_config,
    render_template_toml,
)
from xmm_tools.pipeline import config as legacy_config_shim


class TestConfigPackage(unittest.TestCase):
    def test_backward_compatible_shim_exports(self):
        self.assertIs(legacy_config_shim.AnaConfig, AnaConfig)
        self.assertIs(legacy_config_shim.FlareConfig, FlareConfig)
        self.assertIs(legacy_config_shim.EspfiltConfig, FlareConfig)
        self.assertIs(EspfiltConfig, FlareConfig)

    def test_single_pass_load_and_flare_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ccf.cif").write_text("ccf", encoding="utf-8")
            (root / "SUM.SAS").write_text("odf", encoding="utf-8")
            toml_path = root / "ana.toml"
            toml_path.write_text(render_template_toml(workdir="."), encoding="utf-8")

            cfg = AnaConfig.from_toml(toml_path)
            cfg2 = load_ana_config(toml_path)
            self.assertEqual(cfg.flare, cfg2.flare)
            self.assertIs(cfg.espfilt, cfg.flare)
            self.assertEqual(cfg.env.sas_ccf, str((root / "ccf.cif").resolve()))
            self.assertEqual(cfg.qpb_image.dir, cfg.paths.qpb_image_dir)

    def test_esas_flare_table_overrides_or_replaces_espfilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "ana.toml"
            toml_path.write_text(
                """\
[paths]
workdir = "."

[esas.flare]
elow = 3000
ehigh = 9000
allowsigma = 2.5
""",
                encoding="utf-8",
            )
            cfg = AnaConfig.from_toml(toml_path)
            self.assertEqual(cfg.flare.elow, 3000)
            self.assertEqual(cfg.flare.ehigh, 9000)
            self.assertEqual(cfg.flare.allowsigma, 2.5)
            self.assertEqual(cfg.espfilt.elow, 3000)

    def test_validation_rejects_invalid_bands(self):
        with self.assertRaises(ValueError):
            BandSpec(name="bad", emin=2000, emax=1000)
        with self.assertRaises(ValueError):
            BandSpec(name="neg", emin=-10, emax=1000)
        with self.assertRaises(ValueError):
            BandSpec(
                name="out_of_range",
                emin=500,
                emax=2000,
                subbands=[[400, 1000]],
            )
        with self.assertRaises(ValueError):
            BandSpec(
                name="inverted_subband",
                emin=500,
                emax=2000,
                subbands=[[1500, 1000]],
            )

        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "bad_band.toml"
            toml_path.write_text(
                """\
[sas.qpb_image.bands.soft]
emin = 500
emax = 2000
subbands = [[500, 2500]]
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                AnaConfig.from_toml(toml_path)

    def test_validation_rejects_invalid_spectrum_and_duplicate_regions(self):
        with self.assertRaises(ValueError):
            SpectrumConfig(max_workers=0)
        with self.assertRaises(ValueError):
            SpectrumConfig(elow=5000, ehigh=1000)
        with self.assertRaises(ValueError):
            SpectrumConfig(
                regions=[
                    SpectrumRegionSpec(name="core", file="a.reg"),
                    SpectrumRegionSpec(name="core", file="b.reg"),
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "dup_regions.toml"
            toml_path.write_text(
                render_template_toml(workdir=".")
                + '\n[[esas.spectrum.regions]]\nname = "core"\nfile = "a.reg"\n'
                + '\n[[esas.spectrum.regions]]\nname = "core"\nfile = "b.reg"\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                AnaConfig.from_toml(toml_path)

    def test_validation_rejects_invalid_flare_params(self):
        with self.assertRaises(ValueError):
            FlareConfig(elow=8500, ehigh=2500)
        with self.assertRaises(ValueError):
            FlareConfig(allowsigma=0.0)

    def test_legacy_espfilt_constructor_and_template_escaping(self):
        cfg = AnaConfig(espfilt=FlareConfig(allowsigma=2.5))
        self.assertEqual(cfg.flare.allowsigma, 2.5)
        self.assertIs(cfg.espfilt, cfg.flare)

        rendered = render_template_toml(
            workdir=r'C:\xmm\obs "quoted"',
            sas_ccf=r'sub\ccf "1".cif',
            sas_ccfpath=r'/ccf/path\with"quote',
            sas_odf="SUM.SAS",
        )
        parsed = tomllib.loads(rendered)
        self.assertEqual(parsed["paths"]["workdir"], r'C:\xmm\obs "quoted"')
        self.assertEqual(parsed["env"]["sas_ccf"], r'sub\ccf "1".cif')
        self.assertEqual(parsed["env"]["sas_ccfpath"], r'/ccf/path\with"quote')


if __name__ == "__main__":
    unittest.main()
