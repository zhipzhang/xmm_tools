"""Unit tests for ``xmm_tools.config`` schema, single-pass loader, and validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tomllib

from xmm_tools.config import (
    VALID_BKGRATE_STYLES,
    VALID_RADIUS_STYLES,
    AnaConfig,
    BandSpec,
    EspfiltConfig,
    FlareConfig,
    MaskConfig,
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
        self.assertIs(legacy_config_shim.MaskConfig, MaskConfig)
        self.assertEqual(legacy_config_shim.VALID_RADIUS_STYLES, VALID_RADIUS_STYLES)
        self.assertEqual(legacy_config_shim.VALID_BKGRATE_STYLES, VALID_BKGRATE_STYLES)

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
        self.assertEqual(SpectrumConfig().ehigh, 7200)
        with self.assertRaises(ValueError):
            SpectrumConfig(max_workers=0)
        with self.assertRaises(ValueError):
            SpectrumConfig(elow=5000, ehigh=1000)
        with self.assertRaises(ValueError):
            SpectrumConfig(elow=0, ehigh=7200)
        with self.assertRaises(ValueError):
            SpectrumConfig(elow=400, ehigh=11996)
        with self.assertRaises(ValueError):
            SpectrumRegionSpec(name="bad_low", file="a.reg", elow=0, ehigh=7200)
        with self.assertRaises(ValueError):
            SpectrumRegionSpec(name="bad_high", file="a.reg", ehigh=11996)
        with self.assertRaises(ValueError):
            SpectrumRegionSpec(name="bad_order", file="a.reg", elow=5000, ehigh=1000)
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

    def test_validation_rejects_invalid_mask_params(self):
        with self.assertRaises(ValueError):
            MaskConfig(radius_style="invalid")
        with self.assertRaises(ValueError):
            MaskConfig(bkgrate_style="median")
        for bad_pos in (0.0, -0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MaskConfig(bkg_fraction=bad_pos)
            with self.assertRaises(ValueError):
                MaskConfig(bkgrate_user=bad_pos)
            with self.assertRaises(ValueError):
                MaskConfig(fixed_radius=bad_pos)
        for bad_eef in (0.0, 1.0, -0.1, 1.2, float("nan")):
            with self.assertRaises(ValueError):
                MaskConfig(energy_fraction=bad_eef)
        with self.assertRaises(TypeError):
            MaskConfig(bkg_fraction=True)
        with self.assertRaises(TypeError):
            MaskConfig(bkgrate_user=False)
        with self.assertRaises(TypeError):
            MaskConfig(energy_fraction=True)
        with self.assertRaises(TypeError):
            MaskConfig(fixed_radius=True)
        with self.assertRaises(TypeError):
            MaskConfig(nosrcellipse="false")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            MaskConfig(expression=123)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp:
            bad_bool_toml = Path(tmp) / "bad_mask_bool.toml"
            bad_bool_toml.write_text(
                '[esas.mask]\nnosrcellipse = "false"\n', encoding="utf-8"
            )
            with self.assertRaises(TypeError):
                load_ana_config(bad_bool_toml)

            bad_num_toml = Path(tmp) / "bad_mask_num.toml"
            bad_num_toml.write_text(
                "[esas.mask]\nbkg_fraction = true\n", encoding="utf-8"
            )
            with self.assertRaises(TypeError):
                load_ana_config(bad_num_toml)

    def test_mask_and_mask_dir_toml_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "ana.toml"
            rendered = render_template_toml(workdir=".")
            toml_path.write_text(rendered, encoding="utf-8")
            cfg_default = load_ana_config(toml_path)
            self.assertEqual(cfg_default.mask, MaskConfig())
            self.assertEqual(cfg_default.spectrum.mask_dir, "")

            custom_toml = (
                rendered.replace('radius_style = "contour"', 'radius_style = "enfrac"')
                .replace("energy_fraction = 0.85", "energy_fraction = 0.92")
                .replace(
                    '# mask_dir = ""',
                    'mask_dir = "custom_masks"',
                )
            )
            toml_path.write_text(custom_toml, encoding="utf-8")
            cfg_custom = AnaConfig.from_toml(toml_path)
            self.assertEqual(cfg_custom.mask.radius_style, "enfrac")
            self.assertAlmostEqual(cfg_custom.mask.energy_fraction, 0.92)
            self.assertEqual(cfg_custom.spectrum.mask_dir, "custom_masks")

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
