"""Tests for ESAS TOML configuration and observation initialization."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from xmm_tools.esas import (
    ESASConfig,
    initialize_observation,
    render_template_toml,
    standardize_event_files,
)
from xmm_tools.sas.runner import RunResult


class TestESASConfig(unittest.TestCase):
    def test_template_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "esas.toml"
            toml_path.write_text(
                render_template_toml(
                    workdir=".",
                    sas_ccf="/data/ccf.cif",
                    sas_ccfpath="/ccf/valid",
                    sas_odf="/data/SUM.SAS",
                ),
                encoding="utf-8",
            )

            cfg = ESASConfig.from_toml(toml_path)
            self.assertEqual(str(cfg.env.sas_ccf), "/data/ccf.cif")
            self.assertEqual(cfg.paths.workdir, root.resolve())
            self.assertEqual(
                cfg.paths.resolve(cfg.paths.flare_dir),
                root.resolve() / "soft_proton_flare",
            )
            self.assertEqual(
                cfg.instruments.active_mos(),
                [("mos1", "mos1_evt.fits"), ("mos2", "mos2_evt.fits")],
            )
            self.assertEqual(cfg.instruments.pn, "pn_evt.fits")
            self.assertEqual(cfg.instruments.pn_oot, "pn_oot_evt.fits")
            self.assertEqual(cfg.espfilt.rangescale_pn, 15.0)
            self.assertEqual(cfg.source_detect.elowlist, [350, 2500])
            self.assertEqual(cfg.imaging.bands, [[400, 1250], [2000, 7000]])

    def test_spectrum_regions_toml_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "esas.toml"
            toml_path.write_text(
                render_template_toml(workdir=".")
                + '\n[[spectrum.regions]]\nname = "east"\nfile = "east.reg"\nmode = "split"\noverwrite = true\n',
                encoding="utf-8",
            )
            cfg = ESASConfig.from_toml(toml_path)
            self.assertEqual(len(cfg.spectrum.regions), 1)
            self.assertEqual(cfg.spectrum.regions[0].name, "east")
            self.assertEqual(cfg.spectrum.regions[0].mode, "split")
            self.assertTrue(cfg.spectrum.regions[0].overwrite)


class TestESASInit(unittest.TestCase):
    def test_standardize_event_files_picks_largest_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain_dir = root / "_chain_work"
            work_dir = root / "work"
            chain_dir.mkdir()
            work_dir.mkdir()

            # Two MOS1 exposures: U002 is larger than S001
            (chain_dir / "P0204870101M1S001MIEVLI0000.FIT").write_bytes(b"small")
            (chain_dir / "P0204870101M1U002MIEVLI0000.FIT").write_bytes(b"much_larger_evt")
            (chain_dir / "P0204870101M2S002MIEVLI0000.FIT").write_bytes(b"m2_evt")
            (chain_dir / "P0204870101PNS003PIEVLI0000.FIT").write_bytes(b"pn_evt")
            (chain_dir / "P0204870101PNS003OOEVLI0000.FIT").write_bytes(b"pn_oot")

            res = standardize_event_files(chain_dir, work_dir)
            self.assertEqual(set(res.keys()), {"mos1", "mos2", "pn", "pn_oot"})
            self.assertEqual(
                res["mos1"].resolve().name, "P0204870101M1U002MIEVLI0000.FIT"
            )
            self.assertTrue((work_dir / "mos1_evt.fits").is_symlink())
            self.assertTrue((work_dir / "pn_oot_evt.fits").is_symlink())

    def test_initialize_observation_dry_run_and_mocked_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odf_dir = root / "ODF"
            work_dir = root / "esas_work"
            odf_dir.mkdir()

            dry_res = initialize_observation(odf_dir, work_dir, dry_run=True)
            self.assertEqual(dry_res.workdir, work_dir.resolve())
            self.assertFalse((work_dir / "esas.toml").exists())

            seen_programs: list[str] = []

            def fake_run(self, cmd, *, force=False):
                seen_programs.append(cmd.program)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "cifbuild":
                    (cmd.cwd_path / "ccf.cif").write_text("ccf", encoding="utf-8")
                elif cmd.program == "odfingest":
                    (cmd.cwd_path / "1234_0204870101_SCX00000SUM.SAS").write_text(
                        "sum", encoding="utf-8"
                    )
                elif cmd.program == "emchain":
                    (cmd.cwd_path / "P0204870101M1S001MIEVLI0000.FIT").write_text(
                        "m1", encoding="utf-8"
                    )
                    (cmd.cwd_path / "P0204870101M2S002MIEVLI0000.FIT").write_text(
                        "m2", encoding="utf-8"
                    )
                elif cmd.program == "epchain":
                    if cmd.params.get("withoutoftime"):
                        (cmd.cwd_path / "P0204870101PNS003OOEVLI0000.FIT").write_text(
                            "oot", encoding="utf-8"
                        )
                    else:
                        (cmd.cwd_path / "P0204870101PNS003PIEVLI0000.FIT").write_text(
                            "pn", encoding="utf-8"
                        )
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.sas.runner.SASRunner.run", new=fake_run):
                res = initialize_observation(odf_dir, work_dir)

            self.assertEqual(
                seen_programs,
                ["cifbuild", "odfingest", "emchain", "epchain", "epchain"],
            )
            self.assertTrue(res.config_path.exists())
            self.assertEqual(set(res.event_files.keys()), {"mos1", "mos2", "pn", "pn_oot"})

            loaded = ESASConfig.from_toml(res.config_path)
            self.assertEqual(
                str(loaded.env.sas_ccf), str((work_dir / "ccf.cif").resolve())
            )
            self.assertEqual(
                str(loaded.env.sas_odf),
                str((work_dir / "1234_0204870101_SCX00000SUM.SAS").resolve()),
            )
            self.assertEqual(loaded.instruments.mos1, "mos1_evt.fits")


if __name__ == "__main__":
    unittest.main()
