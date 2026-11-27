"""Tests for ESAS TOML configuration and observation initialization."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    SASCommandError,
    normalize_lexical_path,
)
from xmm_tools.exec.runner import RunResult
from xmm_tools.pipeline import AnaConfig, render_template_toml
from xmm_tools.sas import (
    ODFInitResult,
    discover_odf_summary,
    initialize_observation,
    select_chain_exposures,
    standardize_event_files,
)
from xmm_tools.sas.odf import (
    _compute_ccfpath_fingerprint,
    _compute_odf_fingerprint,
)


class TestESASConfig(unittest.TestCase):
    def test_template_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "ana.toml"
            toml_path.write_text(
                render_template_toml(
                    workdir=".",
                    sas_ccf="/data/ccf.cif",
                    sas_ccfpath="/ccf/valid",
                    sas_odf="/data/SUM.SAS",
                ),
                encoding="utf-8",
            )

            cfg = AnaConfig.from_toml(toml_path)
            self.assertEqual(str(cfg.env.sas_ccf), "/data/ccf.cif")
            self.assertEqual(cfg.paths.workdir, normalize_lexical_path(root))
            self.assertEqual(
                cfg.paths.resolve(cfg.paths.flare_dir),
                normalize_lexical_path(root / "soft_proton_flare"),
            )
            self.assertEqual(
                cfg.instruments.active_mos(),
                [("mos1", "mos1_evt.fits"), ("mos2", "mos2_evt.fits")],
            )
            self.assertEqual(cfg.instruments.pn, "pn_evt.fits")
            self.assertEqual(cfg.instruments.pn_oot, "pn_oot_evt.fits")
            self.assertEqual(cfg.espfilt.rangescale_pn, 15.0)
            self.assertEqual(cfg.cheese.elowlist, [350, 2500])
            self.assertIn("soft", cfg.qpb_image.bands)

    def test_spectrum_regions_toml_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toml_path = root / "ana.toml"
            toml_path.write_text(
                render_template_toml(workdir=".")
                + '\n[[esas.spectrum.regions]]\nname = "east"\nfile = "east.reg"\nmode = "split"\noverwrite = true\n',
                encoding="utf-8",
            )
            cfg = AnaConfig.from_toml(toml_path)
            self.assertEqual(len(cfg.spectrum.regions), 1)
            self.assertEqual(cfg.spectrum.regions[0].name, "east")
            self.assertEqual(cfg.spectrum.regions[0].mode, "split")
            self.assertTrue(cfg.spectrum.regions[0].overwrite)


class TestESASInit(unittest.TestCase):
    def test_exposure_selection_policies_and_relative_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain_dir = root / "work" / "_chain_work"
            work_dir = root / "work"
            chain_dir.mkdir(parents=True)

            # Unique scheduled S001 must beat larger unscheduled U002
            (chain_dir / "P0204870101M1S001MIEVLI0000.FIT").write_bytes(b"small")
            (chain_dir / "P0204870101M1U002MIEVLI0000.FIT").write_bytes(b"much_larger_evt")
            (chain_dir / "P0204870101M2S002MIEVLI0000.FIT").write_bytes(b"m2_evt")
            (chain_dir / "P0204870101PNS003PIEVLI0000.FIT").write_bytes(b"pn_evt")
            (chain_dir / "P0204870101PNS003OOEVLI0000.FIT").write_bytes(b"pn_oot")

            selections = select_chain_exposures(chain_dir)
            self.assertEqual(selections["mos1"].policy, "unique_scheduled")
            self.assertEqual(
                selections["mos1"].selected.name, "P0204870101M1S001MIEVLI0000.FIT"
            )
            self.assertEqual(selections["mos2"].policy, "single")

            res = standardize_event_files(chain_dir, work_dir)
            self.assertEqual(set(res.keys()), {"mos1", "mos2", "pn", "pn_oot"})
            self.assertEqual(
                res["mos1"].resolve().name, "P0204870101M1S001MIEVLI0000.FIT"
            )
            self.assertTrue((work_dir / "mos1_evt.fits").is_symlink())
            self.assertTrue((work_dir / "pn_oot_evt.fits").is_symlink())
            link_target = os.readlink(work_dir / "mos1_evt.fits")
            self.assertFalse(Path(link_target).is_absolute())
            self.assertEqual(link_target, "_chain_work/P0204870101M1S001MIEVLI0000.FIT")

            # Explicit override selects requested file
            explicit_res = standardize_event_files(
                chain_dir,
                work_dir,
                exposure_selection={"mos1": "P0204870101M1U002MIEVLI0000.FIT"},
            )
            self.assertEqual(
                explicit_res["mos1"].resolve().name, "P0204870101M1U002MIEVLI0000.FIT"
            )

            # Invalid exposure_selection key raises ValueError
            with self.assertRaises(ValueError):
                select_chain_exposures(chain_dir, exposure_selection={"bad_inst": "x"})

            # Ambiguous multiple scheduled exposures raise ValueError unless explicit
            (chain_dir / "P0204870101M1S004MIEVLI0000.FIT").write_bytes(b"second_sched")
            with self.assertRaises(ValueError) as ctx:
                select_chain_exposures(chain_dir)
            self.assertIn("exposure_selection", str(ctx.exception))

            explicit_sched = select_chain_exposures(
                chain_dir,
                exposure_selection={"mos1": "P0204870101M1S004MIEVLI0000.FIT"},
            )
            self.assertEqual(explicit_sched["mos1"].policy, "explicit")
            self.assertEqual(
                explicit_sched["mos1"].selected.name, "P0204870101M1S004MIEVLI0000.FIT"
            )

    def test_discover_odf_summary_uniqueness_and_obsid_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                discover_odf_summary(root)

            sum_file = root / "1234_0204870101_SCX00000SUM.SAS"
            sum_file.write_text("OBSERVATION 0204870101", encoding="utf-8")
            self.assertEqual(
                discover_odf_summary(root, odf_dir=root / "0204870101"),
                sum_file,
            )

            with self.assertRaises(ValueError):
                discover_odf_summary(root, odf_dir=root / "0999999999")

            (root / "9999_0204870101_SCX00000SUM.SAS").write_text("dup", encoding="utf-8")
            with self.assertRaises(ValueError):
                discover_odf_summary(root)

    def test_initialize_observation_dry_run_reuse_and_config_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odf_dir = root / "0204870101"
            work_dir = root / "esas_work"
            odf_dir.mkdir()

            # dry_run creates zero directories/files on disk
            dry_res = initialize_observation(odf_dir, work_dir, dry_run=True)
            self.assertEqual(dry_res.workdir, work_dir.resolve())
            self.assertFalse(work_dir.exists())
            self.assertFalse((work_dir / "ana.toml").exists())

            seen_cmds: list[tuple[str, dict]] = []

            def fake_run(self, cmd, *, force=False):
                seen_cmds.append((cmd.program, dict(cmd.params)))
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

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res = initialize_observation(odf_dir, work_dir)

            self.assertEqual(
                [prog for prog, _ in seen_cmds],
                ["cifbuild", "odfingest", "emchain", "epchain", "epchain"],
            )
            # odfingest must not leak host odf_dir path in params
            ingest_params = seen_cmds[1][1]
            self.assertEqual(ingest_params, {"outdir": "."})
            self.assertTrue(res.config_path.exists())
            self.assertTrue(res.manifest_path.exists())
            self.assertEqual(set(res.event_files.keys()), {"mos1", "mos2", "pn", "pn_oot"})

            manifest_raw = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_raw["schema_version"], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest_raw["stage"], "init")
            self.assertEqual(manifest_raw["status"], "completed")
            self.assertEqual(
                set(manifest_raw["steps"].keys()),
                {"cifbuild", "odfingest", "emchain", "epchain:pn", "epchain:pn_oot"},
            )
            reloaded_init = ODFInitResult.from_manifest(res.manifest_path)
            self.assertEqual(reloaded_init.ccf_path, res.ccf_path.resolve())
            self.assertEqual(reloaded_init.odf_sum_path, res.odf_sum_path.resolve())
            self.assertEqual(
                reloaded_init.event_files["mos1"].resolve(),
                res.event_files["mos1"].resolve(),
            )

            loaded = AnaConfig.from_toml(res.config_path)
            self.assertEqual(
                str(loaded.env.sas_ccf), str((work_dir / "ccf.cif").resolve())
            )
            self.assertEqual(
                str(loaded.env.sas_odf),
                str((work_dir / "1234_0204870101_SCX00000SUM.SAS").resolve()),
            )
            self.assertEqual(loaded.instruments.mos1, "mos1_evt.fits")

            # Re-running without force reuses all 5 steps (0 subprocess calls)
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                reused_res = initialize_observation(odf_dir, work_dir)
            self.assertEqual(seen_cmds, [])
            self.assertIsNotNone(reused_res.manifest)
            self.assertEqual(reused_res.manifest.status, "reused")

            # Editing ana.toml and re-running with force=True backs up to ana.toml.bak
            edited_toml = res.config_path.read_text(encoding="utf-8") + "\n# user edit\n"
            res.config_path.write_text(edited_toml, encoding="utf-8")
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf_dir, work_dir, force=True)
            bak_path = res.config_path.with_suffix(".toml.bak")
            self.assertTrue(bak_path.is_file())
            self.assertIn("# user edit", bak_path.read_text(encoding="utf-8"))
            self.assertNotIn("# user edit", res.config_path.read_text(encoding="utf-8"))

    def test_initialize_observation_raises_when_no_scientific_events_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odf_dir = root / "ODF"
            work_dir = root / "esas_work"
            odf_dir.mkdir()

            def fake_run_empty_chains(self, cmd, *, force=False):
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "cifbuild":
                    (cmd.cwd_path / "ccf.cif").write_text("ccf", encoding="utf-8")
                elif cmd.program == "odfingest":
                    (cmd.cwd_path / "1234_0204870101_SCX00000SUM.SAS").write_text(
                        "sum", encoding="utf-8"
                    )
                return RunResult(command=cmd, status="completed", returncode=0)

            with (
                patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run_empty_chains),
                self.assertRaises(SASCommandError),
            ):
                initialize_observation(odf_dir, work_dir)

            manifest_failed = json.loads(
                (work_dir / "init_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_failed["status"], "failed")
            self.assertIn("no scientific", manifest_failed["error"])
            for sid in ("emchain", "epchain:pn", "epchain:pn_oot"):
                self.assertEqual(manifest_failed["steps"][sid]["outputs"], {})

            # Retrying after empty chains must NOT reuse empty chain steps
            retry_cmds: list[str] = []

            def fake_run_retry_with_events(self, cmd, *, force=False):
                retry_cmds.append(cmd.program)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "emchain":
                    (cmd.cwd_path / "P0204870101M1S001MIEVLI0000.FIT").write_text(
                        "m1", encoding="utf-8"
                    )
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch(
                "xmm_tools.exec.runner.SASRunner.run", new=fake_run_retry_with_events
            ):
                retry_res = initialize_observation(odf_dir, work_dir)

            self.assertEqual(retry_cmds, ["emchain", "epchain", "epchain"])
            self.assertEqual(retry_res.manifest.status, "completed")
            self.assertEqual(retry_res.manifest.steps["cifbuild"].status, "reused")
            self.assertEqual(retry_res.manifest.steps["odfingest"].status, "reused")
            self.assertIn("mos1", retry_res.event_files)

    def test_standardize_event_files_removes_stale_vanished_instruments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain_dir = root / "_chain_work"
            work_dir = root / "work"
            chain_dir.mkdir(parents=True)
            work_dir.mkdir(parents=True)

            (chain_dir / "P0204870101M1S001MIEVLI0000.FIT").write_bytes(b"m1")
            (chain_dir / "P0204870101PNS003PIEVLI0000.FIT").write_bytes(b"pn")
            standardize_event_files(chain_dir, work_dir)
            self.assertTrue((work_dir / "pn_evt.fits").is_symlink())

            # Remove PN event from chain_dir and re-run standardization
            (chain_dir / "P0204870101PNS003PIEVLI0000.FIT").unlink()
            res = standardize_event_files(chain_dir, work_dir)
            self.assertEqual(set(res.keys()), {"mos1"})
            self.assertFalse((work_dir / "pn_evt.fits").is_symlink())
            self.assertFalse((work_dir / "pn_evt.fits").exists())

    def test_initialize_observation_invalidates_on_odf_or_ccfpath_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odf_dir = root / "0204870101"
            work_dir = root / "esas_work"
            odf_dir.mkdir()
            (odf_dir / "0001_0204870101_SCX00000SUM.ASC").write_text(
                "odf_v1", encoding="utf-8"
            )

            seen_cmds: list[str] = []
            run_tag = {"val": "v1"}

            def fake_run(self, cmd, *, force=False):
                seen_cmds.append(cmd.program)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                if cmd.program == "cifbuild":
                    (cmd.cwd_path / "ccf.cif").write_text(
                        f"ccf_{run_tag['val']}", encoding="utf-8"
                    )
                elif cmd.program == "odfingest":
                    (cmd.cwd_path / "1234_0204870101_SCX00000SUM.SAS").write_text(
                        f"sum_{run_tag['val']}", encoding="utf-8"
                    )
                elif cmd.program == "emchain":
                    (cmd.cwd_path / "P0204870101M1S001MIEVLI0000.FIT").write_text(
                        f"m1_{run_tag['val']}", encoding="utf-8"
                    )
                return RunResult(command=cmd, status="completed", returncode=0)

            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf_dir, work_dir, sas_ccfpath="/ccf/valid_v1")
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            # Same ODF and same ccfpath reuses everything
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                res_reused = initialize_observation(
                    odf_dir, work_dir, sas_ccfpath="/ccf/valid_v1"
                )
            self.assertEqual(seen_cmds, [])
            self.assertEqual(res_reused.manifest.status, "reused")

            # Changing *SUM.ASC content invalidates cifbuild & odfingest (and cascades)
            (odf_dir / "0001_0204870101_SCX00000SUM.ASC").write_text(
                "odf_v2", encoding="utf-8"
            )
            run_tag["val"] = "v2"
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf_dir, work_dir, sas_ccfpath="/ccf/valid_v1")
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            # Changing sas_ccfpath directory name invalidates cifbuild & odfingest
            run_tag["val"] = "v3"
            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf_dir, work_dir, sas_ccfpath="/ccf/valid_v2")
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

    def test_same_basename_ccfpath_and_summaryless_odf_invalidate_and_stay_mount_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "esas_work"

            odf1 = root / "obs1" / "ODF"
            odf2 = root / "obs2" / "ODF"
            odf1.mkdir(parents=True)
            odf2.mkdir(parents=True)

            ccf_a = root / "ccf_a" / "valid"
            ccf_b = root / "ccf_b" / "valid"
            ccf_a.mkdir(parents=True)
            ccf_b.mkdir(parents=True)

            seen_cmds: list[str] = []
            run_counter = {"n": 0}

            def fake_run(self, cmd, *, force=False):
                seen_cmds.append(cmd.program)
                cmd.cwd_path.mkdir(parents=True, exist_ok=True)
                tag = run_counter["n"]
                if cmd.program == "cifbuild":
                    (cmd.cwd_path / "ccf.cif").write_text(f"ccf_{tag}", encoding="utf-8")
                elif cmd.program == "odfingest":
                    (cmd.cwd_path / "1234_0204870101_SCX00000SUM.SAS").write_text(
                        f"sum_{tag}", encoding="utf-8"
                    )
                elif cmd.program == "emchain":
                    (cmd.cwd_path / "P0204870101M1S001MIEVLI0000.FIT").write_text(
                        f"m1_{tag}", encoding="utf-8"
                    )
                return RunResult(command=cmd, status="completed", returncode=0)

            # 1. Empty directories with identical basename ("ODF" and "valid")
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf1, work_dir, sas_ccfpath=ccf_a)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf1, work_dir, sas_ccfpath=ccf_b)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf2, work_dir, sas_ccfpath=ccf_b)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            # 2. Populated directories with identical basenames and different files
            (odf1 / "0001_0001_OBX.FIT").write_bytes(b"odf_one")
            (odf2 / "0001_0001_OBX.FIT").write_bytes(b"odf_two")
            (ccf_a / "EMOS1_LINCOORD_0019.CCF").write_bytes(b"ccf_alpha")
            (ccf_b / "EMOS1_LINCOORD_0019.CCF").write_bytes(b"ccf_beta")
            # Large file (> 64 KiB) differing only near tail
            (ccf_a / "LARGE_CAL.CCF").write_bytes(b"A" * 70000 + b"TAIL_1")
            (ccf_b / "LARGE_CAL.CCF").write_bytes(b"A" * 70000 + b"TAIL_2")

            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf1, work_dir, sas_ccfpath=ccf_a)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf1, work_dir, sas_ccfpath=ccf_b)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf2, work_dir, sas_ccfpath=ccf_b)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            # 3. Host <-> Docker portability: same calibration files under mounted path
            host_fp = _compute_ccfpath_fingerprint(ccf_a, workspace_root=work_dir)
            container_fp = _compute_ccfpath_fingerprint(
                "/container_mount/ccf/valid",
                workspace_root=work_dir,
                mounts=[(str(root / "ccf_a"), "/container_mount/ccf")],
            )
            self.assertEqual(host_fp, container_fp)
            self.assertTrue(host_fp.startswith("ccf:"))

            odf_host_fp = _compute_odf_fingerprint(odf1, workspace_root=work_dir)
            odf_container_fp = _compute_odf_fingerprint(
                Path("/container_odf/ODF"),
                workspace_root=work_dir,
                mounts=[(str(odf1), "/container_odf/ODF")],
            )
            self.assertEqual(odf_host_fp, odf_container_fp)
            self.assertTrue(odf_host_fp.startswith("files:"))

            # 4. Large file (> 64 KiB) differing ONLY in the middle (identical head, tail, name, size)
            mid_head = b"H" * 35000
            mid_tail = b"T" * (80_000 - 35008)
            mid_payload_1 = mid_head + b"MIDDLE_1" + mid_tail
            mid_payload_2 = mid_head + b"MIDDLE_2" + mid_tail
            self.assertEqual(len(mid_payload_1), 80_000)
            self.assertEqual(len(mid_payload_2), 80_000)
            self.assertEqual(mid_payload_1[:16384], mid_payload_2[:16384])
            self.assertEqual(mid_payload_1[-16384:], mid_payload_2[-16384:])

            ccf_mid = root / "ccf_mid" / "valid"
            ccf_mid.mkdir(parents=True)
            (ccf_mid / "MID_CAL.CCF").write_bytes(mid_payload_1)

            fp_mid_1 = _compute_ccfpath_fingerprint(ccf_mid, workspace_root=work_dir)
            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf2, work_dir, sas_ccfpath=ccf_mid)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )

            seen_cmds.clear()
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf2, work_dir, sas_ccfpath=ccf_mid)
            self.assertEqual(seen_cmds, [])

            (ccf_mid / "MID_CAL.CCF").write_bytes(mid_payload_2)
            fp_mid_2 = _compute_ccfpath_fingerprint(ccf_mid, workspace_root=work_dir)
            self.assertNotEqual(fp_mid_1, fp_mid_2)

            seen_cmds.clear()
            run_counter["n"] += 1
            with patch("xmm_tools.exec.runner.SASRunner.run", new=fake_run):
                initialize_observation(odf2, work_dir, sas_ccfpath=ccf_mid)
            self.assertEqual(
                seen_cmds, ["cifbuild", "odfingest", "emchain", "epchain", "epchain"]
            )


if __name__ == "__main__":
    unittest.main()
