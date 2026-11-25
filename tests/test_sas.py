"""Tests for generic SASEnv and SASRunner command wrapper."""

from pathlib import Path
import sys
import tempfile
import unittest

from xmm_tools.sas import SASCommand, SASCommandError, SASEnv, SASRunner


class TestSASEnv(unittest.TestCase):
    def test_from_toml_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ccf = root / "ccf.cif"
            odf = root / "SUM.SAS"
            ccf.write_text("ccf", encoding="utf-8")
            odf.write_text("odf", encoding="utf-8")

            cfg = root / "esas.toml"
            cfg.write_text(
                f"""\
[env]
sas_ccf = "{ccf}"
sas_ccfpath = "/ccf/valid"
sas_odf = "{odf}"
CUSTOM_FLAG = "1"
""",
                encoding="utf-8",
            )

            env = SASEnv.from_toml(cfg)
            env.validate(check_paths=True)
            merged = env.to_env_dict(base_env={"PATH": "/usr/bin"})
            self.assertEqual(merged["SAS_CCF"], str(ccf))
            self.assertEqual(merged["SAS_CCFPATH"], "/ccf/valid")
            self.assertEqual(merged["SAS_ODF"], str(odf))
            self.assertEqual(merged["CUSTOM_FLAG"], "1")

            exports = "\n".join(env.shell_exports())
            self.assertIn("export SAS_CCF=", exports)
            self.assertIn("export CUSTOM_FLAG=1", exports)

    def test_from_toml_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ccf.cif").write_text("ccf", encoding="utf-8")
            (root / "SUM.SAS").write_text("odf", encoding="utf-8")
            cfg = root / "esas.toml"
            cfg.write_text(
                """\
[env]
sas_ccf = "ccf.cif"
sas_ccfpath = "/ccf/valid"
sas_odf = "SUM.SAS"
regions_path = ""
""",
                encoding="utf-8",
            )

            env = SASEnv.from_toml(cfg)
            self.assertEqual(env.sas_ccf, str((root / "ccf.cif").resolve()))
            self.assertEqual(env.sas_odf, str((root / "SUM.SAS").resolve()))
            self.assertEqual(env.sas_ccfpath, "/ccf/valid")
            self.assertEqual(env.regions_path, "")
            env.validate(check_paths=True)

    def test_validate_missing_raises(self):
        env = SASEnv()
        with self.assertRaises(ValueError):
            env.validate()


class TestSASCommandAndRunner(unittest.TestCase):
    def test_command_argv_and_shell_rendering(self):
        env = SASEnv(sas_ccf="/tmp/ccf.cif", sas_odf="/tmp/SUM.SAS")
        cmd = SASCommand(
            program="cheese",
            cwd="/tmp/cheese",
            params={
                "elowlist": [350, 2500],
                "keepinterfiles": False,
                "omit": None,
                "scale": 0.4,
            },
            symlinks={"mos1S001-allevc.fits": "../soft_proton_flare/mos1S001-allevc.fits"},
        )
        self.assertEqual(
            cmd.argv(),
            ["cheese", "elowlist=350 2500", "keepinterfiles=no", "scale=0.4"],
        )
        shell = cmd.to_shell(env)
        self.assertIn("export SAS_CCF=/tmp/ccf.cif", shell)
        self.assertIn("ln -sfn ../soft_proton_flare/mos1S001-allevc.fits mos1S001-allevc.fits", shell)
        self.assertIn("'elowlist=350 2500'", shell)

    def test_runner_executes_isolates_intermediate_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "src"
            work_dir = root / "work"
            src_dir.mkdir()
            input_file = src_dir / "input.fits"
            input_file.write_text("fits", encoding="utf-8")

            # Use python3 as a stand-in SAS executable that ignores args and writes outputs
            script = (
                "from pathlib import Path; "
                "Path('core.fits').write_text('core_data'); "
                "Path('junk1.tmp').write_text('temp1'); "
                "Path('junk2.tmp').write_text('temp2'); "
                "print('SAS_CCF=' + __import__('os').environ.get('SAS_CCF', ''))"
            )
            cmd = SASCommand(
                program=sys.executable,
                cwd=work_dir,
                params={"-c": None},  # overridden below via subclass/helper test
                core_outputs=["core.fits"],
                symlinks={"linked.fits": input_file},
            )
            # Test dry_run first
            env = SASEnv(sas_ccf="/data/ccf.cif", sas_odf="/data/SUM.SAS")
            dry_runner = SASRunner(env=env, dry_run=True)
            dry_res = dry_runner.run(cmd)
            self.assertEqual(dry_res.status, "dry_run")
            self.assertFalse(work_dir.exists())

            # Run real subprocess with python -c <script>
            class _InlineCmd(SASCommand):
                def argv(self) -> list[str]:
                    return [sys.executable, "-c", script]

            real_cmd = _InlineCmd(
                program="mock_sas",
                cwd=work_dir,
                core_outputs=["core.fits"],
                symlinks={"linked.fits": input_file},
                intermediate_dir="_intermediate",
            )
            runner = SASRunner(env=env, skip_if_exists=True)
            res = runner.run(real_cmd)
            self.assertEqual(res.status, "completed")
            self.assertTrue((work_dir / "core.fits").exists())
            self.assertTrue((work_dir / "linked.fits").is_symlink())
            self.assertFalse((work_dir / "junk1.tmp").exists())
            self.assertTrue((work_dir / "_intermediate" / "junk1.tmp").exists())
            self.assertTrue((work_dir / "_intermediate" / "junk2.tmp").exists())
            self.assertIn("SAS_CCF=/data/ccf.cif", (work_dir / "execution.log").read_text())

            # Re-running with skip_if_exists=True should skip
            skipped = runner.run(real_cmd)
            self.assertEqual(skipped.status, "skipped")

            # force=True re-runs
            forced = runner.run(real_cmd, force=True)
            self.assertEqual(forced.status, "completed")

    def test_runner_raises_on_missing_core_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            class _EmptyCmd(SASCommand):
                def argv(self) -> list[str]:
                    return [sys.executable, "-c", "print('ok')"]

            runner = SASRunner(env=SASEnv())
            with self.assertRaises(SASCommandError):
                runner.run(
                    _EmptyCmd(
                        program="mock_sas",
                        cwd=Path(tmp),
                        core_outputs=["missing.fits"],
                    )
                )

    def test_runner_isolates_heasoft_pfiles_per_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            script = "import os; print('PFILES=' + os.environ.get('PFILES', ''))"

            class _PrintPfilesCmd(SASCommand):
                def argv(self) -> list[str]:
                    return [sys.executable, "-c", script]

            env = SASEnv(
                extra_env={"PFILES": "/home/xmmsas/pfiles;/usr/local/heasoft/syspfiles"}
            )
            runner = SASRunner(env=env, skip_if_exists=False)
            res = runner.run(
                _PrintPfilesCmd(
                    program="mock_sas",
                    cwd=work_dir,
                    core_outputs=[],
                    intermediate_dir=None,
                )
            )
            log_text = res.log_path.read_text(encoding="utf-8").strip()
            self.assertTrue(log_text.startswith("PFILES="))
            pfiles_val = log_text.split("=", 1)[1]
            user_dir, sys_dir = pfiles_val.split(";", 1)
            self.assertEqual(sys_dir, "/usr/local/heasoft/syspfiles")
            self.assertNotEqual(user_dir, "/home/xmmsas/pfiles")
            self.assertIn("xmm_pfiles_", user_dir)
            # Temporary pfiles directory is cleaned up after subprocess exits
            self.assertFalse(Path(user_dir).exists())

    def test_docker_exec_argv(self):
        env = SASEnv(sas_ccf="/ccf.cif")
        runner = SASRunner(env=env, docker_container="xmmsas_box")
        cmd = SASCommand(program="espfilt", cwd="/work/flare", params={"elow": 2500})
        exec_argv = runner._build_exec_argv(cmd)
        self.assertEqual(exec_argv[:5], ["docker", "exec", "-w", "/work/flare", "-e"])
        self.assertIn("xmmsas_box", exec_argv)
        self.assertEqual(exec_argv[-2:], ["espfilt", "elow=2500"])


if __name__ == "__main__":
    unittest.main()
