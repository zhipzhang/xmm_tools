"""Unit tests for the Layer 0 execution, workspace, and manifest foundation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    FileIdentity,
    SASCommand,
    SASCommandError,
    SASEnv,
    SASRunner,
    StageManifest,
    StageWorkspace,
    StepLedger,
    atomic_write_json,
    check_log_markers,
    compute_sha256,
    create_relative_symlink,
    find_unique_file,
    normalize_lexical_path,
    publish_artifact,
    read_json,
    read_log_tail,
    resolve_workspace_path,
    workspace_relative_path,
)


def _inline_python_cmd(
    *,
    program: str,
    cwd: Path,
    script: str,
    params: dict | None = None,
    symlinks: dict | None = None,
    expected_outputs: list[str] | None = None,
    log_file: str = "execution.log",
) -> SASCommand:
    class _InlineCmd(SASCommand):
        def argv(self) -> list[str]:
            return [sys.executable, "-c", script]

    return _InlineCmd(
        program=program,
        cwd=cwd,
        params=params or {},
        symlinks=symlinks or {},
        expected_outputs=expected_outputs or [],
        log_file=log_file,
    )


class TestPathsAndWorkspace(unittest.TestCase):
    def test_lexical_normalization_preserves_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_dir = root / "real"
            real_dir.mkdir()
            link_dir = root / "alias"
            link_dir.symlink_to(real_dir)

            norm = normalize_lexical_path(link_dir / "sub" / ".." / "evt.fits")
            self.assertEqual(norm.parent.name, "alias")

    def test_workspace_relative_path_and_resolve(self):
        root = Path("/Users/alice/obs1")
        stage_dir = root / "qpb_image"
        step_dir = stage_dir / "_work" / "mos1"
        sibling_file = root / "soft_proton_flare" / "mos1S001-gti.fits"
        ccf_file = Path("/ccf/valid/MOS1.CCF")

        self.assertEqual(
            workspace_relative_path(step_dir / "mos1.log", stage_dir, workspace_root=root),
            "_work/mos1/mos1.log",
        )
        self.assertEqual(
            workspace_relative_path(stage_dir, stage_dir, workspace_root=root),
            ".",
        )
        self.assertEqual(
            workspace_relative_path(sibling_file, stage_dir, workspace_root=root),
            "../soft_proton_flare/mos1S001-gti.fits",
        )
        self.assertEqual(
            workspace_relative_path(ccf_file, stage_dir, workspace_root=root),
            "/ccf/valid/MOS1.CCF",
        )
        self.assertEqual(
            resolve_workspace_path(stage_dir, "../soft_proton_flare/mos1S001-gti.fits"),
            normalize_lexical_path(sibling_file),
        )

    def test_relative_symlink_and_publish_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_a = root / "work" / "a.fits"
            src_b = root / "work" / "b.fits"
            src_a.parent.mkdir(parents=True)
            src_a.write_bytes(b"A" * 2880)
            src_b.write_bytes(b"B" * 2880)

            link = root / "stage" / "product.fits"
            create_relative_symlink(src_a, link)
            self.assertTrue(link.is_symlink())
            self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(link.read_bytes(), b"A" * 2880)

            # Symlinks can be atomically replaced
            create_relative_symlink(src_b, link)
            self.assertEqual(link.read_bytes(), b"B" * 2880)

            # Regular files are protected unless overwrite_file=True
            regular = root / "stage" / "existing.fits"
            regular.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                create_relative_symlink(src_a, regular)

            # publish_artifact supports atomic symlink and copy replacement
            published = publish_artifact(src_a, regular, mode="copy")
            self.assertFalse(published.is_symlink())
            self.assertEqual(published.read_bytes(), b"A" * 2880)

            published_sym = publish_artifact(src_b, regular, mode="symlink")
            self.assertTrue(published_sym.is_symlink())
            self.assertEqual(published_sym.read_bytes(), b"B" * 2880)

            empty = root / "work" / "empty.fits"
            empty.write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                publish_artifact(empty, root / "stage" / "empty.fits")

    def test_find_unique_file_and_log_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                find_unique_file(root, "*.fits", description="event file")

            (root / "one.fits").write_bytes(b"data")
            self.assertEqual(
                find_unique_file(root, "*.fits", description="event file"),
                normalize_lexical_path(root / "one.fits"),
            )

            (root / "two.fits").write_bytes(b"data2")
            with self.assertRaises(ValueError):
                find_unique_file(root, "*.fits", description="event file")

            log = root / "task.log"
            log.write_text("line1\n** error: fatal crash\nline3\n", encoding="utf-8")
            self.assertEqual(
                check_log_markers(log, ["** error:", "FATAL"]), "** error:"
            )
            self.assertIsNone(check_log_markers(log, ["MISSING_MARKER"]))
            self.assertEqual(read_log_tail(log, limit=6), "line3\n")

    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nested" / "manifest.json"
            atomic_write_json(p, {"schema_version": 2, "ok": True})
            self.assertEqual(read_json(p), {"schema_version": 2, "ok": True})


class TestPortableSpecAndRunner(unittest.TestCase):
    def test_host_vs_docker_command_to_spec_is_identical_and_portable(self):
        host_root = Path("/Users/alice/projects/obs1")
        docker_root = Path("/home/xmmsas/work/obs1")

        def make_cmd(root: Path) -> SASCommand:
            stage_dir = root / "cheese"
            work_dir = stage_dir / "_work"
            return SASCommand(
                program="cheese",
                cwd=work_dir,
                params={
                    "scale": 0.4,
                    "elowlist": [350, 2500],
                    "gtifile": root / "soft_proton_flare" / "mos1S001-gti.fits",
                    "omit": None,
                },
                symlinks={
                    "mos1S001-allevc.fits": root
                    / "soft_proton_flare"
                    / "mos1S001-allevc.fits"
                },
                expected_outputs=["mos1S001-cheeset.fits"],
                log_file="cheese.log",
            )

        host_spec = make_cmd(host_root).to_spec(
            base_dir=host_root / "cheese", workspace_root=host_root
        )
        docker_spec = make_cmd(docker_root).to_spec(
            base_dir=docker_root / "cheese", workspace_root=docker_root
        )

        self.assertEqual(host_spec, docker_spec)
        serialized = json.dumps(host_spec)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/home/xmmsas/", serialized)
        self.assertEqual(host_spec["cwd"], "_work")
        self.assertEqual(
            host_spec["symlinks"]["mos1S001-allevc.fits"],
            "../../soft_proton_flare/mos1S001-allevc.fits",
        )
        self.assertEqual(
            host_spec["params"]["gtifile"],
            "../../soft_proton_flare/mos1S001-gti.fits",
        )

    def test_docker_pfiles_isolation_wrapper(self):
        env = SASEnv(
            sas_ccf="/home/xmmsas/obs/ccf.cif",
            extra_env={"PFILES": "/home/xmmsas/pfiles;/usr/local/heasoft/syspfiles"},
        )
        runner = SASRunner(env=env, docker_container="xmmsas_box")
        cmd = SASCommand(program="espfilt", cwd="/home/xmmsas/obs/flare", params={"elow": 2500})
        exec_argv = runner._build_exec_argv(cmd)

        self.assertEqual(exec_argv[:5], ["docker", "exec", "-w", "/home/xmmsas/obs/flare", "-e"])
        sh_idx = exec_argv.index("bash")
        self.assertEqual(exec_argv[sh_idx + 1], "-lc")
        wrapper_script = exec_argv[sh_idx + 2]
        self.assertIn("mktemp -d /tmp/xmm_pfiles.XXXXXX", wrapper_script)
        self.assertIn('PFILES="$tmp_pf;${PFILES#*;}"', wrapper_script)
        self.assertEqual(exec_argv[sh_idx + 3 : sh_idx + 6], ["--", "espfilt", "elow=2500"])

        # Execute the bash wrapper directly with python to verify PFILES rewrite
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                wrapper_script,
                "--",
                sys.executable,
                "-c",
                "import os; print(os.environ['PFILES'])",
            ],
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PFILES": "/home/xmmsas/pfiles;/usr/local/heasoft/syspfiles",
            },
            text=True,
            capture_output=True,
            check=True,
        )
        new_user_pf, sys_pf = proc.stdout.strip().split(";", 1)
        self.assertEqual(sys_pf, "/usr/local/heasoft/syspfiles")
        self.assertIn("xmm_pfiles.", new_user_pf)
        self.assertFalse(Path(new_user_pf).exists())


class TestStepLedgerAndManifest(unittest.TestCase):
    def _build_two_step_workspace(self, root: Path):
        ws = StageWorkspace(workspace_root=root, stage_dir=root / "stage_a")
        ws.ensure_dirs()
        evt = root / "mos1_evt.fits"
        evt.write_bytes(b"E" * 2880)

        step1_dir = ws.step_dir("step1")
        cmd1 = _inline_python_cmd(
            program="task_one",
            cwd=step1_dir,
            params={"elow": 2500},
            symlinks={"input.fits": evt},
            expected_outputs=["out1.fits"],
            script="from pathlib import Path; Path('out1.fits').write_bytes(b'O1' * 1440)",
            log_file="step1.log",
        )
        step2_dir = ws.step_dir("step2")
        cmd2 = _inline_python_cmd(
            program="task_two",
            cwd=step2_dir,
            params={"scale": 0.4},
            symlinks={"prev.fits": step1_dir / "out1.fits"},
            expected_outputs=["out2.fits"],
            script="from pathlib import Path; Path('out2.fits').write_bytes(b'O2' * 1440)",
            log_file="step2.log",
        )
        return ws, evt, cmd1, cmd2

    def test_workspace_move_preserves_reusability_and_zero_abs_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            obs_a = base / "Users" / "alice" / "obs_orig"
            obs_a.mkdir(parents=True)
            ws_a, evt_a, cmd1_a, cmd2_a = self._build_two_step_workspace(obs_a)

            runner = SASRunner(env=SASEnv())
            ledger_a = StepLedger(
                ws_a.manifest_path,
                stage="stage_a",
                workspace_root=obs_a,
            )
            rec1 = ledger_a.run_step("s1", runner, cmd1_a, inputs={"evt": evt_a})
            rec2 = ledger_a.run_step(
                "s2",
                runner,
                cmd2_a,
                inputs={"out1": ws_a.step_dir("step1") / "out1.fits"},
            )
            self.assertEqual(rec1.status, "completed")
            self.assertEqual(rec2.status, "completed")
            pub = ws_a.publish(ws_a.step_dir("step2") / "out2.fits", "final_out2.fits")
            manifest = ledger_a.commit({"final": pub})
            self.assertEqual(manifest.status, "completed")

            # Verify manifest JSON has zero absolute host/container paths in portable fields
            raw = read_json(ws_a.manifest_path)
            self.assertEqual(raw["schema_version"], MANIFEST_SCHEMA_VERSION)
            for step_raw in raw["steps"].values():
                step_check = json.dumps(
                    {
                        k: v
                        for k, v in step_raw.items()
                        if k != "shell"
                    }
                )
                self.assertNotIn(str(obs_a), step_check)
            self.assertEqual(raw["products"]["final"], "final_out2.fits")

            # Move the entire workspace directory to a completely new location
            obs_b = base / "home" / "xmmsas" / "obs_moved"
            obs_b.parent.mkdir(parents=True)
            shutil.move(str(obs_a), str(obs_b))

            ws_b, evt_b, cmd1_b, cmd2_b = self._build_two_step_workspace(obs_b)
            # Symlinks inside moved workspace must still resolve cleanly
            self.assertTrue((ws_b.stage_dir / "final_out2.fits").is_symlink())
            self.assertTrue((ws_b.stage_dir / "final_out2.fits").exists())

            loaded = StageManifest.from_dict(ws_b.stage_dir, read_json(ws_b.manifest_path))
            self.assertEqual(loaded.stage, "stage_a")

            ledger_b = StepLedger(
                ws_b.manifest_path,
                stage="stage_a",
                workspace_root=obs_b,
            )
            reused1 = ledger_b.can_reuse("s1", cmd1_b, inputs={"evt": evt_b})
            reused2 = ledger_b.can_reuse(
                "s2",
                cmd2_b,
                inputs={"out1": ws_b.step_dir("step1") / "out1.fits"},
            )
            self.assertIsNotNone(reused1)
            self.assertIsNotNone(reused2)
            self.assertEqual(reused1.status, "reused")
            self.assertEqual(reused2.status, "reused")

    def test_same_size_fits_replacement_invalidates_via_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws, evt, cmd1, _ = self._build_two_step_workspace(root)
            runner = SASRunner(env=SASEnv())

            ledger = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            ledger.run_step("s1", runner, cmd1, inputs={"evt": evt})
            ledger.commit({})

            # Replace evt with exact same 2880-byte size, different payload
            self.assertEqual(evt.stat().st_size, 2880)
            old_sha = compute_sha256(evt)
            evt.write_bytes(b"X" * 2880)
            self.assertEqual(evt.stat().st_size, 2880)
            self.assertNotEqual(compute_sha256(evt), old_sha)

            ledger2 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            self.assertIsNone(ledger2.can_reuse("s1", cmd1, inputs={"evt": evt}))

    def test_parameter_change_invalidates_only_affected_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws, evt, cmd1, cmd2 = self._build_two_step_workspace(root)
            runner = SASRunner(env=SASEnv())

            ledger = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            ledger.run_step("s1", runner, cmd1, inputs={"evt": evt})
            ledger.run_step(
                "s2", runner, cmd2, inputs={"out1": ws.step_dir("step1") / "out1.fits"}
            )
            ledger.commit({})

            # Change parameter on step2 only
            cmd2_changed = _inline_python_cmd(
                program="task_two",
                cwd=ws.step_dir("step2"),
                params={"scale": 0.6},
                symlinks={"prev.fits": ws.step_dir("step1") / "out1.fits"},
                expected_outputs=["out2.fits"],
                script="from pathlib import Path; Path('out2.fits').write_bytes(b'NEW2' * 720)",
                log_file="step2.log",
            )
            ledger2 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            r1 = ledger2.run_step("s1", runner, cmd1, inputs={"evt": evt})
            r2 = ledger2.run_step(
                "s2",
                runner,
                cmd2_changed,
                inputs={"out1": ws.step_dir("step1") / "out1.fits"},
            )
            m2 = ledger2.commit({})
            self.assertEqual(r1.status, "reused")
            self.assertEqual(r2.status, "completed")
            self.assertEqual(m2.status, "completed")

    def test_dry_run_leaves_workspace_untouched_and_reports_reused_vs_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws, evt, cmd1, cmd2 = self._build_two_step_workspace(root)
            runner = SASRunner(env=SASEnv())

            # Populate step1 only
            ledger1 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            ledger1.run_step("s1", runner, cmd1, inputs={"evt": evt})
            ledger1.commit({"s1": ws.step_dir("step1") / "out1.fits"})

            before_manifest = ws.manifest_path.read_bytes()
            before_files = sorted(
                p.relative_to(root).as_posix() for p in root.rglob("*")
            )

            dry_runner = SASRunner(env=SASEnv(), dry_run=True)
            dry_ledger = StepLedger(
                ws.manifest_path,
                stage="stage_a",
                workspace_root=root,
                dry_run=True,
            )
            r1 = dry_ledger.run_step("s1", dry_runner, cmd1, inputs={"evt": evt})
            r2 = dry_ledger.run_step(
                "s2",
                dry_runner,
                cmd2,
                inputs={"out1": ws.step_dir("step1") / "out1.fits"},
            )
            dry_manifest = dry_ledger.commit({})

            self.assertEqual(r1.status, "reused")
            self.assertEqual(r2.status, "dry_run")
            self.assertIsNone(r2.returncode)
            self.assertEqual(dry_manifest.status, "dry_run")
            self.assertEqual(ws.manifest_path.read_bytes(), before_manifest)
            after_files = sorted(
                p.relative_to(root).as_posix() for p in root.rglob("*")
            )
            self.assertEqual(after_files, before_files)

    def test_derived_step_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = StageWorkspace(workspace_root=root, stage_dir=root / "qpb")
            ws.ensure_dirs()
            in_file = root / "in.fits"
            in_file.write_bytes(b"IN1")
            out_file = ws.step_dir("combine") / "sum.fits"

            calls = 0

            def compute_sum():
                nonlocal calls
                calls += 1
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_bytes(in_file.read_bytes() + b"_SUM")
                return {"sum": out_file}

            l1 = StepLedger(ws.manifest_path, stage="qpb", workspace_root=root)
            r1 = l1.record_derived_step(
                "sum_band",
                description="sum_flux_image",
                params={"band": "soft"},
                inputs={"src": in_file},
                compute=compute_sum,
                algorithm_version="1",
            )
            m1 = l1.commit({"soft": out_file})
            self.assertEqual(r1.status, "completed")
            self.assertEqual(m1.status, "completed")
            self.assertEqual(calls, 1)

            # Exact replay reuses without calling compute()
            l2 = StepLedger(ws.manifest_path, stage="qpb", workspace_root=root)
            r2 = l2.record_derived_step(
                "sum_band",
                description="sum_flux_image",
                params={"band": "soft"},
                inputs={"src": in_file},
                compute=compute_sum,
                algorithm_version="1",
            )
            m2 = l2.commit({"soft": out_file})
            self.assertEqual(r2.status, "reused")
            self.assertEqual(m2.status, "reused")
            self.assertEqual(calls, 1)

            # Algorithm version bump invalidates reuse
            l3 = StepLedger(ws.manifest_path, stage="qpb", workspace_root=root)
            r3 = l3.record_derived_step(
                "sum_band",
                description="sum_flux_image",
                params={"band": "soft"},
                inputs={"src": in_file},
                compute=compute_sum,
                algorithm_version="2",
            )
            self.assertEqual(r3.status, "completed")
            self.assertEqual(calls, 2)

            # Input SHA-256 change invalidates reuse
            in_file.write_bytes(b"IN2")
            l4 = StepLedger(ws.manifest_path, stage="qpb", workspace_root=root)
            r4 = l4.record_derived_step(
                "sum_band",
                description="sum_flux_image",
                params={"band": "soft"},
                inputs={"src": in_file},
                compute=compute_sum,
                algorithm_version="2",
            )
            self.assertEqual(r4.status, "completed")
            self.assertEqual(calls, 3)

    def test_v1_manifest_upgrade_and_failed_checkpoint_preserves_prior_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws, evt, cmd1, _ = self._build_two_step_workspace(root)
            # Legacy v1 manifest without schema_version == 2
            atomic_write_json(ws.manifest_path, {"stage": "stage_a", "legacy": True})

            runner = SASRunner(env=SASEnv())
            ledger = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            self.assertIsNone(ledger.previous)
            ledger.run_step("s1", runner, cmd1, inputs={"evt": evt})
            ledger.commit({"s1": ws.step_dir("step1") / "out1.fits"})

            # Stage mismatch raises ValueError
            with self.assertRaises(ValueError):
                StepLedger(ws.manifest_path, stage="other_stage", workspace_root=root)

            # Failed step writes failed checkpoint while preserving previous products & s1
            bad_cmd = _inline_python_cmd(
                program="bad_task",
                cwd=ws.step_dir("step2"),
                script="import sys; sys.exit(2)",
                log_file="bad.log",
            )
            ledger2 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            ledger2.run_step("s1", runner, cmd1, inputs={"evt": evt})
            with self.assertRaises(SASCommandError):
                ledger2.run_step("s2", runner, bad_cmd)

            failed_raw = read_json(ws.manifest_path)
            self.assertEqual(failed_raw["status"], "failed")
            self.assertEqual(failed_raw["steps"]["s1"]["status"], "reused")
            self.assertEqual(failed_raw["steps"]["s2"]["status"], "failed")
            self.assertEqual(failed_raw["steps"]["s2"]["returncode"], 2)
            self.assertIn("s1", failed_raw["products"])

            # Verify FileIdentity.capture missing file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                FileIdentity.capture(root / "does_not_exist.fits")

    def test_expected_outputs_change_invalidates_and_core_outputs_alias_stays_synced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = StageWorkspace(workspace_root=root, stage_dir=root / "stage_a")
            ws.ensure_dirs()
            step_dir = ws.step_dir("step1")

            cmd = _inline_python_cmd(
                program="multi_out",
                cwd=step_dir,
                expected_outputs=["a.fits"],
                script=(
                    "from pathlib import Path; "
                    "Path('a.fits').write_bytes(b'A' * 16); "
                    "Path('b.fits').write_bytes(b'B' * 16)"
                ),
            )
            # core_outputs property tracks expected_outputs directly
            self.assertIs(cmd.core_outputs, cmd.expected_outputs)
            legacy_cmd = SASCommand(
                program="legacy", cwd=step_dir, core_outputs=["x.fits"]
            )
            self.assertEqual(legacy_cmd.expected_outputs, ["x.fits"])
            legacy_cmd.expected_outputs.append("y.fits")
            self.assertEqual(legacy_cmd.core_outputs, ["x.fits", "y.fits"])
            legacy_cmd.core_outputs = ["z.fits"]
            self.assertEqual(legacy_cmd.expected_outputs, ["z.fits"])

            runner = SASRunner(env=SASEnv())
            l1 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            r1 = l1.run_step("s1", runner, cmd)
            l1.commit({})
            self.assertEqual(r1.status, "completed")
            self.assertEqual(r1.expected_outputs, ["a.fits"])

            cmd_changed_outputs = _inline_python_cmd(
                program="multi_out",
                cwd=step_dir,
                expected_outputs=["a.fits", "b.fits"],
                script=(
                    "from pathlib import Path; "
                    "Path('a.fits').write_bytes(b'A' * 16); "
                    "Path('b.fits').write_bytes(b'B' * 16)"
                ),
            )
            l2 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            self.assertIsNone(l2.can_reuse("s1", cmd_changed_outputs))
            r2 = l2.run_step("s1", runner, cmd_changed_outputs)
            self.assertEqual(r2.status, "completed")
            self.assertEqual(r2.expected_outputs, ["a.fits", "b.fits"])

    def test_commit_preserves_unseen_steps_by_default_and_prunes_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws, evt, cmd1, cmd2 = self._build_two_step_workspace(root)
            runner = SASRunner(env=SASEnv())

            l1 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            l1.run_step("s1", runner, cmd1, inputs={"evt": evt})
            l1.run_step(
                "s2", runner, cmd2, inputs={"out1": ws.step_dir("step1") / "out1.fits"}
            )
            l1.commit({"final": ws.step_dir("step2") / "out2.fits"})

            # Incremental run touching only s1 preserves s2 by default
            l2 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            r1 = l2.run_step("s1", runner, cmd1, inputs={"evt": evt})
            self.assertEqual(r1.status, "reused")
            m2 = l2.commit({})
            self.assertIn("s1", m2.steps)
            self.assertIn("s2", m2.steps)
            self.assertEqual(m2.products, {})
            raw_m2 = read_json(ws.manifest_path)
            self.assertIn("s2", raw_m2["steps"])

            # prune_unseen_steps=True drops omitted s2
            l3 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
            l3.run_step("s1", runner, cmd1, inputs={"evt": evt})
            m3 = l3.commit({}, prune_unseen_steps=True)
            self.assertEqual(list(m3.steps.keys()), ["s1"])
            raw_m3 = read_json(ws.manifest_path)
            self.assertEqual(list(raw_m3["steps"].keys()), ["s1"])

    def test_relative_inputs_and_path_params_ignore_process_cwd(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other_cwd:
            root = Path(tmp)
            ws = StageWorkspace(workspace_root=root, stage_dir=root / "stage_a")
            ws.ensure_dirs()
            step_dir = ws.step_dir("step1")
            step_dir.mkdir(parents=True, exist_ok=True)
            local_in = step_dir / "local_in.fits"
            local_in.write_bytes(b"IN" * 16)

            cmd = _inline_python_cmd(
                program="rel_task",
                cwd=step_dir,
                params={"infile": Path("local_in.fits"), "sub": Path("../stage_file.fits")},
                expected_outputs=["out.fits"],
                script="from pathlib import Path; Path('out.fits').write_bytes(b'OUT' * 16)",
            )
            runner = SASRunner(env=SASEnv())

            orig_cwd = os.getcwd()
            try:
                os.chdir(other_cwd)
                spec = cmd.to_spec(base_dir=ws.stage_dir, workspace_root=root)
                self.assertEqual(spec["params"]["infile"], "local_in.fits")
                self.assertEqual(spec["params"]["sub"], "../stage_file.fits")

                l1 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
                r1 = l1.run_step("s1", runner, cmd, inputs={"in": Path("local_in.fits")})
                self.assertEqual(r1.status, "completed")
                l1.commit({})

                # Derived step with relative input & relative Path param under unrelated os.getcwd()
                derived_in = ws.stage_dir / "derived_in.fits"
                derived_in.write_bytes(b"DIN")
                derived_out = ws.stage_dir / "derived_out.fits"

                def _compute():
                    derived_out.write_bytes(b"DOUT")
                    return {"out": derived_out}

                r_der = l1.record_derived_step(
                    "der1",
                    description="derive",
                    params={"ref": Path("derived_in.fits")},
                    inputs={"src": Path("derived_in.fits")},
                    compute=_compute,
                    cwd=ws.stage_dir,
                )
                self.assertEqual(r_der.status, "completed")
                self.assertEqual(r_der.params["ref"], "derived_in.fits")
                l1.commit({})

                l2 = StepLedger(ws.manifest_path, stage="stage_a", workspace_root=root)
                self.assertEqual(
                    l2.can_reuse("s1", cmd, inputs={"in": Path("local_in.fits")}).status,
                    "reused",
                )
                r_der2 = l2.record_derived_step(
                    "der1",
                    description="derive",
                    params={"ref": Path("derived_in.fits")},
                    inputs={"src": Path("derived_in.fits")},
                    compute=_compute,
                    cwd=ws.stage_dir,
                )
                self.assertEqual(r_der2.status, "reused")
            finally:
                os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
