"""Admission checks for disposable local fixture tooling; no real Docker calls."""
import ast
import os
import pathlib
import re
import stat
import subprocess
import sys
import unittest
from unittest.mock import patch


def load_functions(path, names):
    root = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((root / path).read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in funcs} == names
    ns = dict(os=os, stat=stat, re=re, subprocess=subprocess, DOCKER_SOCKET="/var/run/docker.sock")
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns


class FixtureAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions("scripts/verify_disposable_ingest_e2e.py",
                                 {"cmd", "fixture_present", "guard"})

    def test_local_endpoint_pinned(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(os, "stat") as st, \
                patch.object(subprocess, "check_output", return_value="ok") as call:
            st.return_value.st_mode = stat.S_IFSOCK
            self.assertEqual(self.ns["cmd"]("docker", "image", "ls"), "ok")
            self.assertEqual(call.call_args.args[0],
                             ("docker", "--host", "unix:///var/run/docker.sock", "image", "ls"))

    def test_endpoint_overrides_fail_before_command(self):
        for key in ("DOCKER_HOST", "DOCKER_CONTEXT"):
            with self.subTest(key=key), patch.dict(os.environ, {key:"fixture"}, clear=True), \
                    patch.object(subprocess, "check_output") as call:
                with self.assertRaises(RuntimeError):
                    self.ns["cmd"]("docker", "ps")
                call.assert_not_called()

    def test_non_socket_rejected(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(os, "stat") as st, \
                patch.object(subprocess, "check_output") as call:
            st.return_value.st_mode = stat.S_IFREG
            with self.assertRaises(RuntimeError):
                self.ns["cmd"]("docker", "ps")
            call.assert_not_called()

    def test_cleanup_requires_successful_exact_inventory(self):
        cid = "a"*64
        for value, expected in (("",False),(cid,True)):
            with patch.dict(self.ns, {"cmd": lambda *a, **kw: value}):
                self.assertEqual(self.ns["fixture_present"](cid), expected)
        with patch.dict(self.ns, {"cmd": lambda *a, **kw: "b"*64}):
            with self.assertRaises(RuntimeError):
                self.ns["fixture_present"](cid)
        def failed(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "docker")
        with patch.dict(self.ns, {"cmd": failed}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.ns["fixture_present"](cid)
        with self.assertRaises(ValueError):
            self.ns["fixture_present"]("short")

    def test_optimized_harness_refuses_before_admission(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for path, func in (("scripts/verify_disposable_ingest_e2e.py", "guard"),
                           ("scripts/verify_installed_executor.py", "main")):
            code = ("import ast, pathlib; p=pathlib.Path(" + repr(str(root/path)) + "); "
                    "t=ast.parse(p.read_text()); n=next(x for x in t.body if isinstance(x,ast.FunctionDef) "
                    "and x.name==" + repr(func) + "); "
                    "exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec')); " + func + "()")
            result = subprocess.run([sys.executable,"-O","-c",code],capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("requires Python assertions enabled",result.stderr)


if __name__ == "__main__":
    unittest.main()
