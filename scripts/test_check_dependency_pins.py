"""Standard-library regression suite; no package installation or external I/O."""
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from check_dependency_pins import check

class TestPinChecker(unittest.TestCase):
    def run_check(self, lock, extras):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"requirements.txt").write_text(lock)
            (p/"pyproject.toml").write_text(extras)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                return check(p/"requirements.txt",p/"pyproject.toml")

    def test_valid_pin_and_unpinned_extra(self):
        self.assertEqual(self.run_check("demo==1.2.3\n",'[project.optional-dependencies]\napi=["demo==1.2.3","other>=2"]\n'),0)

    def test_single_quoted_drift_is_not_skipped(self):
        self.assertEqual(self.run_check("demo==1.2.3\n",'[project.optional-dependencies]\napi=["demo==1.2.3", \'demo==9.9\']\n'),1)

    def test_comment_is_not_a_dependency(self):
        self.assertEqual(self.run_check("demo==1.2.3\n",'[project.optional-dependencies]\napi=["demo==1.2.3"] # "demo==9.9"\n'),0)

    def test_quoted_table_and_keys(self):
        self.assertEqual(self.run_check("Demo_Pkg==1.2.3\n",'[project."optional-dependencies"]\n"api"=[\'demo-pkg==1.2.3\']\n'),0)

    def test_compound_constraints_denied(self):
        for req in ("demo==1.2.3,!=1.2.3","demo==1.2.3; python_version < '3.11'","demo===1.2.3","demo>=1.2.3","demo==1.2.*"):
            with self.subTest(req=req):
                self.assertEqual(self.run_check("demo==1.2.3\n",f'[project.optional-dependencies]\napi=["{req}"]\n'),1)

    def test_duplicate_normalized_lock_entries(self):
        self.assertEqual(self.run_check("demo-pkg==1.2.3\nDemo_Pkg==9.9\n",'[project.optional-dependencies]\napi=["demo-pkg==9.9"]\n'),1)

    def test_unknown_lock_syntax_is_not_ignored(self):
        for lock in ("demo==1.2.3\n-r other.txt\n","demo==1.2.3; python_version > '3.10'\n","demo==1.2.3,!=1.2.3\n"):
            with self.subTest(lock=lock):
                self.assertEqual(self.run_check(lock,'[project.optional-dependencies]\napi=["demo==1.2.3"]\n'),1)

    def test_invalid_metadata(self):
        for metadata in ('[project.optional-dependencies]\napi=[1]\n','[project.optional-dependencies]\napi="demo==1.2.3"\n','[project.optional-dependencies]\napi=["demo==1.2.3"\n','[project]\nname="demo"\n','[project.optional-dependencies]\napi=["other>=1"]\n'):
            with self.subTest(metadata=metadata):
                self.assertEqual(self.run_check("demo==1.2.3\n",metadata),1)

    def test_local_version_exact(self):
        self.assertEqual(self.run_check("torch==2.6.0+cpu\n",'[project.optional-dependencies]\napi=["torch==2.6.0+cpu"]\n'),0)
        self.assertEqual(self.run_check("torch==2.6.0+cpu\n",'[project.optional-dependencies]\napi=["torch==2.6.0"]\n'),1)

if __name__=="__main__":
    unittest.main()
