"""Check the single UI run target and interpreter recovery."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import detect


ROOT = Path(__file__).resolve().parents[1]


class PyCharmEntryTests(unittest.TestCase):
    def test_structured_prediction_result_is_not_used_as_an_error_exit_code(self):
        predictor = ModuleType("shoplifting.predict")
        predictor.main = Mock(return_value={"status": "insufficient_evidence"})
        with patch.object(detect, "use_project_environment"), patch.object(detect.sys, "argv", ["detect.py", "video.mp4"]), patch.dict(sys.modules, {"shoplifting.predict": predictor}):
            self.assertEqual(detect.main(), 0)
        predictor.main.assert_called_once_with()

    def test_wrong_interpreter_keeps_the_requested_ui_entry_point(self):
        # A clean checkout has no virtual environment. Test recovery against a
        # real temporary interpreter path instead of the author's environment.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            script = root / "run_ui.py"
            local_python = root / ".venv/bin/python"
            local_python.parent.mkdir(parents=True)
            local_python.write_bytes(b"interpreter fixture")
            arguments = [str(script), "--port", "9001", "--no-open"]
            with patch.object(detect, "__file__", str(root / "detect.py")), patch.object(detect.sys, "prefix", "/another/python"), patch.object(detect.sys, "argv", arguments), patch.object(detect.os, "execv") as execute:
                detect.use_project_environment(script)
            interpreter = str(local_python)
        execute.assert_called_once_with(interpreter, [interpreter, *arguments])

    def test_only_retail_review_ui_is_exposed_as_a_shared_run_configuration(self):
        configurations = sorted(path.name for path in (ROOT / ".run").glob("*.run.xml"))
        self.assertEqual(configurations, ["Retail_Review_UI.run.xml"])

    def test_running_detect_without_arguments_opens_the_web_ui(self):
        backend = ModuleType("shoplifting.web_app")
        backend.main = Mock(return_value=0)
        legacy = ModuleType("shoplifting.launcher")
        legacy.main = Mock(side_effect=AssertionError("The legacy video chooser must not open"))
        with patch.object(detect, "use_project_environment"), patch.object(detect.sys, "argv", ["detect.py"]), patch.dict(sys.modules, {"shoplifting.web_app": backend, "shoplifting.launcher": legacy}):
            self.assertEqual(detect.main(), 0)
        backend.main.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
