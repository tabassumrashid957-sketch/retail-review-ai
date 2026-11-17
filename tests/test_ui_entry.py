"""Verify the UI launchers without starting a server or opening a browser."""
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import run_ui


ROOT = Path(__file__).resolve().parents[1]


class UIEntryTests(unittest.TestCase):
    def test_explicit_arguments_and_exit_status_are_forwarded_after_bootstrap(self):
        calls = []
        backend = ModuleType("shoplifting.web_app")
        backend.main = Mock(side_effect=lambda args: calls.append(("server", args)) or 7)
        with patch.object(run_ui, "use_project_environment", side_effect=lambda path: calls.append(("bootstrap", path))), patch.dict(sys.modules, {"shoplifting.web_app": backend}):
            arguments = ["--port", "9001", "--no-open"]
            self.assertEqual(run_ui.main(arguments), 7)
        self.assertEqual(calls, [("bootstrap", Path(run_ui.__file__)), ("server", arguments)])
        self.assertEqual(arguments, ["--port", "9001", "--no-open"])

    def test_default_arguments_come_from_the_command_line(self):
        backend = ModuleType("shoplifting.web_app")
        backend.main = Mock(return_value=0)
        with patch.object(run_ui, "use_project_environment"), patch.object(sys, "argv", ["run_ui.py", "--no-open"]), patch.dict(sys.modules, {"shoplifting.web_app": backend}):
            self.assertEqual(run_ui.main(), 0)
        backend.main.assert_called_once_with(["--no-open"])

    def test_importing_entry_does_not_import_the_web_backend(self):
        original_import = __import__

        def reject_backend_import(name, *args, **kwargs):
            if name == "shoplifting.web_app":
                self.fail("The web backend must load only after environment bootstrap")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_backend_import):
            importlib.reload(run_ui)

    def test_pycharm_configuration_uses_the_project_interpreter_and_entry(self):
        component = ET.parse(ROOT / ".run/Retail_Review_UI.run.xml").getroot()
        config = component.find("configuration")
        self.assertEqual(config.attrib["name"], "Retail Review UI")
        self.assertEqual(config.attrib["type"], "PythonConfigurationType")
        options = {item.attrib["name"]: item.attrib["value"] for item in config.findall("option")}
        self.assertEqual(options["SDK_HOME"], "$PROJECT_DIR$/.venv/bin/python")
        self.assertEqual(options["IS_MODULE_SDK"], "false")
        self.assertNotIn("SDK_NAME", options)
        self.assertEqual(options["SCRIPT_NAME"], "$PROJECT_DIR$/run_ui.py")
        self.assertEqual(options["WORKING_DIRECTORY"], "$PROJECT_DIR$")
        self.assertTrue(ROOT.joinpath("run_ui.py").is_file())

    def test_command_launcher_preserves_paths_arguments_and_server_exit_status(self):
        self.assertTrue(os.access(ROOT / "START_UI.command", os.X_OK))
        shell = shutil.which("zsh")
        if shell is None:
            self.skipTest("The macOS command launcher requires zsh")
        with tempfile.TemporaryDirectory(prefix="retail ui entry ") as folder:
            project = Path(folder)
            launcher = project / "START_UI.command"
            shutil.copy2(ROOT / launcher.name, launcher)
            interpreter = project / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/bin/zsh\nprintf "%s\\0" "$PWD" "$@" > "$UI_TEST_CAPTURE"\nexit 7\n')
            interpreter.chmod(0o755)
            capture = project / "capture.bin"
            environment = {**os.environ, "UI_TEST_CAPTURE": str(capture)}
            result = subprocess.run([shell, str(launcher), "--port", "9001", "--no-open", "a path with spaces"], env=environment, cwd="/", capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(capture.read_bytes().decode().split("\0")[:-1], [str(project), "-u", str(project / "run_ui.py"), "--port", "9001", "--no-open", "a path with spaces"])


if __name__ == "__main__":
    unittest.main()
