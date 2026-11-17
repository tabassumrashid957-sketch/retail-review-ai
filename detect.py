"""Internal video-inference worker; the app entry point is run_ui.py."""
import os
from pathlib import Path
import sys


def use_project_environment(entry_point=None):
    """Use the installed project interpreter when launched from a different IDE Python."""
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    interpreter = environment / "bin" / "python"
    if interpreter.is_file() and Path(sys.prefix).resolve() != environment.resolve():
        script = Path(entry_point).resolve() if entry_point is not None else Path(__file__).resolve()
        os.execv(str(interpreter), [str(interpreter), str(script), *sys.argv[1:]])

def main():
    use_project_environment()
    if len(sys.argv) == 1:
        from shoplifting.web_app import main as launch
        return launch([])
    else:
        from shoplifting.predict import main as predict
        predict()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
