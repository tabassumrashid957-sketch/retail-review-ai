"""Run the local review app from PyCharm or a terminal.

The app accepts --port (default: 8765) and --no-open. The web app owns
argument parsing so both launch methods use the same settings.
"""
from pathlib import Path
import sys

from detect import use_project_environment


def main(argv=None):
    use_project_environment(Path(__file__))
    from shoplifting.web_app import main as run_web_app

    arguments = sys.argv[1:] if argv is None else list(argv)
    return run_web_app(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
