#!/bin/zsh
# Start the local video review app using this project's installed Python.
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  print "Please double-click SETUP.command first."
  if [[ -t 0 ]]; then read "?Press Return to close this window."; fi
  exit 1
fi
exec "$PROJECT_DIR/.venv/bin/python" -u "$PROJECT_DIR/run_ui.py" "$@"
