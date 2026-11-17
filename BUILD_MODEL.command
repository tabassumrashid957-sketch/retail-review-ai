#!/bin/zsh
# Reproduce training and preserve the previous model as a backup.
PROJECT_DIR="${0:A:h}"
cd -- "$PROJECT_DIR" || exit 1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  print "Please double-click SETUP.command first."
  exit 1
fi
candidate="artifacts/model-candidate-$(date -u +%Y%m%d-%H%M%S)"
"$PROJECT_DIR/.venv/bin/python" -u scripts/run_pipeline.py --output "$candidate" --promote
run_status=$?
if [[ -t 0 ]]; then read "?Press Return to finish."; fi
exit $run_status
