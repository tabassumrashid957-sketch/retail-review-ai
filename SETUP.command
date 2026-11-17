#!/bin/zsh
# Install missing packages only into this project's virtual environment.
set -u
PROJECT_DIR="${0:A:h}"
cd -- "$PROJECT_DIR" || exit 1

finish() {
    local result="$1"
    if [[ -t 0 ]]; then
        read -r "?Press Enter to close this window..."
    fi
    exit "$result"
}

print "Setting up the local video project in: $PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        print "Python 3.10 or newer is required. Install Python, then reopen SETUP.command."
        finish 1
    fi
    if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
        print "Python 3.10 or newer is required. Install a current Python version, then reopen SETUP.command."
        finish 1
    fi
    python3 -m venv .venv || finish 1
fi

if ! .venv/bin/python scripts/verify_project.py --dependencies-only --json >/dev/null; then
    print "Installing missing project dependencies. Internet may be needed for packages."
    .venv/bin/python -m pip install -r requirements.txt || finish 1
else
    print "Project dependencies are already installed."
fi

print "Checking local model and encoder files..."
.venv/bin/python scripts/download_encoder.py || finish 1
.venv/bin/python scripts/verify_project.py
setup_result=$?
print "Setup restores the official encoder cache if needed. It does not download datasets or retrain the model."
finish "$setup_result"
