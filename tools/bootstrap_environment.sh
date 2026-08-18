#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-host}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$PROFILE" in
  host)
    VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
    PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.10}"
    LOCK_FILE="$ROOT_DIR/requirements-host.lock"
    EXPECTED_PYTHON="3.10"
    ;;
  export)
    VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-export}"
    PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.13}"
    LOCK_FILE="$ROOT_DIR/requirements-export.lock"
    EXPECTED_PYTHON="3.13"
    EXPORT_CHECK_ARGS=(--fresh-export --export-device cuda)
    ;;
  export-cpu)
    VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-export-cpu}"
    PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.13}"
    LOCK_FILE="$ROOT_DIR/requirements-export-cpu.lock"
    EXPECTED_PYTHON="3.13"
    EXPORT_CHECK_ARGS=(--fresh-export --export-device cpu)
    ;;
  *)
    die "PROFILE must be host, export, or export-cpu"
    ;;
esac

command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 || die "$PYTHON_BOOTSTRAP is required"
[[ -f "$LOCK_FILE" ]] || die "missing lock file: $LOCK_FILE"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
fi

actual_version="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$actual_version" == "$EXPECTED_PYTHON" ]] || \
  die "Python $EXPECTED_PYTHON is required for PROFILE=$PROFILE; venv has $actual_version"

"$VENV_DIR/bin/python" -m pip install --upgrade "pip==26.2"
"$VENV_DIR/bin/python" -m pip install --requirement "$LOCK_FILE"
if [[ "$PROFILE" == "host" ]]; then
  "$VENV_DIR/bin/python" "$ROOT_DIR/tools/check_environment.py" --python-only
else
  "$VENV_DIR/bin/python" "$ROOT_DIR/tools/check_environment.py" \
    --python-only "${EXPORT_CHECK_ARGS[@]}"
fi

echo "Environment ready. Activate with: source $VENV_DIR/bin/activate"
