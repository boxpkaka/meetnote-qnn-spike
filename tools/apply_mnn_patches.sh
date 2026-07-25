#!/usr/bin/env bash
# Apply project-owned MNN patches without resetting unrelated local work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT/third_party/MNN}"
PATCH_DIR="$REPO_ROOT/third_party/patches/mnn"
EXPECTED_BASE="0bff03cbef43c783f44e41484b9f8a0b28bd758d"

if [[ ! -e "$MNN_ROOT/.git" ]]; then
  echo "[ERROR] $MNN_ROOT is not a git checkout at the verified MNN revision"
  exit 1
fi

if [[ "$(git -C "$MNN_ROOT" rev-parse HEAD)" != "$EXPECTED_BASE" ]]; then
  echo "[ERROR] MNN is not at the verified base $EXPECTED_BASE"
  exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if (( ${#patches[@]} == 0 )); then
  echo "[ERROR] no patches under $PATCH_DIR"
  exit 1
fi

for patch in "${patches[@]}"; do
  if git -C "$MNN_ROOT" apply --unidiff-zero --reverse --check "$patch" >/dev/null 2>&1; then
    echo "[SKIP] $(basename "$patch") already applied"
  elif git -C "$MNN_ROOT" apply --unidiff-zero --check "$patch"; then
    git -C "$MNN_ROOT" apply --unidiff-zero "$patch"
    echo "[OK] applied $(basename "$patch")"
  else
    echo "[ERROR] cannot apply $(basename "$patch"); inspect the MNN working tree"
    exit 1
  fi
done
