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

series_has_state() {
  local state="$1"
  local temporary_index
  temporary_index="$(mktemp)"
  rm -f "$temporary_index"
  trap 'rm -f "$temporary_index"' RETURN

  GIT_INDEX_FILE="$temporary_index" git -C "$MNN_ROOT" read-tree HEAD
  if [[ "$state" == applied ]]; then
    GIT_INDEX_FILE="$temporary_index" git -C "$MNN_ROOT" add -A
  fi

  local ordered=("${patches[@]}")
  if [[ "$state" == applied ]]; then
    ordered=()
    local index
    for ((index=${#patches[@]} - 1; index >= 0; index--)); do
      ordered+=("${patches[index]}")
    done
  fi

  local patch command=(git -C "$MNN_ROOT" apply --cached --unidiff-zero)
  [[ "$state" == applied ]] && command+=(--reverse)
  for patch in "${ordered[@]}"; do
    if ! GIT_INDEX_FILE="$temporary_index" "${command[@]}" "$patch" >/dev/null 2>&1; then
      return 1
    fi
  done
}

if series_has_state applied; then
  echo "[SKIP] complete MNN patch series already applied"
  exit 0
fi
if ! series_has_state applicable; then
  echo "[ERROR] MNN patch series is neither fully applicable nor fully applied"
  exit 1
fi

for patch in "${patches[@]}"; do
  git -C "$MNN_ROOT" apply --unidiff-zero "$patch"
  echo "[OK] applied $(basename "$patch")"
done
