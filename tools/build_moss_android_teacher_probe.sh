#!/usr/bin/env bash
set -euo pipefail

MNN_ROOT="${MNN_ROOT:-}"
RELEASE_DIR="${RELEASE_DIR:-}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${ANDROID_NDK:-}}"
OUTPUT="${OUTPUT:-}"

[[ -f "$MNN_ROOT/transformers/llm/engine/include/llm/llm.hpp" ]] || {
  echo "ERROR: MNN_ROOT is required" >&2
  exit 1
}
prebuilt="$(find "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt" -maxdepth 1 -mindepth 1 -type d -print -quit)"
CXX="$prebuilt/bin/aarch64-linux-android28-clang++"
[[ -x "$CXX" ]] || { echo "ERROR: ANDROID_NDK_ROOT is required" >&2; exit 1; }
for library in libllm.so libMNNAudio.so libMNN_Express.so libMNN.so; do
  [[ -f "$RELEASE_DIR/lib/$library" ]] || { echo "ERROR: release is missing lib/$library" >&2; exit 1; }
done
[[ -n "$OUTPUT" ]] || { echo "ERROR: OUTPUT is required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$CXX" -std=c++17 -O2 -fPIE -pie \
  "$SCRIPT_DIR/mnn_teacher_forced_logits.cpp" \
  -I "$MNN_ROOT/transformers/llm/engine/include" -I "$MNN_ROOT/include" \
  -L "$RELEASE_DIR/lib" -Wl,-rpath,'$ORIGIN/lib' \
  -lllm -lMNNAudio -lMNN_Express -lMNN -o "$OUTPUT"
file "$OUTPUT"
