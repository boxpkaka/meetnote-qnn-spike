#!/usr/bin/env bash
set -euo pipefail

MNN_ROOT="${MNN_ROOT:-}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-}"
OUTPUT="${OUTPUT:-}"
CXX="${CXX:-c++}"

[[ -f "$MNN_ROOT/transformers/llm/engine/include/llm/llm.hpp" ]] || {
  echo "ERROR: MNN_ROOT is required" >&2
  exit 1
}
for library in libMNN.so express/libMNN_Express.so tools/audio/libMNNAudio.so libllm.so; do
  [[ -f "$MNN_BUILD_DIR/$library" ]] || {
    echo "ERROR: MNN_BUILD_DIR is missing $library" >&2
    exit 1
  }
done
[[ -n "$OUTPUT" ]] || { echo "ERROR: OUTPUT is required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$CXX" -std=c++17 -O2 \
  -I "$MNN_ROOT/transformers/llm/engine/include" \
  -I "$MNN_ROOT/include" \
  "$SCRIPT_DIR/mnn_teacher_forced_logits.cpp" \
  -L "$MNN_BUILD_DIR" -L "$MNN_BUILD_DIR/express" -L "$MNN_BUILD_DIR/tools/audio" \
  -Wl,-rpath,"$MNN_BUILD_DIR:$MNN_BUILD_DIR/express:$MNN_BUILD_DIR/tools/audio" \
  -lllm -lMNNAudio -lMNN_Express -lMNN -o "$OUTPUT"
file "$OUTPUT"
