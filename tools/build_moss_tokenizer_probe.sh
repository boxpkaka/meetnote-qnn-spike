#!/usr/bin/env bash
set -euo pipefail

MNN_ROOT="${MNN_ROOT:-}"
OUTPUT="${OUTPUT:-}"
CXX="${CXX:-c++}"

[[ -d "$MNN_ROOT/transformers/llm/engine/src/tokenizer" ]] || {
  echo "ERROR: MNN_ROOT must point to the patched MNN checkout" >&2
  exit 1
}
[[ -n "$OUTPUT" ]] || { echo "ERROR: OUTPUT is required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$CXX" -std=c++17 -O2 -DLLM_USE_JINJA \
  -I "$MNN_ROOT/include" \
  -I "$MNN_ROOT/3rd_party" \
  -I "$MNN_ROOT/source" \
  -I "$MNN_ROOT/transformers/llm/engine/src" \
  -I "$MNN_ROOT/transformers/llm/engine/src/tokenizer" \
  "$SCRIPT_DIR/moss_tokenizer_probe.cpp" \
  "$MNN_ROOT/transformers/llm/engine/src/tokenizer/tokenizer.cpp" \
  "$MNN_ROOT/transformers/llm/engine/src/tokenizer/unicode.cpp" \
  "$MNN_ROOT/transformers/llm/engine/src/tokenizer/unicode_data.cpp" \
  "$MNN_ROOT/source/core/AutoTime.cpp" \
  -o "$OUTPUT"
file "$OUTPUT"
