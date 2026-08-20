#!/usr/bin/env bash
set -euo pipefail

MNN_ROOT="${MNN_ROOT:-}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-}"
OUTPUT="${OUTPUT:-}"
CXX="${CXX:-c++}"

[[ -f "$MNN_ROOT/include/MNN/expr/Module.hpp" ]] || { echo "ERROR: MNN_ROOT is required" >&2; exit 1; }
[[ -f "$MNN_BUILD_DIR/libMNN.so" && -f "$MNN_BUILD_DIR/express/libMNN_Express.so" ]] || {
  echo "ERROR: MNN_BUILD_DIR must contain host MNN and MNN_Express libraries" >&2
  exit 1
}
[[ -n "$OUTPUT" ]] || { echo "ERROR: OUTPUT is required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$CXX" -std=c++17 -O2 \
  -I "$MNN_ROOT/include" \
  "$SCRIPT_DIR/moss_mnn_graph_probe.cpp" \
  -L "$MNN_BUILD_DIR/express" -L "$MNN_BUILD_DIR" \
  -Wl,-rpath,"$MNN_BUILD_DIR/express:$MNN_BUILD_DIR" \
  -lMNN_Express -lMNN -o "$OUTPUT"
file "$OUTPUT"
