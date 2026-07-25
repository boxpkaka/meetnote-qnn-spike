#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$ROOT_DIR/third_party/MNN}"
MNN_QNN_HOST_BUILD_DIR="${MNN_QNN_HOST_BUILD_DIR:-$MNN_ROOT/build_qnn_host}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_linux_x86_64() {
  local kernel machine
  kernel="$(uname -s)"
  machine="$(uname -m)"
  [[ "$kernel" == "Linux" && "$machine" == "x86_64" ]] || die \
    "MNN QNN host tools must be built on Linux x86_64 for the QNN offline pipeline; current host is $kernel $machine"
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

usage() {
  cat <<EOF
Usage:
  QNN_SDK_ROOT=/opt/qcom/aistack/qairt/<version> \\
  MNN_QNN_HOST_BUILD_DIR=$MNN_QNN_HOST_BUILD_DIR \\
  $0

Output:
  MNN_QNN_HOST_BUILD_DIR/generateIO
  MNN_QNN_HOST_BUILD_DIR/compilefornpu
  MNN_QNN_HOST_BUILD_DIR/MNNConvert

This only builds MNN's Linux x86_64 helper tools. Run
tools/generate_mnn_qnn_artifacts.sh afterwards to generate config_qnn.json
and qnn/llm.mnn.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -n "${QNN_SDK_ROOT:-}" ]] || die "QNN_SDK_ROOT is required"
[[ -d "$QNN_SDK_ROOT/include/QNN" ]] || die "QNN SDK headers missing: $QNN_SDK_ROOT/include/QNN"

require_linux_x86_64
require_tool cmake

mkdir -p "$MNN_QNN_HOST_BUILD_DIR"

cmake -S "$MNN_ROOT" -B "$MNN_QNN_HOST_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_WITH_PLUGIN=OFF \
  -DMNN_QNN=ON \
  -DMNN_QNN_CONVERT_MODE=ON \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT"

cmake --build "$MNN_QNN_HOST_BUILD_DIR" \
  --target MNNConvert generateIO compilefornpu \
  -j"$(nproc)"

file "$MNN_QNN_HOST_BUILD_DIR/MNNConvert"
file "$MNN_QNN_HOST_BUILD_DIR/generateIO"
file "$MNN_QNN_HOST_BUILD_DIR/compilefornpu"
