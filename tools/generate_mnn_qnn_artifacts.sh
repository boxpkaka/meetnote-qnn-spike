#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$ROOT_DIR/third_party/MNN}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/local-assets/models/Qwen3-4B-MNN}"
MNN_QNN_HOST_BUILD_DIR="${MNN_QNN_HOST_BUILD_DIR:-$MNN_ROOT/build_qnn_host}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/build/qnn-spike/qnn-convert-cache}"
SOC_ID="${SOC_ID:-}"
DSP_ARCH="${DSP_ARCH:-v81}"
MAX_HISTORY_TOKEN="${MAX_HISTORY_TOKEN:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
ALLOW_UNVERIFIED_CHUNK_SIZE="${ALLOW_UNVERIFIED_CHUNK_SIZE:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "$2 missing: $1"
}

require_exec() {
  [[ -x "$1" ]] || die "$2 missing or not executable: $1"
}

require_linux_x86_64() {
  local kernel machine
  kernel="$(uname -s)"
  machine="$(uname -m)"
  [[ "$kernel" == "Linux" && "$machine" == "x86_64" ]] || die \
    "QNN offline conversion must run on Linux x86_64; current host is $kernel $machine"
}

require_tool_is_linux_x86_64() {
  local path="$1"
  local description="$2"
  local kind
  kind="$(file "$path")"
  echo "$description: $kind"
  [[ "$kind" == *"ELF 64-bit"* && "$kind" == *"x86-64"* ]] || die \
    "$description must be a Linux x86_64 executable: $path"
}

usage() {
  cat <<EOF
Usage:
  QNN_SDK_ROOT=/opt/qcom/aistack/qairt/<version> \\
  SOC_ID=<qualcomm-soc-id> \\
  DSP_ARCH=v81 \\
  CHUNK_SIZE=64 \\
  MNN_QNN_HOST_BUILD_DIR=$MNN_ROOT/build_qnn_host \\
  MODEL_DIR=$MODEL_DIR \\
  $0

Required:
  QNN_SDK_ROOT must contain bin/x86_64-linux-clang/qnn-model-lib-generator,
  bin/x86_64-linux-clang/qnn-context-binary-generator,
  lib/x86_64-linux-clang/libQnnHtp.so, and
  lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so.

  MNN_QNN_HOST_BUILD_DIR must contain Linux x86_64 generateIO and compilefornpu.

  CHUNK_SIZE=64 is the smallest configuration verified on the SM8850/V81 test
  device. Smaller values require ALLOW_UNVERIFIED_CHUNK_SIZE=true and may create
  a QNN binary that crashes HTP during context loading.

Output:
  MODEL_DIR/config_qnn.json
  MODEL_DIR/qnn/llm.mnn
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -n "${QNN_SDK_ROOT:-}" ]] || die "QNN_SDK_ROOT is required"
[[ -n "$SOC_ID" ]] || die "SOC_ID is required; do not infer it from SM8850 or libQnnHtpV81.so"
[[ "$CHUNK_SIZE" =~ ^[0-9]+$ && "$CHUNK_SIZE" -gt 0 ]] || die \
  "CHUNK_SIZE must be a positive integer: $CHUNK_SIZE"
if ((CHUNK_SIZE < 64)) && [[ "$ALLOW_UNVERIFIED_CHUNK_SIZE" != "true" ]]; then
  die "CHUNK_SIZE=$CHUNK_SIZE is below the verified SM8850/V81 floor of 64; set ALLOW_UNVERIFIED_CHUNK_SIZE=true only for an isolated device experiment"
fi

require_linux_x86_64

require_file "$MODEL_DIR/config.json" "MNN LLM config"
require_file "$MODEL_DIR/llm_config.json" "MNN LLM model config"
require_file "$MODEL_DIR/llm.mnn" "MNN LLM model"
require_file "$MODEL_DIR/llm.mnn.weight" "MNN LLM weight"
if [[ ! -f "$MODEL_DIR/tokenizer.txt" && ! -f "$MODEL_DIR/tokenizer.mtok" ]]; then
  die "MNN tokenizer missing: $MODEL_DIR/tokenizer.txt or $MODEL_DIR/tokenizer.mtok"
fi

require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator" "QNN model lib generator"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" "QNN context binary generator"
require_file "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" "QNN HTP host backend"
require_file "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so" "QNN HTP netrun extensions"

require_exec "$MNN_QNN_HOST_BUILD_DIR/generateIO" "MNN generateIO"
require_exec "$MNN_QNN_HOST_BUILD_DIR/compilefornpu" "MNN compilefornpu"
require_tool_is_linux_x86_64 "$MNN_QNN_HOST_BUILD_DIR/generateIO" "MNN generateIO"
require_tool_is_linux_x86_64 "$MNN_QNN_HOST_BUILD_DIR/compilefornpu" "MNN compilefornpu"

GENERATOR_SOURCE="$MNN_ROOT/transformers/llm/export/npu/generate_llm_qnn.py"
require_file "$GENERATOR_SOURCE" "MNN QNN generator"

mkdir -p "$CACHE_DIR"

GENERATOR="$GENERATOR_SOURCE"
GENERATOR_TEMP=""
cleanup() {
  [[ -z "$GENERATOR_TEMP" ]] || rm -f "$GENERATOR_TEMP"
}
trap cleanup EXIT

if grep -Eq 'makeIOJson\(args,[[:space:]]*args\.chunk_size,' "$GENERATOR_SOURCE"; then
  :
elif grep -Eq 'makeIOJson\(args,[[:space:]]*128,' "$GENERATOR_SOURCE"; then
  GENERATOR_TEMP="$(mktemp "$CACHE_DIR/generate_llm_qnn.XXXXXX.py")"
  sed -E 's/makeIOJson\(args,[[:space:]]*128,/makeIOJson(args, args.chunk_size,/' \
    "$GENERATOR_SOURCE" > "$GENERATOR_TEMP"
  GENERATOR="$GENERATOR_TEMP"
  echo "Using a temporary generator copy that honors --chunk_size; MNN source is unchanged"
else
  die "cannot verify how $GENERATOR_SOURCE selects the prefill graph sequence length"
fi

cd "$MNN_ROOT"
echo "Generating QNN artifacts for model: $MODEL_DIR"
echo "QNN_SDK_ROOT=$QNN_SDK_ROOT"
echo "SOC_ID=$SOC_ID DSP_ARCH=$DSP_ARCH MAX_HISTORY_TOKEN=$MAX_HISTORY_TOKEN CHUNK_SIZE=$CHUNK_SIZE"

export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
if [[ -d /usr/include/c++ ]]; then
  for cxx_version_dir in /usr/include/c++/*; do
    [[ -d "$cxx_version_dir" ]] || continue
    cxx_version="$(basename "$cxx_version_dir")"
    export CPLUS_INCLUDE_PATH="$cxx_version_dir:/usr/include/x86_64-linux-gnu/c++/$cxx_version:$cxx_version_dir/backward:${CPLUS_INCLUDE_PATH:-}"
    if [[ -d "/usr/lib/gcc/x86_64-linux-gnu/$cxx_version" ]]; then
      export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/$cxx_version:${LIBRARY_PATH:-}"
    fi
  done
fi

"$PYTHON_BIN" "$GENERATOR" \
  --model "$MODEL_DIR" \
  --mnn_path "$MNN_QNN_HOST_BUILD_DIR" \
  --cache_path "$CACHE_DIR" \
  --soc_id "$SOC_ID" \
  --dsp_arch "$DSP_ARCH" \
  --model_name llm.mnn \
  --chunk_size "$CHUNK_SIZE" \
  --max_history_token "$MAX_HISTORY_TOKEN" \
  --need_config_json true

require_file "$MODEL_DIR/config_qnn.json" "generated QNN config"
require_file "$MODEL_DIR/qnn/llm.mnn" "generated QNN model"
"$PYTHON_BIN" - "$MODEL_DIR/config_qnn.json" "$CHUNK_SIZE" <<'PY'
import json
import sys

config_path = sys.argv[1]
expected = [int(sys.argv[2]), 1]
with open(config_path, encoding="utf-8") as handle:
    actual = json.load(handle).get("chunk_limits")
if actual != expected:
    raise SystemExit(f"generated chunk_limits mismatch: expected {expected}, got {actual}")
PY
GRAPH_COUNT="$(find "$MODEL_DIR/qnn" -maxdepth 1 -name 'graph*.bin' -type f | wc -l)"
((GRAPH_COUNT > 0)) || die "generated QNN graph binaries missing: $MODEL_DIR/qnn"
echo "QNN artifacts generated:"
echo "  $MODEL_DIR/config_qnn.json"
echo "  $MODEL_DIR/qnn/llm.mnn"
echo "  graph binaries: $GRAPH_COUNT"
