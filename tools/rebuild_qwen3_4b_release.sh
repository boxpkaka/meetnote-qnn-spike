#!/usr/bin/env bash
# Rebuild the pinned SM8850/V81 Qwen3-4B release in a new external workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_DIR="$REPO_ROOT/releases/qwen3-4b-sm8850-v81-c64-rope-cpu"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT/third_party/MNN}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-$MNN_ROOT/build_qnn_host}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_MODEL_DIR="${HF_MODEL_DIR:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
WORK_DIR="${WORK_DIR:-}"
SOC_ID="${SOC_ID:-87}"
DSP_ARCH="${DSP_ARCH:-v81}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
MAX_HISTORY_TOKEN="${MAX_HISTORY_TOKEN:-0}"
BUILD_MNN_TOOLS="${BUILD_MNN_TOOLS:-true}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_exec() {
  [[ -x "$1" ]] || die "missing executable: $1"
}

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  die "release rebuild requires Linux x86_64"
[[ -n "$HF_MODEL_DIR" ]] || die "HF_MODEL_DIR is required"
[[ -n "$QNN_SDK_ROOT" ]] || die "QNN_SDK_ROOT is required"
[[ -n "$WORK_DIR" ]] || die "WORK_DIR is required and must name a new directory"
[[ ! -e "$WORK_DIR" ]] || die "WORK_DIR already exists: $WORK_DIR"
[[ "$SOC_ID" == "87" ]] || die "the pinned release requires SOC_ID=87"
[[ "$DSP_ARCH" == "v81" ]] || die "the pinned release requires DSP_ARCH=v81"
[[ "$CHUNK_SIZE" == "64" ]] || die "the pinned release requires CHUNK_SIZE=64"
[[ "$MAX_HISTORY_TOKEN" == "0" ]] || \
  die "the pinned release requires MAX_HISTORY_TOKEN=0"

require_file "$MNN_ROOT/transformers/llm/export/llmexport.py"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"

mkdir -p "$WORK_DIR/logs"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
SOURCE_MANIFEST="$RELEASE_DIR/source-manifest.json"
EXPORTED_MODEL="$WORK_DIR/exported-model"
BASELINE_MODEL="$WORK_DIR/baseline-model"
HYBRID_MODEL="$WORK_DIR/hybrid-model"
BASELINE_CACHE="$WORK_DIR/baseline-cache"
HYBRID_CACHE="$WORK_DIR/hybrid-cache"
OUTPUT_QNN="$WORK_DIR/release/qnn"

"$PYTHON_BIN" "$SCRIPT_DIR/verify_source_manifest.py" \
  "$HF_MODEL_DIR" \
  --manifest "$SOURCE_MANIFEST" | tee "$WORK_DIR/logs/source-verification.json"

actual_mnn_revision="$(git -C "$MNN_ROOT" rev-parse HEAD)"
expected_mnn_revision="0bff03cbef43c783f44e41484b9f8a0b28bd758d"
[[ "$actual_mnn_revision" == "$expected_mnn_revision" ]] || \
  die "MNN revision mismatch: expected $expected_mnn_revision, got $actual_mnn_revision"

if [[ "$BUILD_MNN_TOOLS" == "true" ]]; then
  [[ -z "$(git -C "$MNN_ROOT" status --porcelain)" ]] || \
    die "MNN must be clean before applying project patches"
  MNN_ROOT="$MNN_ROOT" "$SCRIPT_DIR/apply_mnn_patches.sh" \
    2>&1 | tee "$WORK_DIR/logs/apply-mnn-patches.log"
  QNN_SDK_ROOT="$QNN_SDK_ROOT" \
    MNN_ROOT="$MNN_ROOT" \
    MNN_QNN_HOST_BUILD_DIR="$MNN_BUILD_DIR" \
    "$SCRIPT_DIR/build_mnn_qnn_host_tools.sh" \
    2>&1 | tee "$WORK_DIR/logs/build-mnn-tools.log"
elif [[ "$BUILD_MNN_TOOLS" != "false" ]]; then
  die "BUILD_MNN_TOOLS must be true or false"
fi

require_exec "$MNN_BUILD_DIR/MNNConvert"
require_exec "$MNN_BUILD_DIR/generateIO"
require_exec "$MNN_BUILD_DIR/compilefornpu"

export QNN_SDK_ROOT
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0

if [[ -d /usr/include/c++ ]]; then
  for version_dir in /usr/include/c++/*; do
    [[ -d "$version_dir" ]] || continue
    version="$(basename "$version_dir")"
    export CPLUS_INCLUDE_PATH="$version_dir:/usr/include/x86_64-linux-gnu/c++/$version:$version_dir/backward:${CPLUS_INCLUDE_PATH:-}"
    if [[ -d "/usr/lib/gcc/x86_64-linux-gnu/$version" ]]; then
      export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/$version:${LIBRARY_PATH:-}"
    fi
  done
fi

echo "[1/5] Exporting the pinned W4A16 MNN model"
(
  cd "$MNN_ROOT/transformers/llm/export"
  /usr/bin/time -v "$PYTHON_BIN" llmexport.py \
    --path "$HF_MODEL_DIR" \
    --tokenizer_path "$HF_MODEL_DIR" \
    --dst_path "$EXPORTED_MODEL" \
    --export mnn \
    --mnnconvert "$MNN_BUILD_DIR/MNNConvert" \
    --quant_bit 4 \
    --quant_block 64 \
    --lm_quant_bit 4 \
    --lm_quant_block 64 \
    --seperate_embed \
    --generate_for_npu \
    --act_bit 16 \
    --sym \
    --omni \
    --hqq \
    --omni_epochs 1
) 2>&1 | tee "$WORK_DIR/logs/export.log"

for file in config.json llm_config.json llm.mnn llm.mnn.weight; do
  require_file "$EXPORTED_MODEL/$file"
done
if [[ ! -f "$EXPORTED_MODEL/tokenizer.mtok" && \
      ! -f "$EXPORTED_MODEL/tokenizer.txt" ]]; then
  die "exported tokenizer is missing"
fi

echo "[2/5] Creating isolated baseline and hybrid inputs"
cp -a "$EXPORTED_MODEL" "$BASELINE_MODEL"
cp -a "$EXPORTED_MODEL" "$HYBRID_MODEL"

COMMON_GENERATOR_ARGS=(
  --model
  "$BASELINE_MODEL"
  --mnn_path
  "$MNN_BUILD_DIR"
  --cache_path
  "$BASELINE_CACHE"
  --soc_id
  "$SOC_ID"
  --dsp_arch
  "$DSP_ARCH"
  --model_name
  llm.mnn
  --chunk_size
  "$CHUNK_SIZE"
  --max_history_token
  "$MAX_HISTORY_TOKEN"
  --need_config_json
  true
)

echo "[3/5] Generating the wide-logits baseline"
(
  cd "$MNN_ROOT"
  "$PYTHON_BIN" \
    "$REPO_ROOT/experiments/ablation/generate_llm_qnn_wide64.py" \
    "${COMMON_GENERATOR_ARGS[@]}"
) 2>&1 | tee "$WORK_DIR/logs/baseline.log"

COMMON_GENERATOR_ARGS[1]="$HYBRID_MODEL"
COMMON_GENERATOR_ARGS[5]="$HYBRID_CACHE"

echo "[4/5] Generating the CPU phase-multiply and trigonometry prefix"
(
  cd "$MNN_ROOT"
  "$PYTHON_BIN" \
    "$REPO_ROOT/experiments/ablation/generate_qnn_with_cpu_ops.py" \
    --base-generator \
    "$REPO_ROOT/experiments/ablation/generate_llm_qnn_wide64.py" \
    --cpu-op /rotary/Mul_output_0 \
    --cpu-op /rotary/Cos_output_0 \
    --cpu-op /rotary/Sin_output_0 \
    "${COMMON_GENERATOR_ARGS[@]}"
) 2>&1 | tee "$WORK_DIR/logs/hybrid.log"

echo "[5/5] Assembling and verifying the 39-context release"
"$PYTHON_BIN" "$SCRIPT_DIR/assemble_qnn_cpu_rope_model.py" \
  --mnn-convert "$MNN_BUILD_DIR/MNNConvert" \
  --baseline-qnn-dir "$BASELINE_MODEL/qnn" \
  --hybrid-qnn-dir "$HYBRID_MODEL/qnn" \
  --output-qnn-dir "$OUTPUT_QNN"

"$PYTHON_BIN" "$SCRIPT_DIR/verify_release_manifest.py" "$OUTPUT_QNN" \
  | tee "$WORK_DIR/logs/release-verification.json"

"$PYTHON_BIN" - "$WORK_DIR" "$REPO_ROOT" "$MNN_ROOT" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

work_dir, repo_root, mnn_root = map(Path, sys.argv[1:])
record = {
    "format": "meetnote.clean_rebuild.v1",
    "repository_revision": subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "mnn_revision": subprocess.check_output(
        ["git", "-C", str(mnn_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "platform": platform.platform(),
    "work_dir": str(work_dir),
    "status": "server-build-complete",
}
(work_dir / "rebuild.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

echo "Release rebuild completed: $OUTPUT_QNN"
