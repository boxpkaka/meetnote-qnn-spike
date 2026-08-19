#!/usr/bin/env bash
# Rebuild the pinned MOSS-Transcribe-Diarize SM8850/V81 PoC in external storage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT/third_party/MNN}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-$MNN_ROOT/build_qnn_host}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR:-}"
CALIBRATION_ARCHIVE="${CALIBRATION_ARCHIVE:-}"
CALIBRATION_MANIFEST="${CALIBRATION_MANIFEST:-}"
CALIBRATION_LOGITS_MAX_ABS="${CALIBRATION_LOGITS_MAX_ABS:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${ANDROID_NDK:-}}"
WORK_DIR="${WORK_DIR:-}"
SOC_ID="${SOC_ID:-87}"
DSP_ARCH="${DSP_ARCH:-v81}"
VTCM_MB="${VTCM_MB:-8}"
HVX_THREADS="${HVX_THREADS:-8}"
BUILD_MNN_TOOLS="${BUILD_MNN_TOOLS:-true}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

RELEASE_ID="moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1"
SOURCE_MANIFEST="$REPO_ROOT/releases/$RELEASE_ID/source-manifest.json"

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_exec() { [[ -x "$1" ]] || die "missing executable: $1"; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,34p' "$0"
  cat <<EOF

Required environment: MODEL_DIR, CALIBRATION_ARCHIVE, CALIBRATION_MANIFEST,
QNN_SDK_ROOT, ANDROID_NDK_ROOT, WORK_DIR. CALIBRATION_LOGITS_MAX_ABS optionally overrides the manifest value.
Pinned environment: SOC_ID=87, DSP_ARCH=v81, VTCM_MB=8, HVX_THREADS=8,
QAIRT 2.48.40.260702.
EOF
  exit 0
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "requires Linux x86_64"
[[ "$SOC_ID" == 87 ]] || die "pinned PoC requires SOC_ID=87"
[[ "$DSP_ARCH" == v81 ]] || die "pinned PoC requires DSP_ARCH=v81"
[[ "$VTCM_MB" == 8 ]] || die "SM8850 requires the QAIRT-reported 8 MiB VTCM configuration"
[[ "$HVX_THREADS" == 8 ]] || die "SM8850 requires the QAIRT-reported 8 HVX threads"
[[ "$BUILD_MNN_TOOLS" == true || "$BUILD_MNN_TOOLS" == false ]] || die "BUILD_MNN_TOOLS must be true or false"
[[ "$PREFLIGHT_ONLY" == true || "$PREFLIGHT_ONLY" == false ]] || die "PREFLIGHT_ONLY must be true or false"
[[ -n "$MODEL_DIR" && -n "$CALIBRATION_ARCHIVE" && -n "$CALIBRATION_MANIFEST" ]] || die "model and calibration inputs are required"
[[ -n "$QNN_SDK_ROOT" && -n "$WORK_DIR" ]] || die "QNN SDK and WORK_DIR are required"
require_file "$SOURCE_MANIFEST"
require_file "$CALIBRATION_ARCHIVE"
require_file "$CALIBRATION_MANIFEST"
require_file "$QNN_SDK_ROOT/QAIRT_ReleaseNotes.txt"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
command -v clang++ >/dev/null || die "clang++ is required by qnn-model-lib-generator"
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" --help >/dev/null 2>&1 \
  || die "qnn-context-binary-generator cannot start; check the libc++ runtime and LD_LIBRARY_PATH"
grep -a -q 'v2.48.40.260702' \
  "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" \
  || die "QAIRT version mismatch"

"$PYTHON_BIN" - <<'PY' || die "Python environment cannot load the pinned MOSS model"
from transformers.utils import torch_compilable_check  # noqa: F401
PY

"$PYTHON_BIN" "$SCRIPT_DIR/verify_source_manifest.py" "$MODEL_DIR" --manifest "$SOURCE_MANIFEST"
LOGITS_RANGE="$($PYTHON_BIN - "$CALIBRATION_MANIFEST" "$CALIBRATION_ARCHIVE" "$CALIBRATION_LOGITS_MAX_ABS" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
manifest_path, archive_path = Path(sys.argv[1]), Path(sys.argv[2])
record = json.loads(manifest_path.read_text(encoding="utf-8"))
max_abs = float(sys.argv[3]) if sys.argv[3] else float(record["hf_bf16_logits_max_abs"])
if record.get("format") != "meetnote.moss_decoder_calibration.v1":
    raise SystemExit("unsupported calibration manifest")
if record.get("model_revision") != "e8681d68e7042738ffca8ac8212bc8fcb1131ab8":
    raise SystemExit("calibration model revision mismatch")
if record.get("window_count") != 128 or record.get("window_size") != 64:
    raise SystemExit("calibration must contain 128 C64 windows")
if set(record.get("phase_counts", {}).values()) != {32} or not record.get("no_acceptance_overlap"):
    raise SystemExit("calibration phase coverage or no-overlap gate failed")
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if digest != record["file"]["sha256"]:
    raise SystemExit("calibration archive checksum mismatch")
required = max(64.0, max_abs * 1.1)
print(float(2 ** math.ceil(math.log2(required))))
PY
)" || die "calibration validation failed"
echo "Pinned logits range: [-$LOGITS_RANGE, $LOGITS_RANGE]"

if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "MOSS QNN preflight passed; WORK_DIR was not created"
  exit 0
fi
[[ -f "$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" ]] \
  || die "ANDROID_NDK_ROOT must point to an Android NDK"
[[ ! -e "$WORK_DIR" ]] || die "WORK_DIR must not already exist: $WORK_DIR"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/cache" "$WORK_DIR/release"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if [[ "$BUILD_MNN_TOOLS" == true ]]; then
  MNN_ROOT="$MNN_ROOT" "$SCRIPT_DIR/apply_mnn_patches.sh"
  QNN_SDK_ROOT="$QNN_SDK_ROOT" MNN_ROOT="$MNN_ROOT" \
    MNN_QNN_HOST_BUILD_DIR="$MNN_BUILD_DIR" \
    "$SCRIPT_DIR/build_mnn_qnn_host_tools.sh"
fi
for executable in MNNConvert generateIO compilefornpu; do require_exec "$MNN_BUILD_DIR/$executable"; done
export MNN_QNN_VTCM_MB="$VTCM_MB"
export MNN_QNN_HVX_THREADS="$HVX_THREADS"
export MNN_QNN_MERGE_WORKERS="${MNN_QNN_MERGE_WORKERS:-1}"

EXPORT_DIR="$WORK_DIR/exported-model"
echo "[1/5] Exporting W8A16 decoder and FP16 audio graphs"
(
  cd "$MNN_ROOT/transformers/llm/export"
  "$PYTHON_BIN" llmexport.py \
    --path "$MODEL_DIR" --tokenizer_path "$MODEL_DIR" --dst_path "$EXPORT_DIR" \
    --export mnn --mnnconvert "$MNN_BUILD_DIR/MNNConvert" \
    --quant_bit 8 --quant_block 128 --lm_quant_bit 8 --lm_quant_block 128 \
    --seperate_embed --generate_for_npu --act_bit 16 --sym --omni --hqq \
    --omni_epochs 1 --calib_data "$CALIBRATION_ARCHIVE"
) 2>&1 | tee "$WORK_DIR/logs/export.log"
for name in llm.mnn llm.mnn.weight audio_encoder_front.mnn audio_encoder_back.mnn config.json llm_config.json; do
  require_file "$EXPORT_DIR/$name"
done
"$PYTHON_BIN" "$SCRIPT_DIR/widen_mnn_logits.py" \
  --mnn-convert "$MNN_BUILD_DIR/MNNConvert" --model "$EXPORT_DIR/llm.mnn" --range "$LOGITS_RANGE"

GENERATOR="$REPO_ROOT/experiments/ablation/generate_llm_qnn_wide64.py"
CPU_GENERATOR="$REPO_ROOT/experiments/ablation/generate_qnn_with_cpu_ops.py"
COMMON=(--mnn_path "$MNN_BUILD_DIR" --soc_id 87 --dsp_arch v81 --chunk_size 64 --max_history_token 8192 --need_config_json true)
CPU_OPS=(
  --cpu-op /Slice_output_0
  --cpu-op /rotary/Mul_output_0
  --cpu-op /rotary/Cos_output_0
  --cpu-op /rotary/Sin_output_0
)
for layer in $(seq 0 27); do
  CPU_OPS+=(--cpu-op "/layers.$layer/self_attn/FusedAttention")
done

echo "[2/5] Compiling decoder with CPU attention/RoPE and HTP linear/FFN graphs"
cp -al "$EXPORT_DIR" "$WORK_DIR/decoder"
(
  cd "$MNN_ROOT"
  "$PYTHON_BIN" "$CPU_GENERATOR" --base-generator "$GENERATOR" \
    "${CPU_OPS[@]}" \
    --model "$WORK_DIR/decoder" --cache_path "$WORK_DIR/cache/decoder" --model_name llm.mnn "${COMMON[@]}"
) 2>&1 | tee "$WORK_DIR/logs/decoder-qnn.log"
mv "$WORK_DIR/decoder/qnn" "$WORK_DIR/release/decoder"

make_audio_json() {
  local output="$1" input_name="$2" input_shape="$3" output_name="$4"
  "$PYTHON_BIN" - "$output" "$input_name" "$input_shape" "$output_name" <<'PY'
import json, sys
from pathlib import Path
path, name, shape, output = sys.argv[1:]
Path(path).write_text(json.dumps({"configs": [{"inputs": [{"name": name, "shape": [int(x) for x in shape.split(',')]}], "outputs": [output]}]}, indent=2) + "\n", encoding="utf-8")
PY
}
echo "[3/5] Compiling fixed-shape audio contexts"
for graph in audio_encoder_front audio_encoder_back; do cp -al "$EXPORT_DIR" "$WORK_DIR/$graph"; done
make_audio_json "$WORK_DIR/front-io.json" input_features 1,80,3000 hidden_states
make_audio_json "$WORK_DIR/back-io.json" hidden_states 1,1500,1024 audio_embeds
(
  cd "$MNN_ROOT"
  "$PYTHON_BIN" "$SCRIPT_DIR/generate_moss_audio_qnn.py" \
    --base-generator "$MNN_ROOT/transformers/llm/export/npu/generate_llm_qnn.py" \
    --model "$WORK_DIR/audio_encoder_front" --cache_path "$WORK_DIR/cache/audio-front" \
    --mnn_path "$MNN_BUILD_DIR" --soc_id 87 --dsp_arch v81 --model_name audio_encoder_front.mnn \
    --input_json "$WORK_DIR/front-io.json" --external_file audio_encoder_front.mnn.weight
  "$PYTHON_BIN" "$SCRIPT_DIR/generate_moss_audio_qnn.py" \
    --base-generator "$MNN_ROOT/transformers/llm/export/npu/generate_llm_qnn.py" \
    --model "$WORK_DIR/audio_encoder_back" --cache_path "$WORK_DIR/cache/audio-back" \
    --mnn_path "$MNN_BUILD_DIR" --soc_id 87 --dsp_arch v81 --model_name audio_encoder_back.mnn \
    --input_json "$WORK_DIR/back-io.json" --external_file audio_encoder_back.mnn.weight
) 2>&1 | tee "$WORK_DIR/logs/audio-qnn.log"
mv "$WORK_DIR/audio_encoder_front/qnn" "$WORK_DIR/release/audio_encoder_front"
mv "$WORK_DIR/audio_encoder_back/qnn" "$WORK_DIR/release/audio_encoder_back"

echo "[4/5] Assembling runtime payload"
cp -al "$EXPORT_DIR/config.json" "$EXPORT_DIR/llm_config.json" "$WORK_DIR/release/"
for optional in tokenizer.mtok tokenizer.txt embeddings_bf16.bin; do
  [[ ! -f "$EXPORT_DIR/$optional" ]] || cp -al "$EXPORT_DIR/$optional" "$WORK_DIR/release/"
done
"$PYTHON_BIN" - "$WORK_DIR/release/config.json" "$WORK_DIR/release/llm_config.json" <<'PY'
import json, sys
from pathlib import Path
path, llm_path = map(Path, sys.argv[1:])
config = json.loads(path.read_text())
config.update({"llm_model": "decoder/llm.mnn", "audio_front_model": "audio_encoder_front/audio_encoder_front.mnn", "audio_back_model": "audio_encoder_back/audio_encoder_back.mnn", "sampler_type": "greedy", "max_all_tokens": 8192, "max_new_tokens": 4096, "async": False})
config.pop("penalty", None)
path.write_text(json.dumps(config, indent=2) + "\n")
llm_config = json.loads(llm_path.read_text())
llm_config.update({"audio_front_model": "audio_encoder_front/audio_encoder_front.mnn", "audio_back_model": "audio_encoder_back/audio_encoder_back.mnn"})
llm_path.write_text(json.dumps(llm_config, indent=2) + "\n")
PY
MNN_ROOT="$MNN_ROOT" QNN_SDK_ROOT="$QNN_SDK_ROOT" ANDROID_NDK_ROOT="$ANDROID_NDK_ROOT" \
  BUILD_DIR="$WORK_DIR/mnn-android" OUTPUT_DIR="$WORK_DIR/release" \
  "$SCRIPT_DIR/build_moss_android_runner.sh"

echo "[5/5] Writing artifact manifest"
"$PYTHON_BIN" - "$WORK_DIR/release" "$LOGITS_RANGE" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, logits_range = Path(sys.argv[1]), float(sys.argv[2])
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file(): files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
manifest = {"format": "meetnote.moss_qnn_artifacts.v1", "release_id": "moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1", "target": {"soc": "SM8850", "soc_id": 87, "dsp_arch": "v81", "vtcm_mb": 8, "hvx_threads": 8, "qairt_version": "2.48.40.260702"}, "logits_range": [-logits_range, logits_range], "files": files, "status": "server-build-complete-device-validation-required"}
(root / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
echo "MOSS QNN server build complete: $WORK_DIR/release"
echo "Device and quality gates remain required; this output is not production-ready."
