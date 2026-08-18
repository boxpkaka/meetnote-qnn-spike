#!/usr/bin/env bash
# Rebuild the pinned SM8850/V81 Qwen3-4B release in a new external workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT/third_party/MNN}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-$MNN_ROOT/build_qnn_host}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_MODEL_DIR="${HF_MODEL_DIR:-}"
CALIBRATION_DATA="${CALIBRATION_DATA:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
WORK_DIR="${WORK_DIR:-}"
CACHE_ROOT="${CACHE_ROOT:-}"
SOC_ID="${SOC_ID:-87}"
DSP_ARCH="${DSP_ARCH:-v81}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
MAX_HISTORY_TOKEN="${MAX_HISTORY_TOKEN:-0}"
BUILD_MNN_TOOLS="${BUILD_MNN_TOOLS:-true}"
EXPORTED_MODEL_SOURCE="${EXPORTED_MODEL_SOURCE:-}"
RESUME="${RESUME:-false}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
EXPORT_ONLY="${EXPORT_ONLY:-false}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"
RELEASE_VARIANT="${RELEASE_VARIANT:-cuda-baseline}"
CPU_CANDIDATE_FILE="${CPU_CANDIDATE_FILE:-}"
CPU_CANDIDATE_GATE="${CPU_CANDIDATE_GATE:-}"
MIN_FREE_GIB="${MIN_FREE_GIB:-32}"

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
[[ -n "$CALIBRATION_DATA" ]] || die "CALIBRATION_DATA is required"
[[ -n "$QNN_SDK_ROOT" ]] || die "QNN_SDK_ROOT is required"
[[ -n "$WORK_DIR" ]] || die "WORK_DIR is required and must name a new directory"
[[ "$RESUME" == "true" || "$RESUME" == "false" ]] || die "RESUME must be true or false"
[[ "$PREFLIGHT_ONLY" == "true" || "$PREFLIGHT_ONLY" == "false" ]] || \
  die "PREFLIGHT_ONLY must be true or false"
[[ "$EXPORT_ONLY" == "true" || "$EXPORT_ONLY" == "false" ]] || \
  die "EXPORT_ONLY must be true or false"
[[ "$EXPORT_DEVICE" == "cuda" || "$EXPORT_DEVICE" == "cpu" ]] || \
  die "EXPORT_DEVICE must be cuda or cpu"
[[ "$RELEASE_VARIANT" == "cuda-baseline" || "$RELEASE_VARIANT" == "cpu-candidate" ]] || \
  die "RELEASE_VARIANT must be cuda-baseline or cpu-candidate"
if [[ "$EXPORT_DEVICE" == "cpu" && "$RELEASE_VARIANT" == "cuda-baseline" ]]; then
  [[ "$EXPORT_ONLY" == "true" ]] || \
    die "EXPORT_DEVICE=cpu is experimental and requires EXPORT_ONLY=true"
  [[ -z "$EXPORTED_MODEL_SOURCE" ]] || \
    die "EXPORT_DEVICE=cpu cannot be combined with EXPORTED_MODEL_SOURCE"
fi
if [[ "$RELEASE_VARIANT" == "cpu-candidate" ]]; then
  [[ "$EXPORT_DEVICE" == "cpu" ]] || \
    die "RELEASE_VARIANT=cpu-candidate requires EXPORT_DEVICE=cpu"
  [[ "$EXPORT_ONLY" == "false" ]] || \
    die "RELEASE_VARIANT=cpu-candidate must run QNN generation"
  [[ -n "$EXPORTED_MODEL_SOURCE" ]] || \
    die "RELEASE_VARIANT=cpu-candidate requires EXPORTED_MODEL_SOURCE"
  require_file "$CPU_CANDIDATE_FILE"
  require_file "$CPU_CANDIDATE_GATE"
fi
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || die "MIN_FREE_GIB must be a non-negative integer"
if [[ "$RESUME" == "true" ]]; then
  [[ -d "$WORK_DIR" ]] || die "RESUME=true requires an existing WORK_DIR: $WORK_DIR"
  if [[ -n "$CACHE_ROOT" ]]; then
    [[ -d "$CACHE_ROOT" ]] || die "RESUME=true requires an existing CACHE_ROOT: $CACHE_ROOT"
  fi
else
  [[ ! -e "$WORK_DIR" ]] || die "WORK_DIR already exists: $WORK_DIR"
  if [[ -n "$CACHE_ROOT" ]]; then
    [[ ! -e "$CACHE_ROOT" ]] || die "CACHE_ROOT already exists: $CACHE_ROOT"
  fi
fi
[[ "$SOC_ID" == "87" ]] || die "the pinned release requires SOC_ID=87"
[[ "$DSP_ARCH" == "v81" ]] || die "the pinned release requires DSP_ARCH=v81"
[[ "$CHUNK_SIZE" == "64" ]] || die "the pinned release requires CHUNK_SIZE=64"
[[ "$MAX_HISTORY_TOKEN" == "0" ]] || \
  die "the pinned release requires MAX_HISTORY_TOKEN=0"

require_file "$MNN_ROOT/transformers/llm/export/llmexport.py"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"
require_exec "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"

EXPECTED_EXPORTED_WEIGHT_SHA256="$($PYTHON_BIN - "$REPO_ROOT" "$RELEASE_VARIANT" \
  "$CPU_CANDIDATE_FILE" "$CPU_CANDIDATE_GATE" <<'PY'
import json
import sys
from pathlib import Path

root, variant, candidate_path, gate_path = sys.argv[1:]
if variant == "cuda-baseline":
    provenance = Path(root) / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu/provenance.json"
    print(json.loads(provenance.read_text(encoding="utf-8"))["export"]["llm_weight_sha256"])
    raise SystemExit

candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
if candidate.get("format") != "meetnote.cpu_export_candidate.v1":
    raise SystemExit("unsupported CPU candidate format")
if gate.get("format") != "meetnote.cpu_release_gate.v1":
    raise SystemExit("unsupported CPU candidate gate format")
if gate.get("candidate") != candidate.get("release_id"):
    raise SystemExit("CPU candidate gate targets a different release")
if gate.get("stage") != "qnn-eligible" or gate.get("production_ready") is not False:
    raise SystemExit("CPU candidate gate must be at the qnn-eligible stage")
expected = candidate["export"]["llm_weight_sha256"]
if gate.get("export_determinism", {}).get("sha256") != expected:
    raise SystemExit("CPU candidate gate and candidate weight hashes differ")
print(expected)
PY
)" || die "failed to validate release weight policy"
[[ "$EXPECTED_EXPORTED_WEIGHT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "release weight policy did not produce a valid SHA-256"
CPU_CANDIDATE_POLICY_SHA256="none"
if [[ "$RELEASE_VARIANT" == "cpu-candidate" ]]; then
  CPU_CANDIDATE_POLICY_SHA256="$(
    sha256sum "$CPU_CANDIDATE_FILE" "$CPU_CANDIDATE_GATE" | sha256sum | cut -d' ' -f1
  )"
fi
HOST_CLANGXX="none"
HOST_CLANGXX_VERSION="none"
if [[ "$EXPORT_ONLY" == "false" ]]; then
  HOST_CLANGXX="$(command -v clang++-9 || command -v clang++ || true)"
  [[ -n "$HOST_CLANGXX" ]] || die "QAIRT host compilation requires clang++ on PATH"
  HOST_CLANGXX_VERSION="$($HOST_CLANGXX --version | head -n 1)"
fi

preflight_build_mode="$BUILD_MNN_TOOLS"
if [[ "$RESUME" == "true" && "$BUILD_MNN_TOOLS" == "true" ]]; then
  preflight_build_mode="resume"
fi
preflight_report="$(mktemp -t meetnote-preflight.XXXXXX.json)"
cleanup_preflight() {
  rm -f "$preflight_report"
}
trap cleanup_preflight EXIT
preflight_args=(
  --qnn-sdk-root "$QNN_SDK_ROOT"
  --hf-model-dir "$HF_MODEL_DIR"
  --calibration-data "$CALIBRATION_DATA"
  --mnn-root "$MNN_ROOT"
  --mnn-build-dir "$MNN_BUILD_DIR"
  --work-dir "$WORK_DIR"
  --build-mnn-tools "$preflight_build_mode"
  --min-free-gib "$MIN_FREE_GIB"
)
if [[ -n "$CACHE_ROOT" ]]; then
  preflight_args+=(--cache-root "$CACHE_ROOT")
fi
if [[ "$EXPORT_ONLY" == "false" ]]; then
  preflight_args+=(--require-qnn-host-build)
fi
if [[ -n "$EXPORTED_MODEL_SOURCE" ]]; then
  preflight_args+=(
    --exported-model-source "$EXPORTED_MODEL_SOURCE"
    --expected-exported-weight-sha256 "$EXPECTED_EXPORTED_WEIGHT_SHA256"
  )
else
  preflight_args+=(--fresh-export --export-device "$EXPORT_DEVICE")
fi
if ! "$PYTHON_BIN" "$SCRIPT_DIR/check_environment.py" "${preflight_args[@]}" >"$preflight_report"; then
  cat "$preflight_report"
  die "preflight failed"
fi
cat "$preflight_report"
if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
  echo "Preflight completed; no build directories were created"
  exit 0
fi

mkdir -p "$WORK_DIR/logs"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
cp "$preflight_report" "$WORK_DIR/logs/environment-preflight.json"
if [[ -z "$CACHE_ROOT" ]]; then
  CACHE_ROOT="$WORK_DIR"
else
  mkdir -p "$CACHE_ROOT"
  CACHE_ROOT="$(cd "$CACHE_ROOT" && pwd)"
fi
EXPORTED_MODEL="$WORK_DIR/exported-model"
BASELINE_MODEL="$WORK_DIR/baseline-model"
HYBRID_MODEL="$WORK_DIR/hybrid-model"
BASELINE_CACHE="$CACHE_ROOT/baseline-cache"
HYBRID_CACHE="$CACHE_ROOT/hybrid-cache"
OUTPUT_QNN="$WORK_DIR/release/qnn"
STAGE_DIR="$WORK_DIR/.stages"
mkdir -p "$STAGE_DIR"

BUILD_FINGERPRINT="$($PYTHON_BIN - "$REPO_ROOT" "$MNN_ROOT" "$QNN_SDK_ROOT" \
  "$SOC_ID" "$DSP_ARCH" "$CHUNK_SIZE" "$MAX_HISTORY_TOKEN" "$EXPORTED_MODEL_SOURCE" \
  "$CALIBRATION_DATA" "$EXPORT_DEVICE" "$EXPORT_ONLY" "$RELEASE_VARIANT" \
  "$EXPECTED_EXPORTED_WEIGHT_SHA256" "$CPU_CANDIDATE_POLICY_SHA256" \
  "$HOST_CLANGXX" "$HOST_CLANGXX_VERSION" <<'PY'
import hashlib
import subprocess
import sys
from pathlib import Path

repo, mnn, qnn = map(Path, sys.argv[1:4])
values = sys.argv[4:]
digest = hashlib.sha256()
for value in values:
    digest.update(value.encode())
    digest.update(b"\0")
paths = [
    repo / "requirements-export.lock",
    repo / "requirements-export-cpu.lock",
    repo / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu/source-manifest.json",
    repo / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu/calibration-manifest.json",
    repo / "tools/rebuild_qwen3_4b_release.sh",
    repo / "tools/check_environment.py",
    repo / "tools/widen_mnn_logits.py",
    repo / "tools/assemble_qnn_cpu_rope_model.py",
    repo / "tools/verify_release_manifest.py",
    repo / "tools/verify_cpu_release_candidate.py",
    repo / "experiments/ablation/generate_llm_qnn_wide64.py",
    repo / "experiments/ablation/generate_qnn_with_cpu_ops.py",
    *sorted((repo / "third_party/patches/mnn").glob("*.patch")),
    qnn / "bin/x86_64-linux-clang/qnn-model-lib-generator",
    qnn / "bin/x86_64-linux-clang/qnn-context-binary-generator",
]
digest.update(
    subprocess.check_output(["git", "-C", str(mnn), "rev-parse", "HEAD"]).strip()
)
for path in paths:
    digest.update(str(path.relative_to(repo) if path.is_relative_to(repo) else path).encode())
    digest.update(b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"

stage_done() {
  local stage="$1"
  shift
  local marker="$STAGE_DIR/$stage.sha256"
  [[ "$RESUME" == "true" && -f "$marker" ]] || return 1
  if ! "$PYTHON_BIN" - "$marker" "$BUILD_FINGERPRINT" "$stage" "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path, expected_fingerprint, stage, *outputs = sys.argv[1:]
try:
    marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError) as exc:
    raise SystemExit(f"invalid stage marker for {stage}; use a new WORK_DIR: {exc}")
if marker.get("format") != "meetnote.rebuild_stage.v1":
    raise SystemExit(f"unsupported stage marker for {stage}; use a new WORK_DIR")
if marker.get("fingerprint") != expected_fingerprint:
    raise SystemExit(f"stage fingerprint changed for {stage}; use a new WORK_DIR")
expected_outputs = marker.get("outputs")
if not isinstance(expected_outputs, dict) or set(expected_outputs) != set(outputs):
    raise SystemExit(f"stage output contract changed for {stage}; use a new WORK_DIR")
for output_name, expected_hash in expected_outputs.items():
    output = Path(output_name)
    if not output.is_file():
        raise SystemExit(f"completed stage {stage} is missing output: {output}")
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"completed stage {stage} output changed: {output}; use a new WORK_DIR"
        )
PY
  then
    die "completed stage verification failed for $stage"
  fi
  return 0
}

mark_stage() {
  local stage="$1"
  shift
  "$PYTHON_BIN" - "$STAGE_DIR/$stage.sha256" "$BUILD_FINGERPRINT" "$stage" "$@" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

marker_path, fingerprint, stage, *outputs = sys.argv[1:]
hashes = {}
for output_name in outputs:
    output = Path(output_name)
    if not output.is_file():
        raise SystemExit(f"cannot complete stage {stage}; output is missing: {output}")
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    hashes[output_name] = digest.hexdigest()
document = {
    "format": "meetnote.rebuild_stage.v1",
    "stage": stage,
    "fingerprint": fingerprint,
    "outputs": hashes,
}
destination = Path(marker_path)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_name, destination)
except BaseException:
    Path(temporary_name).unlink(missing_ok=True)
    raise
PY
}

reset_generated_path() {
  local allowed_root="$1"
  local target="$2"
  "$PYTHON_BIN" - "$allowed_root" "$target" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if target == root or not target.is_relative_to(root):
    raise SystemExit(f"refusing to clear path outside stage root: {target} (root={root})")
if target.is_dir() and not target.is_symlink():
    shutil.rmtree(target)
elif target.exists() or target.is_symlink():
    target.unlink()
PY
}

actual_mnn_revision="$(git -C "$MNN_ROOT" rev-parse HEAD)"
expected_mnn_revision="0bff03cbef43c783f44e41484b9f8a0b28bd758d"
[[ "$actual_mnn_revision" == "$expected_mnn_revision" ]] || \
  die "MNN revision mismatch: expected $expected_mnn_revision, got $actual_mnn_revision"

if [[ "$BUILD_MNN_TOOLS" == "true" ]]; then
  if stage_done mnn-tools "$MNN_BUILD_DIR/MNNConvert" \
    "$MNN_BUILD_DIR/generateIO" "$MNN_BUILD_DIR/compilefornpu"; then
    echo "[resume] Reusing verified MNN host tools"
  else
    MNN_ROOT="$MNN_ROOT" "$SCRIPT_DIR/apply_mnn_patches.sh" \
      2>&1 | tee "$WORK_DIR/logs/apply-mnn-patches.log"
    QNN_SDK_ROOT="$QNN_SDK_ROOT" \
      MNN_ROOT="$MNN_ROOT" \
      MNN_QNN_HOST_BUILD_DIR="$MNN_BUILD_DIR" \
      "$SCRIPT_DIR/build_mnn_qnn_host_tools.sh" \
      2>&1 | tee "$WORK_DIR/logs/build-mnn-tools.log"
    mark_stage mnn-tools "$MNN_BUILD_DIR/MNNConvert" \
      "$MNN_BUILD_DIR/generateIO" "$MNN_BUILD_DIR/compilefornpu"
  fi
elif [[ "$BUILD_MNN_TOOLS" != "false" ]]; then
  die "BUILD_MNN_TOOLS must be true or false"
fi

require_exec "$MNN_BUILD_DIR/MNNConvert"
require_exec "$MNN_BUILD_DIR/generateIO"
require_exec "$MNN_BUILD_DIR/compilefornpu"
grep -Fq '"vtcm_mb": 4' "$MNN_ROOT/source/backend/qnn/npu_convert.py" || \
  die "MNN QNN converter does not contain the pinned 4 MiB VTCM budget"

export QNN_SDK_ROOT
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$MNN_BUILD_DIR:$MNN_BUILD_DIR/express:$MNN_BUILD_DIR/tools/converter:$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0
export MNN_OMNI_EXPORT_DEVICE="$EXPORT_DEVICE"

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

if stage_done export "$EXPORTED_MODEL/config.json" "$EXPORTED_MODEL/llm_config.json" \
  "$EXPORTED_MODEL/llm.mnn" "$EXPORTED_MODEL/llm.mnn.weight" \
  "$WORK_DIR/logs/export-verification.json"; then
  echo "[resume] Reusing verified exported MNN model"
else
  resume_export_payload=false
  if [[ "$RESUME" == "true" && -z "$EXPORTED_MODEL_SOURCE" && \
        -f "$EXPORTED_MODEL/config.json" && -f "$EXPORTED_MODEL/llm_config.json" && \
        -f "$EXPORTED_MODEL/llm.mnn" && -f "$EXPORTED_MODEL/llm.mnn.weight" && \
        -f "$WORK_DIR/logs/export-verification.json" && \
        ( -f "$EXPORTED_MODEL/tokenizer.mtok" || -f "$EXPORTED_MODEL/tokenizer.txt" ) ]]; then
    if "$PYTHON_BIN" - "$WORK_DIR/logs/export-verification.json" \
      "$EXPORTED_MODEL/llm.mnn.weight" "$EXPORT_DEVICE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

record_path, weight_path, expected_device = sys.argv[1:]
record = json.loads(Path(record_path).read_text(encoding="utf-8"))
digest = hashlib.sha256()
with Path(weight_path).open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if record.get("format") != "meetnote.export_verification.v1":
    raise SystemExit("unexpected export verification format")
if record.get("device") != expected_device:
    raise SystemExit("export verification device mismatch")
if record.get("llm_weight_sha256") != digest.hexdigest():
    raise SystemExit("export verification weight mismatch")
PY
    then
      resume_export_payload=true
      echo "[resume] Recovering verified export payload before wide-logits completion"
    fi
  fi
  if [[ "$resume_export_payload" != "true" && "$RESUME" == "true" && \
        -e "$EXPORTED_MODEL" ]]; then
    echo "[resume] Clearing incomplete export stage"
    reset_generated_path "$WORK_DIR" "$EXPORTED_MODEL"
  fi
  if [[ "$resume_export_payload" == "true" ]]; then
    :
  elif [[ -n "$EXPORTED_MODEL_SOURCE" ]]; then
    echo "[1/5] Reusing a separately verified exported MNN model"
    cp -al "$EXPORTED_MODEL_SOURCE" "$EXPORTED_MODEL"
  else
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
        --omni_epochs 1 \
        --calib_data "$CALIBRATION_DATA"
    ) 2>&1 | tee "$WORK_DIR/logs/export.log"
  fi

  for file in config.json llm_config.json llm.mnn llm.mnn.weight; do
    require_file "$EXPORTED_MODEL/$file"
  done
  if [[ ! -f "$EXPORTED_MODEL/tokenizer.mtok" && \
        ! -f "$EXPORTED_MODEL/tokenizer.txt" ]]; then
    die "exported tokenizer is missing"
  fi

  expected_weight_sha256="$EXPECTED_EXPORTED_WEIGHT_SHA256"
  actual_weight_sha256="$(sha256sum "$EXPORTED_MODEL/llm.mnn.weight" | cut -d' ' -f1)"
  baseline_match=false
  if [[ "$actual_weight_sha256" == "$expected_weight_sha256" ]]; then
    baseline_match=true
  fi
  "$PYTHON_BIN" - "$WORK_DIR/logs/export-verification.json" "$EXPORT_DEVICE" \
    "$actual_weight_sha256" "$expected_weight_sha256" "$baseline_match" \
    "$RELEASE_VARIANT" <<'PY'
import json
import sys
from pathlib import Path

output, device, actual, expected, baseline_match, release_variant = sys.argv[1:]
record = {
    "format": "meetnote.export_verification.v1",
    "device": device,
    "llm_weight_sha256": actual,
    "expected_release_weight_sha256": expected,
    "validated_cuda_baseline_sha256": expected if device == "cuda" else None,
    "matches_validated_cuda_baseline": baseline_match == "true" and device == "cuda",
    "release_variant": release_variant,
    "promotion_status": (
        "hash-verified" if baseline_match == "true" and device == "cuda"
        else "cpu-candidate-qnn-eligible"
        if baseline_match == "true" and release_variant == "cpu-candidate"
        else "experimental-export-only"
    ),
}
Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY
  if [[ "$baseline_match" != "true" && \
        ( "$EXPORT_DEVICE" == "cuda" || "$RELEASE_VARIANT" == "cpu-candidate" ) ]]; then
    die "exported weight checksum mismatch: expected=$expected_weight_sha256 actual=$actual_weight_sha256"
  fi

  echo "Applying the verified wide-logits quantization contract"
  "$PYTHON_BIN" "$SCRIPT_DIR/widen_mnn_logits.py" \
    --mnn-convert "$MNN_BUILD_DIR/MNNConvert" \
    --model "$EXPORTED_MODEL/llm.mnn" \
    | tee "$WORK_DIR/logs/wide-logits.json"
  mark_stage export "$EXPORTED_MODEL/config.json" "$EXPORTED_MODEL/llm_config.json" \
    "$EXPORTED_MODEL/llm.mnn" "$EXPORTED_MODEL/llm.mnn.weight" \
    "$WORK_DIR/logs/export-verification.json"
fi

if [[ "$EXPORT_ONLY" == "true" ]]; then
  echo "Export-only validation completed: $EXPORTED_MODEL"
  echo "Verification record: $WORK_DIR/logs/export-verification.json"
  exit 0
fi

echo "[2/5] Creating isolated baseline and hybrid inputs"
if [[ ! -e "$BASELINE_MODEL" ]]; then
  cp -al "$EXPORTED_MODEL" "$BASELINE_MODEL"
fi
if [[ ! -e "$HYBRID_MODEL" ]]; then
  cp -al "$EXPORTED_MODEL" "$HYBRID_MODEL"
fi

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
if stage_done baseline "$BASELINE_MODEL/qnn/llm.mnn" "$BASELINE_MODEL/qnn/graph0.bin"; then
  echo "[resume] Reusing verified wide-logits baseline"
else
  if [[ "$RESUME" == "true" ]]; then
    reset_generated_path "$CACHE_ROOT" "$BASELINE_CACHE"
    reset_generated_path "$WORK_DIR" "$BASELINE_MODEL/qnn"
  fi
  (
    cd "$MNN_ROOT"
    "$PYTHON_BIN" \
      "$REPO_ROOT/experiments/ablation/generate_llm_qnn_wide64.py" \
      "${COMMON_GENERATOR_ARGS[@]}"
  ) 2>&1 | tee "$WORK_DIR/logs/baseline.log"
  mark_stage baseline "$BASELINE_MODEL/qnn/llm.mnn" "$BASELINE_MODEL/qnn/graph0.bin"
fi

COMMON_GENERATOR_ARGS[1]="$HYBRID_MODEL"
COMMON_GENERATOR_ARGS[5]="$HYBRID_CACHE"

echo "[4/5] Generating the CPU phase-multiply and trigonometry prefix"
if stage_done hybrid "$HYBRID_MODEL/qnn/llm.mnn" "$HYBRID_MODEL/qnn/graph0.bin"; then
  echo "[resume] Reusing verified CPU-RoPE hybrid"
else
  if [[ "$RESUME" == "true" ]]; then
    reset_generated_path "$CACHE_ROOT" "$HYBRID_CACHE"
    reset_generated_path "$WORK_DIR" "$HYBRID_MODEL/qnn"
  fi
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
  mark_stage hybrid "$HYBRID_MODEL/qnn/llm.mnn" "$HYBRID_MODEL/qnn/graph0.bin"
fi

echo "[5/5] Assembling and verifying the 39-context release"
if stage_done assembly "$OUTPUT_QNN/assembly-manifest.json" "$OUTPUT_QNN/llm.mnn"; then
  echo "[resume] Reusing verified assembled release"
else
  if [[ "$RESUME" == "true" ]]; then
    reset_generated_path "$WORK_DIR" "$OUTPUT_QNN"
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/assemble_qnn_cpu_rope_model.py" \
    --mnn-convert "$MNN_BUILD_DIR/MNNConvert" \
    --baseline-qnn-dir "$BASELINE_MODEL/qnn" \
    --hybrid-qnn-dir "$HYBRID_MODEL/qnn" \
    --output-qnn-dir "$OUTPUT_QNN"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/verify_release_manifest.py" "$OUTPUT_QNN" \
  | tee "$WORK_DIR/logs/release-verification.json"
mark_stage assembly "$OUTPUT_QNN/assembly-manifest.json" "$OUTPUT_QNN/llm.mnn"

"$PYTHON_BIN" - "$WORK_DIR" "$CACHE_ROOT" "$REPO_ROOT" "$MNN_ROOT" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

work_dir, cache_root, repo_root, mnn_root = map(Path, sys.argv[1:])
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
    "cache_root": str(cache_root),
    "status": "server-build-complete",
}
(work_dir / "rebuild.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

echo "Release rebuild completed: $OUTPUT_QNN"
