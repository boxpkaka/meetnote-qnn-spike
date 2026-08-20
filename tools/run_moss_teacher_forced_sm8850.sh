#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -eq 8 ]] || {
  echo "Usage: $0 RELEASE INPUT.wav ANDROID_PROBE PROMPT.txt REFERENCE_IDS.txt HF_TOKEN_IDS.json MNN_CPU_PREFIX OUTPUT_DIR" >&2
  exit 2
}

RELEASE_DIR="$(cd "$1" && pwd)"
INPUT_WAV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
ANDROID_PROBE="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
PROMPT="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
REFERENCE_IDS="$(cd "$(dirname "$5")" && pwd)/$(basename "$5")"
HF_TOKEN_IDS="$(cd "$(dirname "$6")" && pwd)/$(basename "$6")"
MNN_CPU_PREFIX="$(cd "$(dirname "$7")" && pwd)/$(basename "$7")"
OUTPUT_DIR="$8"
ADB_BIN="${ADB:-adb}"
TEACHER_STEPS="${TEACHER_STEPS:-48}"
[[ "$TEACHER_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: TEACHER_STEPS must be a positive integer" >&2
  exit 2
}

for path in "$INPUT_WAV" "$ANDROID_PROBE" "$PROMPT" "$REFERENCE_IDS" "$HF_TOKEN_IDS" \
  "$MNN_CPU_PREFIX.jsonl" "$MNN_CPU_PREFIX.f32"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2; exit 1; }
command -v "$ADB_BIN" >/dev/null || { echo "ERROR: adb is required" >&2; exit 1; }
python3 "$SCRIPT_DIR/verify_moss_runtime_payload.py" "$RELEASE_DIR" >/dev/null

[[ "$($ADB_BIN shell getprop ro.product.cpu.abi | tr -d '\r')" == arm64-v8a ]] || {
  echo "ERROR: connected device is not arm64-v8a" >&2
  exit 1
}
SOC_MODEL="$($ADB_BIN shell getprop ro.soc.model | tr -d '\r')"
[[ "$(printf '%s' "$SOC_MODEL" | tr '[:lower:]' '[:upper:]')" == *SM8850* ]] || {
  echo "ERROR: connected device reports '$SOC_MODEL', expected SM8850" >&2
  exit 1
}
MANIFEST_SHA="$(python3 - "$RELEASE_DIR/artifact-manifest.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest()[:16])
PY
)"
DEVICE_DIR="/data/local/tmp/meetnote-moss-sm8850-$MANIFEST_SHA"

mkdir -p "$OUTPUT_DIR"
$ADB_BIN shell "mkdir -p '$DEVICE_DIR'"
$ADB_BIN push "$RELEASE_DIR/." "$DEVICE_DIR/" >/dev/null
$ADB_BIN push "$INPUT_WAV" "$DEVICE_DIR/teacher-input.wav" >/dev/null
$ADB_BIN push "$ANDROID_PROBE" "$DEVICE_DIR/moss_teacher_forced_runner" >/dev/null
$ADB_BIN push "$PROMPT" "$DEVICE_DIR/teacher-prompt.txt" >/dev/null
$ADB_BIN push "$REFERENCE_IDS" "$DEVICE_DIR/teacher-reference-ids.txt" >/dev/null

set +e
$ADB_BIN shell "cd '$DEVICE_DIR' && chmod 755 moss_teacher_forced_runner && \
  export LD_LIBRARY_PATH='$DEVICE_DIR/lib' && \
  export ADSP_LIBRARY_PATH='$DEVICE_DIR/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' && \
  ./moss_teacher_forced_runner config.json teacher-prompt.txt teacher-reference-ids.txt teacher-qnn \
    '$TEACHER_STEPS' 20 4 -1 teacher-token-ids.json teacher-input.wav" \
  2>&1 | tee "$OUTPUT_DIR/run.log"
RUN_STATUS="${PIPESTATUS[0]}"
set -e

for suffix in jsonl f32; do
  $ADB_BIN pull "$DEVICE_DIR/teacher-qnn.$suffix" "$OUTPUT_DIR/teacher-qnn.$suffix" >/dev/null 2>&1 || true
done
$ADB_BIN pull "$DEVICE_DIR/teacher-token-ids.json" "$OUTPUT_DIR/teacher-token-ids.json" >/dev/null 2>&1 || true

if [[ "$RUN_STATUS" -eq 0 ]]; then
  python3 "$SCRIPT_DIR/compare_token_ids.py" \
    --hf "$HF_TOKEN_IDS" --mnn "$OUTPUT_DIR/teacher-token-ids.json" \
    > "$OUTPUT_DIR/token-alignment.json"
  node "$SCRIPT_DIR/compare_teacher_forced_logits.mjs" \
    "$MNN_CPU_PREFIX" "$OUTPUT_DIR/teacher-qnn" "$OUTPUT_DIR/mnn-vs-qnn.json" \
    > "$OUTPUT_DIR/comparison-summary.json"
fi
exit "$RUN_STATUS"
