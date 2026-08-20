#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ $# -ge 2 && $# -le 3 ]] || {
  echo "Usage: $0 RELEASE_DIR INPUT.wav [hotwords]" >&2
  exit 2
}
RELEASE_DIR="$(cd "$1" && pwd)"
INPUT_WAV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
HOTWORDS="${3:-}"
EVIDENCE_DIR="${MOSS_EVIDENCE_DIR:-}"
PROMPT_CONTRACT="${MOSS_PROMPT_CONTRACT:-}"
ADB_BIN="${ADB:-adb}"
printf -v HOTWORDS_QUOTED "'%s'" "${HOTWORDS//\'/\'\\\'\'}"

command -v "$ADB_BIN" >/dev/null || { echo "ERROR: adb is required" >&2; exit 1; }
[[ -x "$RELEASE_DIR/moss_qnn_runner" ]] || { echo "ERROR: Android runner is missing" >&2; exit 1; }
[[ -f "$RELEASE_DIR/artifact-manifest.json" ]] || { echo "ERROR: artifact manifest is missing" >&2; exit 1; }
[[ -f "$INPUT_WAV" ]] || { echo "ERROR: input WAV is missing" >&2; exit 1; }
if [[ -z "$PROMPT_CONTRACT" && -d "$RELEASE_DIR/prompt-contract" ]]; then
  SAMPLE_COUNT="$(python3 - "$INPUT_WAV" <<'PY'
import sys, wave
with wave.open(sys.argv[1], "rb") as wav:
    print(wav.getnframes())
PY
)"
  if [[ -f "$RELEASE_DIR/prompt-contract/$SAMPLE_COUNT.json" ]]; then
    PROMPT_CONTRACT="$RELEASE_DIR/prompt-contract/$SAMPLE_COUNT.json"
  fi
fi
[[ -z "$PROMPT_CONTRACT" || -f "$PROMPT_CONTRACT" ]] || {
  echo "ERROR: prompt contract report is missing: $PROMPT_CONTRACT" >&2
  exit 1
}
python3 "$SCRIPT_DIR/verify_moss_runtime_payload.py" "$RELEASE_DIR" >/dev/null
[[ "$($ADB_BIN shell getprop ro.product.cpu.abi | tr -d '\r')" == arm64-v8a ]] || {
  echo "ERROR: connected device is not arm64-v8a" >&2
  exit 1
}
SOC_MODEL="$($ADB_BIN shell getprop ro.soc.model | tr -d '\r')"
DEVICE_MODEL="$($ADB_BIN shell getprop ro.product.model | tr -d '\r')"
ANDROID_VERSION="$($ADB_BIN shell getprop ro.build.version.release | tr -d '\r')"
SOC_MODEL_UPPER="$(printf '%s' "$SOC_MODEL" | tr '[:lower:]' '[:upper:]')"
[[ "$SOC_MODEL_UPPER" == *SM8850* ]] || {
  echo "ERROR: connected device reports '$SOC_MODEL', expected SM8850 (Snapdragon 8 Elite Gen 5)" >&2
  exit 1
}
MANIFEST_SHA="$(python3 - "$RELEASE_DIR/artifact-manifest.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest()[:16])
PY
)"
DEVICE_DIR="/data/local/tmp/meetnote-moss-sm8850-$MANIFEST_SHA"

$ADB_BIN shell "mkdir -p '$DEVICE_DIR'"
$ADB_BIN push "$RELEASE_DIR/." "$DEVICE_DIR/"
$ADB_BIN push "$INPUT_WAV" "$DEVICE_DIR/input.wav"
run_device() {
  $ADB_BIN shell "cd '$DEVICE_DIR' && chmod 755 moss_qnn_runner && \
  export LD_LIBRARY_PATH='$DEVICE_DIR/lib' && \
  export ADSP_LIBRARY_PATH='$DEVICE_DIR/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' && \
  ./moss_qnn_runner config.json input.wav $HOTWORDS_QUOTED"
}

if [[ -z "$EVIDENCE_DIR" ]]; then
  run_device
  exit $?
fi
[[ ! -e "$EVIDENCE_DIR" ]] || { echo "ERROR: evidence directory already exists: $EVIDENCE_DIR" >&2; exit 1; }
mkdir -p "$EVIDENCE_DIR"
set +e
run_device 2>&1 | tee "$EVIDENCE_DIR/run.log"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
python3 "$SCRIPT_DIR/record_moss_run_evidence.py" \
  --release "$RELEASE_DIR" --input-wav "$INPUT_WAV" \
  --run-log "$EVIDENCE_DIR/run.log" --output-dir "$EVIDENCE_DIR" \
  --exit-code "$RUN_STATUS" --device-model "$DEVICE_MODEL" \
  --soc-model "$SOC_MODEL" --android-version "$ANDROID_VERSION"
if [[ -n "$PROMPT_CONTRACT" && -f "$EVIDENCE_DIR/result.json" ]]; then
  set +e
  python3 "$SCRIPT_DIR/verify_moss_device_prompt.py" \
    --result "$EVIDENCE_DIR/result.json" --contract "$PROMPT_CONTRACT" \
    > "$EVIDENCE_DIR/prompt-alignment.json"
  PROMPT_STATUS="$?"
  set -e
  if [[ "$RUN_STATUS" -eq 0 && "$PROMPT_STATUS" -ne 0 ]]; then
    RUN_STATUS="$PROMPT_STATUS"
  fi
  python3 "$SCRIPT_DIR/record_moss_run_evidence.py" \
    --release "$RELEASE_DIR" --input-wav "$INPUT_WAV" \
    --run-log "$EVIDENCE_DIR/run.log" --output-dir "$EVIDENCE_DIR" \
    --exit-code "$RUN_STATUS" --device-model "$DEVICE_MODEL" \
    --soc-model "$SOC_MODEL" --android-version "$ANDROID_VERSION" >/dev/null
fi
exit "$RUN_STATUS"
