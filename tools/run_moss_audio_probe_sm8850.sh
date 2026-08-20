#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -eq 7 ]] || {
  echo "Usage: $0 RELEASE DIAGNOSTIC_RUNNER DIAGNOSTIC_LIBLLM INPUT.wav MNN_REFERENCE_DIR OUTPUT_DIR DUMP_PREFIX" >&2
  exit 2
}
RELEASE_DIR="$(cd "$1" && pwd)"
DIAGNOSTIC_RUNNER="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
DIAGNOSTIC_LIBLLM="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
INPUT_WAV="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
MNN_REFERENCE_DIR="$(cd "$5" && pwd)"
OUTPUT_DIR="$6"
DUMP_PREFIX="$7"
ADB_BIN="${ADB:-adb}"

for path in "$DIAGNOSTIC_RUNNER" "$DIAGNOSTIC_LIBLLM" "$INPUT_WAV"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2; exit 1; }
python3 "$SCRIPT_DIR/verify_moss_runtime_payload.py" "$RELEASE_DIR" >/dev/null
[[ "$($ADB_BIN shell getprop ro.product.cpu.abi | tr -d '\r')" == arm64-v8a ]] || exit 1
SOC_MODEL="$($ADB_BIN shell getprop ro.soc.model | tr -d '\r')"
[[ "$(printf '%s' "$SOC_MODEL" | tr '[:lower:]' '[:upper:]')" == *SM8850* ]] || exit 1

read -r MANIFEST_SHA CHUNKS < <(python3 - "$RELEASE_DIR/artifact-manifest.json" "$INPUT_WAV" <<'PY'
import hashlib, sys, wave
with wave.open(sys.argv[2], "rb") as wav:
    chunks = (wav.getnframes() + 30 * wav.getframerate() - 1) // (30 * wav.getframerate())
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()[:16], chunks)
PY
)
DEVICE_DIR="/data/local/tmp/meetnote-moss-sm8850-$MANIFEST_SHA-audio-probe"
mkdir -p "$OUTPUT_DIR"
$ADB_BIN shell "mkdir -p '$DEVICE_DIR'"
$ADB_BIN push "$RELEASE_DIR/." "$DEVICE_DIR/" >/dev/null
$ADB_BIN push "$DIAGNOSTIC_RUNNER" "$DEVICE_DIR/moss_qnn_runner" >/dev/null
$ADB_BIN push "$DIAGNOSTIC_LIBLLM" "$DEVICE_DIR/lib/libllm.so" >/dev/null
$ADB_BIN push "$INPUT_WAV" "$DEVICE_DIR/input.wav" >/dev/null

set +e
$ADB_BIN shell "cd '$DEVICE_DIR' && rm -f '$DUMP_PREFIX'-chunk*-front.f32 '$DUMP_PREFIX'-chunk*-back.f32 && \
  chmod 755 moss_qnn_runner && export LD_LIBRARY_PATH='$DEVICE_DIR/lib' && \
  export ADSP_LIBRARY_PATH='$DEVICE_DIR/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' && \
  export MOSS_AUDIO_DUMP_PREFIX='$DUMP_PREFIX' && ./moss_qnn_runner config.json input.wav" \
  2>&1 | tee "$OUTPUT_DIR/run.log"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$RUN_STATUS" > "$OUTPUT_DIR/process-exit-code.txt"

for ((chunk = 0; chunk < CHUNKS; ++chunk)); do
  $ADB_BIN pull "$DEVICE_DIR/$DUMP_PREFIX-chunk$chunk-front.f32" \
    "$OUTPUT_DIR/chunk$chunk-qnn-front.f32" >/dev/null
  $ADB_BIN pull "$DEVICE_DIR/$DUMP_PREFIX-chunk$chunk-back.f32" \
    "$OUTPUT_DIR/chunk$chunk-qnn-back.f32" >/dev/null
  python3 "$SCRIPT_DIR/compare_moss_audio_tensors.py" \
    --mnn-front "$MNN_REFERENCE_DIR/chunk$chunk-mnn-front.f32" \
    --mnn-back "$MNN_REFERENCE_DIR/chunk$chunk-mnn-back.f32" \
    --qnn-front "$OUTPUT_DIR/chunk$chunk-qnn-front.f32" \
    --qnn-back-chain "$OUTPUT_DIR/chunk$chunk-qnn-back.f32" \
    > "$OUTPUT_DIR/chunk$chunk-alignment.json"
done

python3 - "$OUTPUT_DIR" "$RELEASE_DIR/artifact-manifest.json" "$DIAGNOSTIC_RUNNER" "$DIAGNOSTIC_LIBLLM" "$INPUT_WAV" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = [Path(raw) for raw in sys.argv[2:]] + sorted(path for path in root.iterdir() if path.is_file())
files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
(root / "evidence-sha256.json").write_text(json.dumps(files, indent=2) + "\n")
PY
exit "$RUN_STATUS"
