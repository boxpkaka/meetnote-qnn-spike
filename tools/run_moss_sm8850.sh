#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 2 && $# -le 3 ]] || {
  echo "Usage: $0 RELEASE_DIR INPUT.wav [hotwords]" >&2
  exit 2
}
RELEASE_DIR="$(cd "$1" && pwd)"
INPUT_WAV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
HOTWORDS="${3:-}"
DEVICE_DIR=/data/local/tmp/meetnote-moss-sm8850
printf -v HOTWORDS_QUOTED "'%s'" "${HOTWORDS//\'/\'\\\'\'}"

command -v adb >/dev/null || { echo "ERROR: adb is required" >&2; exit 1; }
[[ -x "$RELEASE_DIR/moss_qnn_runner" ]] || { echo "ERROR: Android runner is missing" >&2; exit 1; }
[[ -f "$RELEASE_DIR/artifact-manifest.json" ]] || { echo "ERROR: artifact manifest is missing" >&2; exit 1; }
[[ -f "$INPUT_WAV" ]] || { echo "ERROR: input WAV is missing" >&2; exit 1; }
[[ "$(adb shell getprop ro.product.cpu.abi | tr -d '\r')" == arm64-v8a ]] || {
  echo "ERROR: connected device is not arm64-v8a" >&2
  exit 1
}
SOC_MODEL="$(adb shell getprop ro.soc.model | tr -d '\r')"
[[ "${SOC_MODEL^^}" == *SM8850* ]] || {
  echo "ERROR: connected device reports '$SOC_MODEL', expected SM8850 (Snapdragon 8 Elite Gen 5)" >&2
  exit 1
}

adb shell "mkdir -p '$DEVICE_DIR'"
adb push "$RELEASE_DIR/." "$DEVICE_DIR/"
adb push "$INPUT_WAV" "$DEVICE_DIR/input.wav"
adb shell "cd '$DEVICE_DIR' && chmod 755 moss_qnn_runner && \
  export LD_LIBRARY_PATH='$DEVICE_DIR/lib' && \
  export ADSP_LIBRARY_PATH='$DEVICE_DIR/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' && \
  ./moss_qnn_runner config.json input.wav $HOTWORDS_QUOTED"
