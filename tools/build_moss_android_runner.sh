#!/usr/bin/env bash
set -euo pipefail

MNN_ROOT="${MNN_ROOT:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${ANDROID_NDK:-}}"
BUILD_DIR="${BUILD_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -d "$MNN_ROOT" ]] || die "MNN_ROOT is required"
[[ -f "$QNN_SDK_ROOT/include/QNN/QnnTypes.h" ]] || die "QNN_SDK_ROOT is required"
[[ -f "$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" ]] || die "ANDROID_NDK_ROOT is required"
[[ -n "$BUILD_DIR" && -n "$OUTPUT_DIR" ]] || die "BUILD_DIR and OUTPUT_DIR are required"

cmake -S "$MNN_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_shared \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_SEP_BUILD=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_AUDIO=ON \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_WITH_PLUGIN=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_QNN=ON \
  -DMNN_QNN_CONVERT_MODE=OFF \
  -DLLM_SUPPORT_HTTP_RESOURCE=OFF \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT"
jobs="${BUILD_JOBS:-}"
if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then jobs="$(nproc)"; else jobs="$(sysctl -n hw.logicalcpu)"; fi
fi
cmake --build "$BUILD_DIR" --target moss_qnn_runner -j"$jobs"

mkdir -p "$OUTPUT_DIR/lib" "$OUTPUT_DIR/dsp/cdsp"
cp "$BUILD_DIR/moss_qnn_runner" "$OUTPUT_DIR/"
cp "$BUILD_DIR/libMNN.so" "$BUILD_DIR/libllm.so" "$OUTPUT_DIR/lib/"
cp "$BUILD_DIR/libMNN_Express.so" "$OUTPUT_DIR/lib/"
cp "$BUILD_DIR/tools/audio/libMNNAudio.so" "$OUTPUT_DIR/lib/"
cp "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$OUTPUT_DIR/lib/"
cp "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so" "$OUTPUT_DIR/lib/"
cp "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so" "$OUTPUT_DIR/lib/"
cp "$QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "$OUTPUT_DIR/dsp/cdsp/"

libcxx="$(find "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt" -path '*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so' -print -quit)"
[[ -f "$libcxx" ]] || die "cannot locate NDK libc++_shared.so"
cp "$libcxx" "$OUTPUT_DIR/lib/"
file "$OUTPUT_DIR/moss_qnn_runner"
