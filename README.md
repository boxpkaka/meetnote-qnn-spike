# MeetNote QNN Quantization

Reproducible model conversion, QNN/HTP graph generation, numerical alignment,
and release-manifest tooling for MeetNote on-device LLM models.

This repository owns model engineering. The MeetNote Android repository owns
runtime integration, model download/install, remote-process isolation, summary
contracts, quality gates, CPU fallback, and end-to-end product validation.

## Verified Baseline

- Base model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- MNN base: `0bff03cbef43c783f44e41484b9f8a0b28bd758d`
- QAIRT/QNN: `2.48.40.260702`
- Target: Qualcomm `SOC_ID=87`, Hexagon `v81`
- Graph configuration: prefill chunk `64`, max history token `0`
- Quantization: all Linear W4 block 64 with static A16
- Numerical workaround: wide logits plus CPU RoPE phase multiply and Sin/Cos

The baseline is device-specific. It must not be described as a generic
Qualcomm QNN model.

## Repository Layout

```text
docs/                       Root-cause and operating notes
experiments/                Reproducible ablation generators and configs
fixtures/calibration/       Quantization calibration fixtures
fixtures/teacher/           Teacher-forced prompt/reference fixtures
releases/                   Metadata and checksums only, never model binaries
third_party/patches/mnn/    Project-owned patches over the pinned MNN revision
tools/                      Build, conversion, alignment, and assembly tools
```

The existing `sdk/`, `work/`, `ablation/`, `server-teacher/out/`, logs, model
weights, and generated contexts are deliberately ignored. Qualcomm SDK files
and Hugging Face/MNN model artifacts must never be committed.

## Build Flow

1. Check out MNN at the pinned revision under `third_party/MNN`.
2. Apply the patches:

   ```bash
   tools/apply_mnn_patches.sh
   ```

3. Build Linux x86_64 host tools:

   ```bash
   QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
   tools/build_mnn_qnn_host_tools.sh
   ```

4. Generate the baseline C64 QNN graphs:

   ```bash
   QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
   SOC_ID=87 DSP_ARCH=v81 CHUNK_SIZE=64 \
   MODEL_DIR=/path/to/Qwen3-4B-MNN \
   tools/generate_mnn_qnn_artifacts.sh
   ```

5. Run teacher-forced alignment in three stages:
   `HF BF16 -> MNN CPU W4A16 -> QNN/HTP`.
6. Assemble the CPU-RoPE prefix with the verified QNN suffix and publish only
   after every manifest hash and device quality gate passes.

## Required Release Checks

- Model revision, MNN revision, QAIRT version, SoC ID, DSP architecture, and
  quantization parameters are recorded.
- QNN wrapper references contiguous graph indexes.
- Every context binary matches the release manifest SHA-256.
- Teacher-forced number/evidence fixtures meet the recorded parity threshold.
- Long-meeting tests retain the final risk and decision evidence.
- Device validation records logs, result JSON, latency, and memory.

The currently verified assembly metadata is under
`releases/qwen3-4b-sm8850-v81-c64-rope-cpu/`.
