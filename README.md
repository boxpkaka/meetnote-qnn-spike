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

1. Create the pinned Linux host environment:

   ```bash
   tools/bootstrap_environment.sh
   source .venv/bin/activate
   ```

   The validated release export uses CUDA and a separate Python 3.13
   environment:

   ```bash
   PROFILE=export tools/bootstrap_environment.sh
   source .venv-export/bin/activate
   ```

   A CPU-only export profile is also available for reproducibility and parity
   work. It uses the official PyTorch CPU wheel. An ordinary CPU run is limited
   to `EXPORT_ONLY=true`; only the separately identified `cpu-candidate`
   workflow can enter QNN generation after the determinism and numerical gates
   in `docs/releasing.md` pass:

   ```bash
   PROFILE=export-cpu tools/bootstrap_environment.sh
   source .venv-export-cpu/bin/activate
   EXPORT_DEVICE=cpu EXPORT_ONLY=true \
   tools/rebuild_qwen3_4b_release.sh
   ```

2. Check out MNN at the pinned revision under `third_party/MNN`.
3. Apply the patches:

   ```bash
   tools/apply_mnn_patches.sh
   ```

4. Build Linux x86_64 host tools:

   ```bash
   QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
   tools/build_mnn_qnn_host_tools.sh
   ```

5. Generate the baseline C64 QNN graphs:

   ```bash
   QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
   SOC_ID=87 DSP_ARCH=v81 CHUNK_SIZE=64 \
   MODEL_DIR=/path/to/Qwen3-4B-MNN \
   tools/generate_mnn_qnn_artifacts.sh
   ```

6. Run teacher-forced alignment in three stages:
   `HF BF16 -> MNN CPU W4A16 -> QNN/HTP`.
7. Assemble the CPU-RoPE prefix with the verified QNN suffix and publish only
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

## Repository Checks

Run the checks that do not require model weights or QAIRT:

```bash
python3 tools/check_repository.py
```

To also validate the project patches against a clean pinned MNN checkout:

```bash
python3 tools/check_repository.py --mnn-root third_party/MNN
```

The full promotion procedure and tokenizer parity gate are documented in
[`docs/releasing.md`](docs/releasing.md).

The pinned end-to-end rebuild entry point is:

```bash
tools/rebuild_qwen3_4b_release.sh
```

It requires a local licensed QAIRT installation and an external work directory;
neither SDK files nor generated model binaries are written into Git.

Run only the complete environment and storage preflight without creating a
build directory:

```bash
QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
HF_MODEL_DIR=/path/to/Qwen3-4B \
CALIBRATION_DATA=/path/to/meetnote-omni-wikitext-128.jsonl \
WORK_DIR=/large-volume/new-rebuild \
PREFLIGHT_ONLY=true BUILD_MNN_TOOLS=false \
tools/rebuild_qwen3_4b_release.sh
```

After an interrupted build, use the same inputs with `RESUME=true`. Completed
stages are reused only when their build fingerprint and required outputs match.

Server-side AliMeeting and MeetingBank normalization is documented in
[`docs/server-dataset-preparation.md`](docs/server-dataset-preparation.md).
