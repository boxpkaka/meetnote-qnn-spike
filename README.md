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

## MOSS Transcribe-Diarize PoC

The SM8850 (Snapdragon 8 Elite Gen 5) PoC splits the fixed 30-second Whisper path into two FP16 HTP
graphs and uses a W8A16 Qwen3-0.6B decoder with CPU attention/RoPE and HTP
projection, FFN, and lm_head graphs. Prepare the external
128-window calibration archive with `tools/prepare_moss_calibration.py`, then
run the pinned build outside the repository:

```bash
MODEL_DIR=/path/to/MOSS-Transcribe-Diarize \
CALIBRATION_ARCHIVE=/path/to/calibration.npz \
CALIBRATION_MANIFEST=/path/to/calibration-manifest.json \
QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
ANDROID_NDK_ROOT=/path/to/android-ndk \
WORK_DIR=/large-volume/moss-poc \
tools/rebuild_moss_transcribe_diarize_qnn.sh
```

The host must provide `clang++` and a loadable `libc++.so.1` for the QAIRT host
tools (set `LD_LIBRARY_PATH` when it is not installed system-wide).
`CALIBRATION_LOGITS_MAX_ABS` is optional and otherwise comes from the
calibration manifest.

The runner accepts `config.json input.wav [hotwords]` and emits raw text,
parsed segments, prompt token evidence, stage timings, RTF, and peak PSS as JSON. The checked-in
release record remains explicitly non-production-ready until
`tools/verify_moss_validation.py` accepts component, AliMeeting, and SM8850
device evidence. QAIRT 2.48.40 identifies SM8850 as SoC model `87`, Hexagon
`v81`, with 8 MiB VTCM and 8 HVX threads; the rebuild pins those values
explicitly. Full HTP attention needs 16 MiB for its 8192-token key slice, so
the runnable candidate keeps fused attention on CPU instead.

The first SM8850 device run and its remaining long-sequence quality blocker are
recorded in [`docs/moss-sm8850-device-retrospective-20260819.md`](docs/moss-sm8850-device-retrospective-20260819.md).
Resume device work from
[`docs/moss-sm8850-device-todo.md`](docs/moss-sm8850-device-todo.md); the full promotion worklist is
[`docs/moss-transcribe-diarize-todo.md`](docs/moss-transcribe-diarize-todo.md).

The later full-decoder work is recorded in
[`docs/qnn-genie-decoder-poc.md`](docs/qnn-genie-decoder-poc.md). QAIRT GGUF HTP export now runs the
complete 28-layer decoder with external audio embeddings, c4096, AR1/AR64, and four weight-sharing
shards. A real 118-second Q8 query matched its CPU teacher exactly, but sustained decode throttled
from roughly 14.4 to 8.8 token/s. The 130-window speaker-only quality experiment still beat
FireRed + Sortformer on all eight meetings, while the projected hot audio-plus-decoder RTF remained
1.05 overall and reached 1.53 on the worst window. Consequently this is a post-`stop()` asynchronous
quality-enhancement candidate, not a released real-time path. FireRed + Sortformer remains the
immediate transcript baseline, and any MOSS failure must leave that result intact.

After the build, run the external payload on an attached SM8850 device with:

```bash
tools/run_moss_sm8850.sh /large-volume/moss-poc/release input.wav [hotwords]
```

`tools/verify_moss_audio_mnn.py` performs the intermediate HF FP32 to MNN CPU
audio-graph comparison using the same official `input_features`; it does not
replace the separate log-mel or QNN/HTP gates.

The log-mel gate requires the exact `[N, 80, 3000]` shape, finite values,
cosine similarity at least `0.9999`, and mean absolute error at most `1e-3` on
fixed full 30-second AliMeeting chunks. Maximum absolute error is recorded for
diagnostics but is not a promotion blocker. The actual CPU frontend must also
retain at least `0.995` cosine at the final audio embedding.

Build the small frontend probe against the same patched MNN build, then pass it
to the audio verifier to collect both frontend and graph evidence:

```bash
c++ -std=c++17 tools/moss_fbank_probe.cpp \
  -I "$MNN_ROOT/tools/audio/include" -I "$MNN_ROOT/include" \
  -L "$MNN_BUILD_DIR/tools/audio" -L "$MNN_BUILD_DIR/express" -L "$MNN_BUILD_DIR" \
  -Wl,-rpath,"$MNN_BUILD_DIR/tools/audio:$MNN_BUILD_DIR/express:$MNN_BUILD_DIR" \
  -lMNNAudio -lMNN_Express -lMNN -o "$MNN_BUILD_DIR/moss_fbank_probe"

MNN_ROOT="$MNN_ROOT" MNN_BUILD_DIR="$MNN_BUILD_DIR" \
  OUTPUT="$MNN_BUILD_DIR/moss_mnn_graph_probe" \
  tools/build_moss_mnn_graph_probe.sh

python3 tools/verify_moss_audio_mnn.py \
  --model-dir "$MODEL_DIR" --mnn-dir "$EXPORTED_MODEL_DIR" --wav "$ALIGNMENT_WAV" \
  --mnn-fbank-probe "$MNN_BUILD_DIR/moss_fbank_probe" \
  --mnn-graph-probe "$MNN_BUILD_DIR/moss_mnn_graph_probe"
```

For inputs longer than 30 seconds the verifier checks every 30-second chunk independently and the
concatenated embedding, so a passing first chunk cannot mask later frontend drift.

`tools/hf_teacher_forced_logits.py` also accepts `--audio` and `--reference-result-json` for MOSS.
Build `tools/mnn_teacher_forced_logits.cpp` with `tools/build_mnn_teacher_forced_logits.sh`; its final
optional `moss-wav` argument switches the MNN probe to the multimodal prompt path and reads the
reference file as whitespace-separated token IDs. Always run `compare_token_ids.py` before comparing
the resulting full-vocabulary logits with `compare_teacher_forced_logits.mjs`.
Build the ARM64 diagnostic with `tools/build_moss_android_teacher_probe.sh`, then use
`tools/run_moss_teacher_forced_sm8850.sh` after the normal device regression. It preserves the QNN
logits, checks token IDs, and writes the MNN CPU versus QNN comparison into a new evidence directory.

`tools/verify_moss_audio_token_contract.py` rejects drift between the pinned HF processor and
`llm_config.json`, including audio start/end IDs and the five-second time-marker interval. Full-prompt
reports must also match pinned independent HF token-count and token-ID SHA references, preventing a
shared broken MNN chat template from passing circular comparison.
`tools/prepare_moss_regression_inputs.py` creates SHA-pinned 30/60/90/120-second prefixes from one
acceptance WAV for boundary regression.

Build `tools/moss_tokenizer_probe.cpp` with `tools/build_moss_tokenizer_probe.sh` against the same
patched MNN checkout to compare complete official/runtime prompt IDs before device execution. Set
`MOSS_EVIDENCE_DIR` when invoking `tools/run_moss_sm8850.sh` to preserve the run log, result JSON,
device properties, and SHA-256 records even when the native runner fails.
The rebuild embeds 30/60/90/120/300-second reports under `prompt-contract/`; the device script selects one
automatically by WAV sample count, fails closed when that exact contract is absent or prompt IDs differ, and saves
`prompt-alignment.json`.
`MOSS_PROMPT_CONTRACT` can still override the selected report.

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
