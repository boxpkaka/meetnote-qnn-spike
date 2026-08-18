# Release Checklist

This repository publishes metadata and checksums, not model binaries or
proprietary SDK files.

## Reproduce

1. Start from a clean clone on Linux x86_64.
2. Check out MNN at the revision recorded in `provenance.json`.
3. Apply `third_party/patches/mnn` with `tools/apply_mnn_patches.sh`.
4. Build host tools with the recorded QAIRT version.
5. Generate the model from the pinned model revision and fixture checksums.
6. Assemble the CPU-RoPE graphs without `--link-contexts` for publication.
7. Verify every binary with `tools/verify_release_manifest.py`.

The SM8850 release pins the QNN HTP graph VTCM budget to 4 MiB through
`0003-pin-sm8850-vtcm-budget.patch`. This value affects context compilation
and is part of the binary provenance, not a runtime-only tuning knob.

The pinned Qwen3-4B release has a single orchestration entry point. `WORK_DIR`
must be a new directory outside the repository and needs enough space for two
QNN variants plus the assembled release:

```bash
MNN_ROOT=$PWD/third_party/MNN \
MNN_BUILD_DIR=$PWD/third_party/MNN/build_qnn_host \
HF_MODEL_DIR=/path/to/Qwen3-4B \
CALIBRATION_DATA=/path/to/meetnote-omni-wikitext-128.jsonl \
QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
PYTHON_BIN=/path/to/python-with-mnn-export-dependencies \
WORK_DIR=/large-volume/meetnote-qnn-rebuild \
CACHE_ROOT=/fast-ephemeral-volume/meetnote-qnn-cache \
tools/rebuild_qwen3_4b_release.sh
```

The command verifies the pinned source hashes before exporting. It refuses to
reuse an existing work directory so that stale graph files cannot silently
enter a release. By default it also requires a clean pinned MNN checkout,
applies the project patches, and builds fresh host tools. Set
`BUILD_MNN_TOOLS=false` only when separately testing the model stages with
already verified host binaries.

The exported `llm.mnn.weight` must also match the SHA-256 pinned in
`provenance.json`. This gate runs before QNN generation and applies equally to
fresh exports and `EXPORTED_MODEL_SOURCE`; source-model hashes alone are not
proof that quantization produced the validated MNN weights.

`CALIBRATION_DATA` is mandatory. Generate the pinned raw-text JSONL once from
the Wikitext revision recorded in `calibration-manifest.json`:

```bash
python3 tools/prepare_omni_calibration.py \
  --cache-root /path/to/pinned-huggingface-cache \
  --output /data/models/hf/meetnote-omni-wikitext-128.jsonl
```

The preflight checks its byte count and SHA-256. Export never falls back to an
online, moving calibration dataset.

CUDA is not a technical requirement for export. It remains the validated
release baseline because its `llm.mnn.weight` hash is pinned in
`provenance.json`. The MNN patch adds an explicit CPU autocast path for the
mixed BF16/FP32 OmniQuant forward pass and allows a CPU-only export with:

```bash
PROFILE=export-cpu tools/bootstrap_environment.sh
source .venv-export-cpu/bin/activate

EXPORT_DEVICE=cpu EXPORT_ONLY=true \
MNN_ROOT=$PWD/third_party/MNN \
MNN_BUILD_DIR=$PWD/third_party/MNN/build_qnn_host \
HF_MODEL_DIR=/path/to/Qwen3-4B \
CALIBRATION_DATA=/path/to/meetnote-omni-wikitext-128.jsonl \
QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
PYTHON_BIN=$PWD/.venv-export-cpu/bin/python \
WORK_DIR=/large-volume/new-cpu-export \
BUILD_MNN_TOOLS=false \
tools/rebuild_qwen3_4b_release.sh
```

An unqualified CPU export is fenced to `EXPORT_ONLY=true`. The command writes
`logs/export-verification.json` and stops before QNN generation. A CPU artifact
can cross that fence only through the independent candidate workflow below;
merely producing files is not sufficient.

The pinned CPU path was exercised end to end on 2026-08-06 with Python 3.13.7
and Torch 2.8.0+cpu. Export took 45:53 and peaked at about 26.4 GiB RSS. Its
weight SHA-256 was `ecafb34af1f5fe9f837113cc58f5b3807ca26239982546725175d0b8ff413e5c`,
which differs from the CUDA baseline. In the full 255-step teacher-forced
fixture, both exports used the same x86 MNN CPU runner and had 247/255 top-1
agreement, mean logits cosine 0.9712, and mean top-20 overlap 18.20/20. Two
steps were logits outliers despite retaining the same top-1 token. This is not
evidence for bit-exact equivalence; the HF comparison and the remaining gates
below are still mandatory.

After two independent CPU exports produce the hash pinned in
`releases/qwen3-4b-sm8850-v81-c64-rope-cpu-export-v1/candidate.json`, create a
tamper-evident pre-QNN gate from the full CPU/CUDA/HF comparisons:

```bash
python3 tools/verify_cpu_release_candidate.py \
  --candidate releases/qwen3-4b-sm8850-v81-c64-rope-cpu-export-v1/candidate.json \
  --export-record /path/to/cpu-run-1/logs/export-verification.json \
  --export-record /path/to/cpu-run-2/logs/export-verification.json \
  --export-dir /path/to/cpu-run-1/exported-model \
  --export-dir /path/to/cpu-run-2/exported-model \
  --mnn-convert /path/to/MNNConvert \
  --token-ids /path/to/cpu-token-ids.json \
  --token-ids /path/to/cuda-token-ids.json \
  --token-ids /path/to/hf-token-ids.json \
  --cpu-probe-prefix /path/to/cpu-teacher-255step \
  --cuda-probe-prefix /path/to/cuda-teacher-255step \
  --hf-probe-prefix /path/to/hf-teacher-255step \
  --output /path/to/cpu-release-gate.json
```

Only a gate at `qnn-eligible` can authorize the CPU artifact to enter the
baseline/hybrid QNN build. The normal CUDA path is unchanged:

```bash
RELEASE_VARIANT=cpu-candidate EXPORT_DEVICE=cpu EXPORT_ONLY=false \
CPU_CANDIDATE_FILE=$PWD/releases/qwen3-4b-sm8850-v81-c64-rope-cpu-export-v1/candidate.json \
CPU_CANDIDATE_GATE=/path/to/cpu-release-gate.json \
EXPORTED_MODEL_SOURCE=/path/to/verified-cpu-export/exported-model \
tools/rebuild_qwen3_4b_release.sh
```

The gate recomputes all three comparisons from the bound raw `.jsonl` and
`.f32` probes; precomputed comparison summaries are not trusted inputs.

After assembly, rerun `verify_cpu_release_candidate.py` with `--qnn-dir`.
That advances the gate only to `device-validation-required`. A
`production-ready` result additionally requires `--device-validation` and
`--device-runtime-gate` generated by `analyze_qnn_runtime_log.py`. Both records
must be bound to the same release ID, exported weight hash,
assembly-manifest hash, and a clean SM8850/v81 run without crashes, DSP SSR,
non-finite logits, shared-weight mapping failures, or context-size mismatches.

The 2026-08-07 CPU candidate reached `device-validation-required`: two
independent exports had identical deterministic payloads, and the assembled
QNN release contained 39 verified contexts totaling 2,141,081,600 bytes. Its
server-side evidence is recorded in
`releases/qwen3-4b-sm8850-v81-c64-rope-cpu-export-v1/server-validation.json`.
It is intentionally not marked production-ready without the bound SM8850
record. The evidence also retains the step-227 logits outlier instead of
hiding it behind mean metrics.

The reproducible profiles are `PROFILE=host` (Python 3.10, QNN/assembly
tools), `PROFILE=export` (Python 3.13, CUDA packages from
`requirements-export.lock`), and `PROFILE=export-cpu` (Python 3.13 with the
official CPU-only Torch wheel from `requirements-export-cpu.lock`).

QNN generation additionally requires `clang++` and its runtime `libc++` on
`PATH`/`LD_LIBRARY_PATH`. This can be a system LLVM install or a user-owned
toolchain; root access is not required. The preflight executes both QAIRT host
tools and records the resolved compiler/version before it creates or resumes a
QNN stage.

Use `PREFLIGHT_ONLY=true` to validate Python package versions, the model
manifest, the exact QAIRT release and four pinned SDK file hashes, MNN revision
and patch state, free space, and hard-link support before creating `WORK_DIR`.
The default minimum is 32 GiB and can be raised with `MIN_FREE_GIB`.

Use `RESUME=true` after interruption. Each completed stage stores the same
input fingerprint under `WORK_DIR/.stages`; a changed script, patch, SDK tool,
model manifest, or build parameter requires a new work directory. An incomplete
stage clears only its own generated cache/output before restarting.

`EXPORTED_MODEL_SOURCE=/path/to/model` skips the expensive HF-to-MNN export
when resuming from a separately verified export. The rebuild copies it through
hard links and still reapplies and validates the wide-logits contract before
QNN generation.

`WORK_DIR` and `CACHE_ROOT` may use different filesystems. Keep the persistent
models and final release under `WORK_DIR`; point `CACHE_ROOT` at fast
ephemeral storage. The two model variants use hard links for their immutable
exported inputs, so `WORK_DIR` must support normal POSIX hard links.

## Numerical Gates

Record each comparison separately:

1. Hugging Face BF16 to MNN CPU W4A16.
2. MNN CPU W4A16 to QNN/HTP.
3. Free generation to the grounded product pipeline.

Do not combine these deltas into a single quality claim. A successful QNN
execution does not prove token-ranking parity.

Before logits comparison, dump token IDs from both runtimes and require exact
identity:

```bash
python3 tools/hf_teacher_forced_logits.py \
  --model /path/to/Qwen3-4B \
  --prompt fixtures/teacher/prompt.txt \
  --reference fixtures/teacher/reference.txt \
  --validate-only \
  --token-ids-output build/token-ids-hf.json

build/mnn_teacher_forced_logits \
  /path/to/config.json \
  fixtures/teacher/prompt.txt \
  fixtures/teacher/reference.txt \
  build/mnn-tokenizer-probe \
  0 20 4 -1 build/token-ids-mnn.json

python3 tools/compare_token_ids.py \
  --hf build/token-ids-hf.json \
  --mnn build/token-ids-mnn.json
```

The pinned baseline passed this check on 2026-07-25: all 2,079 prompt token
IDs and all 256 reference token IDs were identical. The machine-readable
record is stored with the release metadata in `validation.json`.

## Device Gate

For every publishable build, retain:

- device model, SoC, DSP architecture, QAIRT runtime, and app revision;
- model and manifest hashes;
- result JSON, trace JSONL, and relevant logcat;
- prefill/decode latency, RSS, and native failures;
- grounded long-meeting quality results.

Linux QNN CPU validation cannot replace the final SM8850/HTP test.

Treat shared-weight mapping failures as a promotion failure. Analyze retained
device logs with:

```bash
python3 tools/analyze_qnn_runtime_log.py path/to/logcat.txt
```

`--allow-known-mapping-warning` exists only for isolated experimental runs; it
must not be used for release promotion.

## Publish

1. Upload immutable binary objects first.
2. Verify anonymous HTTPS GET and Range behavior.
3. Upload the manifest last.
4. Record the immutable URL and manifest SHA in the MeetNote Android project.
5. Tag this repository only after the Android consumer has passed its release
   gate.
