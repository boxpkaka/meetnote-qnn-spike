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

The pinned Qwen3-4B release has a single orchestration entry point. `WORK_DIR`
must be a new directory outside the repository and needs enough space for two
QNN variants plus the assembled release:

```bash
MNN_ROOT=$PWD/third_party/MNN \
MNN_BUILD_DIR=$PWD/third_party/MNN/build_qnn_host \
HF_MODEL_DIR=/path/to/Qwen3-4B \
QNN_SDK_ROOT=/path/to/qairt/2.48.40.260702 \
PYTHON_BIN=/path/to/python-with-mnn-export-dependencies \
WORK_DIR=/large-volume/meetnote-qnn-rebuild \
tools/rebuild_qwen3_4b_release.sh
```

The command verifies the pinned source hashes before exporting. It refuses to
reuse an existing work directory so that stale graph files cannot silently
enter a release. By default it also requires a clean pinned MNN checkout,
applies the project patches, and builds fresh host tools. Set
`BUILD_MNN_TOOLS=false` only when separately testing the model stages with
already verified host binaries.

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

## Publish

1. Upload immutable binary objects first.
2. Verify anonymous HTTPS GET and Range behavior.
3. Upload the manifest last.
4. Record the immutable URL and manifest SHA in the MeetNote Android project.
5. Tag this repository only after the Android consumer has passed its release
   gate.
