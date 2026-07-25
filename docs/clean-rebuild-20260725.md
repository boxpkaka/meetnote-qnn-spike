# Clean Rebuild, 2026-07-25

## Status

The server rebuild is complete. It is not yet a publishable replacement because
no SM8850 device was connected for the final HTP gate.

## Root Cause Found During Rebuild

The first clean QNN rebuild used MNN's default 8 MiB VTCM graph budget. The
validated release had actually been compiled with 4 MiB, but that input was
missing from both the project patches and release provenance.

The 8 MiB build produced 39 valid contexts totaling 2,132,402,176 bytes. After
pinning 4 MiB, the same pipeline produced 39 contexts totaling 2,141,081,600
bytes, exactly matching the validated release's per-context sizes and total.

## Server Evidence

- Pinned Qwen source hashes passed before conversion.
- The freshly exported `llm.mnn.weight` SHA-256 matched the validated source:
  `b096a22a1d61fff84a77c2277202eeafba3802df941b7eef711f3faf570558db`.
- The wide-logits contract was applied and round-trip validated before QNN
  conversion.
- The rebuilt MNN CPU teacher-forced logits were byte-identical to the retained
  48-step reference. Both `.f32` files have SHA-256
  `e52df0b37e6c51c703684cbedbeffe244e569821941d517c19d385dc9841ce61`.
- All 39 QNN context sizes matched the validated release. Two contexts were
  byte-identical.
- `qnn-context-binary-utility` metadata was equal for 38 of 39 contexts after
  removing generated tensor IDs and ordering tensor records canonically.
- The remaining context, `graph1.bin`, differed only in reported
  `opDataSize` (`316928` retained versus `316416` rebuilt). Its file size was
  unchanged.
- The assembled MNN wrappers were semantically equal after removing their
  generated `mnn_uuid`; their raw hashes differed.

The context byte differences are therefore consistent with generated tensor
ordering and host-tool build output, but server metadata comparison cannot
prove HTP numerical equivalence.

## Remaining Gate

Run the focused and realistic teacher-forced probes on the Vivo V2505A
SM8850/V81 device, retain result JSON, full logits, and logcat, and require:

1. focused fixture top-1 agreement of 25/25;
2. realistic fixture top-1 agreement of 43/48 or better;
3. no context-load failure, DSP SSR, or non-finite logits.

Do not replace the validated manifest or release tag until this device gate
passes.
