# Clean Rebuild, 2026-07-25

## Status

The clean rebuild reproduced the validated SM8850 numerical behavior. The QNN
context binaries are not byte-identical, so the retained release manifest and
tag remain the canonical published artifacts.

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
ordering and host-tool build output. The device results below establish the
required numerical equivalence.

## Device Gate

The rebuilt release was deployed to a separate model directory on a Vivo
V2505A with SM8850, Android 16, and the QAIRT/QNN 2.48 V81 runtime. All 40
wrapper and context hashes were verified on-device before execution.

| Probe | Result |
|---|---:|
| focused top-1 agreement | 25/25 |
| focused logits range | -41.000 to 54.031 |
| realistic top-1 agreement | 47/48 |
| realistic cosine mean / min | 0.9480 / 0.5839 |
| realistic top-20 overlap | 16.58 / 20 |
| sampled peak RSS | 1,056,148 KiB |
| warm realistic instrumentation time | 26.6 seconds |

The 48-step realistic logits were byte-identical across two rebuilt runs and
the retained validated CPU-RoPE run. All three `.f32` files have SHA-256
`8719be74547b22c22b014bd85a9c5b22d3374c33ad0a79daeb259b2432aecdc2`.
There was no process crash, DSP SSR, or non-finite output, and device uptime
remained continuous.

QNN still logged two failed attempts to map the shared weights for context ID
39 (`map result 8003`, `err 1002`) before completing successfully. This is the
same retained runtime-warning class seen before the clean rebuild, not a
numerical regression. It remains tracked separately in issue 8; the QNN path
stays experimental until that memory-mapping boundary is resolved.
