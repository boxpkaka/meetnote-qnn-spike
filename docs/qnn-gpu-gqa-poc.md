# MOSS QNN GroupQueryAttention PoC

日期：2026-08-26

## 结论

QAIRT `2.48.40` 内置的 `GroupQueryAttention` 在 SM8850 的 QNN GPU backend 上可以正确执行
packed QKV、GQA、RoPE 和 KV update，但 c4096 的单层 op 时间为 `19.748 ms`，约为现有 MNN
CPU Attention `2.10 ms/layer` 的 9.4 倍，也远高于进入产品集成所需的约
`0.68 ms/layer`。QNN CPU FP32 同一 op 为 `14.525 ms/layer`。两条路线均在单层硬门禁停止。

此外，c10240 graph 在 GPU compose 阶段以 `GPU_ERROR_INVALID_SIZE(10011)` 被拒绝；c4096
可以 compose、finalize 和重复执行。即使不看速度，当前 GPU op 也不能直接承载 MOSS 的
10240 context。

## 实现边界

QNN master op definition 支持：

- packed query shape `[B, S, (num_heads + 2 * kv_heads) * head_dim]`；
- 16 Q heads、8 KV heads、head dim 128；
- FP16 KV cache、预计算 FP16 cos/sin 和 INT32 sequence length；
- `QNN_TENSOR_TYPE_NULL` 表示缺省的独立 key/value 输入。

ONNX custom-op converter 对中间 optional input 和 UINT32 scalar attribute 处理不完整。PoC
只为生成 QNN C++ scaffold 临时修正 scalar dtype，生成后立即恢复 vendor 文件，恢复后的
SHA256 为 `333d4479...`。最终模型把 key/value tensor 标为 `QNN_TENSOR_TYPE_NULL`，并把
converter 错排的 BNSH 维度恢复为 `[1, 8, context, 128]`。

## 真机结果

测试输入使用 Q=K=0、每个 KV head 不同的确定性 V，history 为 4095；因此 attention output
应等于对应 KV head 的 V，且 position 4095 必须写入新 K/V。

| Backend | Context | 单 op 平均 | 单 op 最小 | 正确性 |
| --- | ---: | ---: | ---: | --- |
| QNN GPU FP16 | 4096 | `19.748 ms` | `19.701 ms` | finite，attention max abs `0`，cosine `1.0`，K/V update max abs `0` |
| QNN CPU FP32 | 4096 | `14.525 ms` | `13.523 ms` | graph 可执行；性能门禁已否定，不再扩大验证 |
| QNN GPU FP16 | 10240 | N/A | N/A | compose 失败，`GPU_ERROR_INVALID_SIZE(10011)` |

GPU 的 `QnnGraph_execute` 平均为 `34.581 ms`，其中 op 本身 `19.748 ms`，其余主要是大 KV
graph I/O。产品集成即使使用 in-place shared buffer 消掉这些 I/O，也无法改变 19.748 ms 的
kernel 下界；28 层将达到约 `553 ms/token`，比当前完整 decode 慢一个数量级。

## 决策

1. 不继续构建 c8192/c10240、28 层或 MNN runtime glue。
2. 不把 QNN GPU/CPU `GroupQueryAttention` 作为 HTP grouped-GQA 的替代品。
3. QNN HTP 对该 built-in op 明确不支持；真正的 Qualcomm-native HTP 路线仍需
   Genie/GAIT/QPM 的官方 LLM context recipe。
4. 当前无需外部资产且仍有性能信号的候选，只剩专用 mixed-precision OpenCL Attention：
   保持 INT32 position 和正确 RoPE 路径，只把 QK/softmax/QKV 与 FP16 KV 放到 GPU kernel。
   该方向也必须先做单层 c4096 门禁，目标不高于约 `0.68 ms/layer`。

## Artifacts

- 服务器：`work/qnn-gpu-gqa-poc/`
- 本机：`build/device-validation/20260826-moss-qnn-gpu-gqa-poc/`
- GPU model SHA256：c4096 `28c3dba4...`，c10240 `15206cf7...`
- CPU model SHA256：c4096 `48f7958b...`
- QNN runtime：`v2.48.40.260702151143`
- 手机：Vivo V2547A / SM8850 / Adreno 840，OpenCL driver `0842.27.3`
