# MOSS Qualcomm Genie Decoder PoC

日期：2026-08-25，更新于 2026-08-28

## 结论

QAIRT `2.48.40` 的 GenAITransformer Composer 可以完整识别并转换 MOSS 的 28 层
Qwen3 text decoder；FP32 host runtime 在四个不同 token history 上与 Hugging Face FP16
reference 的 next-token top-1 全部一致。因此 MOSS 的非标准外层
`MossTranscribeDiarizeForConditionalGeneration`、`hidden_size=1024`、16 Q heads、8 KV
heads、`head_dim=128`、per-head Q/K RMSNorm 和 tied lm_head 都不是 Genie 模型表达阻碍。

Genie HTP 接口也明确支持直接输入外部 embeddings，因而现有 audio encoder、VQ adaptor
和 prompt embedding assembly 可以保留。后续已经取得并打通 QAIRT GGUF HTP builder，完成
full-28、c4096、AR1/AR64 weight-sharing、4 shard 和外部 embedding 真机验证。最终结论不是
“HTP 尚待验证”，而是：正确性和单窗执行已通过，长会议持续实时性能未通过；当前只适合作为
保留 FireRed preview 的会后增强候选。

## 模型转换

源模型：`/data/experiments/meetnote-moss/model`

自定义 generic transformer config 的关键参数：

- context `10240`，vocab `151936`，embedding `1024`，feedforward `3072`；
- 28 layers，16 Q heads，8 KV heads，head dim `128`；
- Qwen3 sequential pre-norm、gated SiLU、RMSNorm、RoPE theta `1000000`；
- 权重前缀映射到 `model.language_model.layers.N.*`；
- Q/K per-head RMSNorm 和 tied output weight 均保留。

Composer 在 24.6 秒内完成 FP16 产物，在 13.7 秒内完成 FP32 产物：

| 产物 | 大小 | SHA256 |
|---|---:|---|
| `moss-text-decoder-fp16.bin` | 1,503,284,224 B | `e22af4fc6c1de6cb44c6e4a579aefe447a1a7d1fd877be7ad79e24eba8e804ba` |
| `moss-text-decoder-fp32.bin` | 3,006,548,992 B | `2757bb3454cfea606c06444a34cca884a06be6c3ac3ae1f524e62a8ad907f8c3` |
| FP16 `LUT.bin` | 311,164,928 B | `d4f549609a6129dfa5d8d18bb747a6a77fed5ee4120998bb08849152f70f8ac9` |
| transformer config | 2,173 B | `bdf3ceb124c93ddf70d9a3874d30bda8eacf04989f0723a9cda00c828e9d86fa` |

Composer 写出了预期的 311 个 tensor：每层 11 个 tensor，加 token embedding、final
norm 和 tied output。FP16 LUT 与源 BF16 embedding 转 float 后逐项一致。

## Host 正确性

FP32 GenAITransformer CPU runtime 与 HF FP16 reference 的首个生成 token：

| 输入 token history | HF top-1 | Genie top-1 | 结果 |
|---|---:|---:|---|
| `[14990]` | `1790` | `1790` | 一致 |
| `[151644]` | `6606` | `6606` | 一致 |
| `[151644, 872]` | `3837` | `3837` | 一致 |
| 9-token chat prompt | `9707` (`Hello`) | `9707` | 一致 |

额外的一层 decoder 消融也一致：HF 和 Genie 的 top-1 都是 `34209`。这说明正确性不是
偶然来自最终层或 tokenizer，tensor mapping、layer math 和 lm_head contract 均已打通。

9-token FP32 host query 的 prompt rate 为 `13.57 token/s`，generation rate 为
`6.02 token/s`。这些数据只验证 runtime 可执行，不用于手机 HTP 性能外推。

## 两个必要边界

### FP16 host fallback 不可用

x86 GenAITransformer runtime 明确报告 host 不支持 FP16、回退 FP32，但直接加载 FP16
model-bin 时 logits 放大到数万，并对不同输入退化为 token `100260`。同一模型改用 FP32
model-bin 后立即与 HF 对齐。因此不能用 FP16 host 结果否定模型或 HTP；host correctness
基线必须使用 FP32。

### GenAITransformer CPU 不支持外部 embedding

`GenieDialog_embeddingTokenQuery` 的 JSON 和 C API 都能创建，但 QnnGenAiTransformer CPU
engine 在 query 时明确返回：

```text
qnn-cpu-engine does not support embedding as input
```

这不是 Genie HTP 的限制。SDK 自带 `llava-e2t-htp.json` 使用 QnnHtp、FP32 embedding
input 和 `embeddingTokenQuery`；MOSS 需要复用的正是这条 HTP contract。

## 后续 HTP 实测与最终结论

`tools/build_moss_genie_gguf_htp.py` 在 QAIRT `2.48.40` 上生成了 full-28、c4096、
AR1/AR64 weight-sharing 的 4-shard Q8 external-embedding container。移除 token embedding 后，
4 个 context binary 合计约 `669 MiB`。真实 FP16 prompt embedding 能直接进入 HTP graph；同一
118.19 秒窗口的默认时间戳格式生成 1158 tokens，与 Q8 CPU teacher 逐字节一致，冷机
prefill/decode 为 `484.97/14.10 tok/s`，query 为 `85.41s`。

连续运行暴露了决定性的热态边界：同窗 decode 从 `14.44` 下降到
`12.25/9.75/8.80/8.79 tok/s`，query 从 `83.37s` 上升到 `137.18s`。Q4 context 虽从
约 `669 MiB` 降到 `456 MiB`，热态只提高约 3-5%，不能解决持续吞吐。QNN power profile 也只把
降频推迟一轮，最终仍回到约 `8.8 tok/s`。

为降低输出 token，官方 speaker-only prompt 在 AliMeeting validation 8 场、130 个均衡 120 秒窗
完成 Q8 CPU 质量回归。复用录音期间已有的 FireRed token 时间 anchor 和整场 Sortformer 后，整场
CER/角色 CER 为 `0.22627/0.22646`，对 FireRed + Sortformer 的 `0.32204/0.37406` 为
8/8 逐会双指标胜出；相对默认 MOSS Q8 仅退化 `0.00274/0.00181`。generated tokens 从
`138,483` 降到 `102,923`，减少 25.68%，且 130/130 自然结束。

但是按真机热态 `298 prompt tok/s`、`8.8 decode tok/s`、W16 audio encoder
`RTF=0.2211` 和每窗约 `1.43s` dialog create 投影，audio + decoder 串行的 corpus RTF
仍为 `1.0507`，p95/max 为 `1.4037/1.5300`。最密集的 31 分钟会议停止后仍需约 14.5 分钟；
同密度一小时会议会积压约 25-30 分钟。平均 corpus 实时需要热态约 `9.42 tok/s`，当前最坏窗口
实时需要约 `15.3 tok/s`。

跨进程并发不能补缺口：分别验证 audio-first 和 Genie-first 后，audio 都正常完成，但 Genie 都成功
返回截断的 `"[S"` 和 0 generated token，并伴随 FastRPC/adspmsgd 错误。两个独立 HTP context
不得重叠。其他已拒绝变量包括 audio W8、60 秒切窗、KV INT8、prompt 强制减少 speaker tags 和现有
power profile。

因此当前产品边界是：FireRed + Sortformer 继续提供即时 transcript；MOSS 只能在 `stop()` 后的独立
进程中按 120 秒窗串行做高质量会后增强，失败或超时时保留 preview。它不是已发布的实时链路。若继续
推进发布候选，仍需完整 HTP 代表集、第二目标域和 `10x300s`/一小时稳定性；若要满足所有一小时会议
`RTF <= 1.0`，下一变量必须是输出格式 fine-tune/蒸馏、热态约 15 tok/s 的新 decoder/runtime、
可证明正确的同 runtime 调度或新芯片，而不是继续局部调参。

## Artifacts

- 服务器：`work/moss-genie-transformer-poc/`
- full-28 GGUF/HTP：`work/moss-genie-gguf-htp-poc/`
- speaker-only 质量：
  `work/moss-output-format-poc/speaker-only-balanced120-shards/summary-speaker-only-full-anchor.json`
- 热态排队估算：
  `work/moss-output-format-poc/speaker-only-balanced120-shards/performance-hot-estimate.json`
- 本机小型证据：`build/device-validation/20260825-moss-genie-poc/`
- 本机 full-28 设备证据：`build/device-validation/20260828-moss-genie-full28-c4096/`
- 大型 model-bin 和 LUT 只保存在服务器，hash 记录如上。
