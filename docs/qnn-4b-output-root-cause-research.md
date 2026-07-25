# QNN 4B 输出问题根因调研

日期：2026-07-23 至 2026-07-24

## 结论摘要

MeetNote 当前问题不是单一的“QNN 后端坏了”，而是多个层次叠加，且不同症状的主因不同：

1. **旧的无限重复主因已确认是采样器配置语义。** 旧配置虽写了 `repetition_penalty=1.1`，但 MNN mixed pipeline 未包含 `penalty`，所以该值没有执行。加入 `penalty` 后长循环明显减少，但本轮 QNN 对照仍在尾部出现 `ut-1` 循环，不能把重复问题视为完全结案。
2. **`penalize_prompt_tokens=true` 会放大循环，但不是坏输出的总根因。** MNN 默认惩罚完整 prompt，这与重复 JSON key、数字和 evidence ID 冲突。本轮把它改成 `false` 后，256-token QNN 输出不再触发重复检测，但仍出现游离的 `20, 3, 4`、`"evidence_ids": ["-1...`、全角逗号和数组/对象错误。
3. **同一权重、tokenizer、prompt、采样参数的 CPU/QNN 对照已证明：QNN 转换执行路径是实质性贡献因素。** 根 `llm.mnn` 在 MNN CPU 上生成连贯中文和基本正确的 `utt-*`，主要问题是 schema、幻觉和 256-token 截断；QNN C64 图则出现 `0-1`、`ut-4`、空元素和循环。两次 prompt 的 SHA-256 完全相同。
4. **teacher-forced logits 已确认两个独立的 QNN 误差层。** 聚焦 fixture 中的 `29.96875` 上限精确来自导出模型的 logits uint16 encoding；把最终输出量程单独扩到 `[-64, 64]` 后，25 步 CPU/QNN top-1 从 96% 提升到 100%，`utt-1` 数字冲突消失。长 prompt 的 5/48 个语义 top-1 分叉则完全不变，证明它们来自 lm_head 之前的内部 A16/转图/执行路径。
5. **代表性会议校准不是剩余问题的充分根因。** 用 128 条中文会议、数字、JSON 和约束样本重做 W4A16 后，完整模型在长 prompt 上仍是 43/48 top-1 对齐，QNN target top-1 反而从 44/48 降到 41/48；只移植 A16 encoding 也仍为 43/48。校准改善了部分平均 cosine/RMSE，但没有改善真实生成所依赖的局部 token 排序。
6. **采样参数不是主要修复点。** 按 Qwen non-thinking 官方建议改为 `temperature=0.7, top_p=0.8, top_k=20`，并只惩罚生成 token，QNN 仍输出坏 JSON 和错误 evidence ID。它改善循环概率，但没有恢复结构正确性。
7. **Qwen3-4B 能力和复杂 prompt 是次级但真实的因素。** CPU 同权重对照仍把 `topics` 生成为对象、编造 owner，并在 256 tokens 截断。这说明即便修复 QNN 路径，自由生成完整业务 schema 也不能提供可靠的证据约束。
8. **step 32 的语义翻转已用 tensor injection 定位到 lm_head 之前，且第一层 Q 路径是目前最强的局部贡献点。** 注入 CPU 最终 hidden 后，QNN 从“会议”恢复为 CPU 的“评审”，logits cosine 从 0.6705 升到 0.9480；只注入 layer 0 进入 Attention 前的 CPU Q，也把最终 hidden cosine 从 0.8789 提升到 0.9363、logits cosine 提升到 0.8047。因此 prompt、sampler 和 lm_head 都不是这个分叉的主要起点。
9. **layer 0 `q_proj` 的 W4 block 64 不是剩余误差的充分根因，QNN 的局部 W8 路径本身不可靠。** 只把该权重的 W4 block 从 64 缩到 32 后，layer 0 Q cosine 从 0.9458 变为 0.9440，整体 top-1 反而从 43/48 降到 42/48；变化很小且方向混合。改为 W8 后，K/V 保持 0.9991/0.9994，但 Q cosine 跌到 -0.0173，48-step top-1 只剩 6/48。该 W8 结果在正确的 SM8850 `87/v81` 上复现，且 MNN QNN blockwise 权重代码明确留有 `result is wrong, need to verify`，因此它证明的是当前 W8 QNN 转换路径不可作为精度 reference，而不是“8-bit 模型能力更差”。
10. **服务器 MNN CPU teacher-forced 对齐已证明局部 W8 根模型有效。** 在相同的 2,079-token
    生产 prompt 和 48-step reference 上，W4 与“仅 layer 0 q_proj W8”的 CPU top-1 为
    48/48 一致，target top-1 均为 45/48；logits cosine mean/min 为 0.9794/0.8030。
    因而 QNN W8 的 6/48 灾难性结果来自 QNN graph/backend 路径，而不是 W8 导出模型本身。
11. **QNN 局部 W8 灾难性失真已定位并修复。** MNN 把所有多 block 权重都编码成
    `BLOCKWISE_EXPANSION`，并硬编码 `blockScaleBitwidth=4`；QAIRT HTP 的 LPBQ 只支持
    4-bit 权重扩展到 int8，不能承载原生 W8 block 权重。将 W8 先按原 block scale 还原、
    再按输出通道 requantize 为合法的 `AXIS_SCALE_OFFSET` 后，真机 48-step top-1 从
    6/48 恢复到 43/48，layer 0 Q cosine 从 -0.0173 恢复到 0.9436，均回到 W4 基线水平。
12. **官方 BF16 reference 已把模型能力与 W4 导出从主要嫌疑中分离。** 服务器使用
    `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`，模型 shard
    与 Hugging Face etag 一致；同一长 prompt/reference 上 BF16 与 MNN W4 CPU top-1
    为 47/48。这不代表复杂 JSON 能力没有上限，但证明当前 43/48 的 QNN 差值不是
    “4B 模型本来就会在这些位置选错”。
13. **剩余 W4 QNN 的首个主要差值已定位到 layer 0 的 Q RoPE。** 同一手机、同一 MNN
    runtime、同一 2,079-token prompt 和 step 32，ARM CPU 到 QNN 的 `q_proj_input`、
    `q_norm` cosine 分别为 0.99999995、0.99975918；经过 RoPE 后 `attn_q` 降到
    0.94581558。Q 向量 RMS 基本不变，但低维旋转角错误，说明不是 projection、RMSNorm
    或饱和，而是 RoPE 相位计算。
14. **生成图与角度反演把 RoPE 问题进一步收敛到 HTP FP16 大相位计算。** QNN 图将
    `position_ids × inv_freq` 存为 FP16，再直接调用 `ElementWiseCos/Sin`；位置约 2112
    时，CPU 的前两个旋转角为 0.8497、-0.7432，QNN 却都接近 π/2，频率降低、相位变小后
    才逐步恢复。只把 Sin/Cos 移到 CPU 后，Q cosine 仅从 0.94582 升到 0.96102，并残留
    精确约 1/2 rad 的相位差，证明 HTP 的 phase multiply 也是独立贡献点。
15. **把 RoPE phase multiply 与 Sin/Cos 一起移到 CPU 后，layer 0 Q 已恢复到 norm 边界。**
    同一真机最小图只运行 1.398 秒，`attn_q` cosine 达到 **0.99975932**，phase RMSE 从
    0.35497 降到 **0.0001206**，所有抽检维度的角度误差均小于 0.0003 rad。当前首个主要
    QNN 差值已闭环为 HTP FP16 大相位的 multiply + trig 路径，而非量化、prompt 或采样。
16. **无 debug 完整模型的 48-step 验收确认该修正能恢复最终 token 排序。** 修正 graph
    映射后，QNN 对 MNN CPU 的 top-1 从 43/48 提升到 **47/48**，target top-1 从
    44/48 提升到 **45/48**，logits cosine mean 从 0.8683 提升到 **0.9480**，top-20
    overlap 从 14.00 提升到 **16.58**。唯一剩余分叉的 top-2 margin 只有 0.172，且
    CPU 自己也没有选择 reference token，因此不再支持“RoPE 修正没有传递到最终输出”的假设。

因此，当前最可能的组合根因是：

> **局部 W8 灾难性错误来自 MNN/HTP blockwise encoding 合约不匹配；修复后仍存在的 W4
> CPU/QNN 首个主要差值来自 HTP 的 FP16 大相位 multiply + Sin/Cos RoPE 路径**，再叠加采样器
> 与 prompt 冲突、全 Linear W4 和原版 4B 在复杂自由 JSON 任务上的能力上限。

严格说，最终 logits 输出 encoding 已经从整条路径中单独拆出并修复；代表性业务校准和 layer 0
`q_proj` 的 W4 block size 也已排除为充分解释，W8 blockwise 权重实现已经闭环。无 debug
完整 wrapper 已证明 RoPE 修正能把最终 top-1 恢复到 47/48。尚未拆开的只剩一个低 margin
排序分叉、后续层的残余 A16/HTP 差值，以及独立的 prompt/schema 问题。

## 当前系统的可核验事实

### 模型与转换

- 基础模型：原版 `Qwen/Qwen3-4B`，不是后续 `Qwen3-4B-Instruct-2507`。
- 模型规模：4.0B 参数、36 层、32 个 Q heads、8 个 KV heads。官方原版模型为 BF16，原生上下文 32,768 tokens。[Qwen3-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)
- MeetNote 转换参数见 [`export_args.json`](../local-assets/models/Qwen3-4B-MNN-NPU-C64/export_args.json)：
  - `quant_bit=4`, `quant_block=64`
  - `lm_quant_bit=4`, `lm_quant_block=64`
  - `hqq=true`, `omni=true`
  - `act_bit=16`, `act_sym=false`
  - `omni_epochs=1`
  - `calib_data=null`
- 实际 [`llm.mnn.json`](../local-assets/models/Qwen3-4B-MNN-NPU-C64/llm.mnn.json) 中 253 个 Convolution/Linear 全部为 4-bit，包括输出词表为 151,936 的 `lm_head`；另有 4,114 个静态 `DT_INT16` tensor quant encodings。因此准确描述是 **全 Linear W4 + 静态 A16**，不是“仅部分权重做轻量 4-bit”。
- MNN 的 OmniQuant 实现当 `calib_data` 为空时使用 `Salesforce/wikitext`，默认最多 128 样本、单样本最长 128 tokens；源码见 [`omni_quantizer.py`](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/export/utils/omni_quantizer.py#L13-L44) 和 [校准数据加载](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/export/utils/omni_quantizer.py#L95-L155)。
- OmniQuant 原论文用 128 个样本验证 W4A16 等配置，但论文结论是平均 perplexity/通用 benchmark 的保持，不等价于“中文数字逐字复制和长 JSON 100% 正确”。[OmniQuant 原论文](https://arxiv.org/abs/2308.13137)
- Qualcomm 官方明确说明固定点量化需要有代表性的校准数据；不合适的校准数据可以保持相同延迟却得到很差的准确率。[Qualcomm AI Hub Quantization](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)

### 推理与采样

当前 [`config_qnn.json`](../local-assets/models/Qwen3-4B-MNN-NPU-C64/config_qnn.json) 为：

```json
{
  "sampler_type": "mixed",
  "mixed_samplers": ["penalty", "topK", "topP", "temperature"],
  "temperature": 0.2,
  "top_k": 20,
  "top_p": 0.8,
  "repetition_penalty": 1.1,
  "presence_penalty": 0.0,
  "penalty_window": 0
}
```

关键语义：

- MNN 只会执行 `mixed_samplers` 中显式列出的步骤；`repetition_penalty` 仅仅存在于配置中并不会自动把 `penalty` 加入 pipeline。[MNN sampler pipeline](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/engine/src/sampler.cpp#L124-L169)
- MNN 会把 `penalty` 移到 pipeline 最前，再执行 top-k、top-p 和最终抽样。[MNN pipeline 构建](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/engine/src/sampler.cpp#L182-L215)
- `penalize_prompt_tokens` 默认 `true`。[MNN LlmConfig](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/engine/src/llmconfig.hpp#L484-L493)
- 当它是 `true` 时，penalty 会扫描完整 `history_tokens`；只有显式设为 `false` 才从已生成 tokens 开始。[MNN penalty 实现](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/engine/src/sampler.cpp#L233-L315)
- Qwen 官方对 non-thinking 模式建议 `temperature=0.7, top_p=0.8, top_k=20, min_p=0`；当前 `0.2` 明显更接近贪心，容易把量化后已经偏斜的首选 token 固化。官方也说明 penalty 过高可能导致语言混合和性能下降。[Qwen3-4B Best Practices](https://huggingface.co/Qwen/Qwen3-4B#best-practices)
- MeetNote 已通过 JNI 的 Jinja context 显式设置 `enable_thinking=false`，这点与 non-thinking 用法一致；见 [`meetnote_mnn_llm_jni.cpp`](../samples/meeting-demo/src/main/cpp/meetnote_mnn_llm_jni.cpp)。

### 已观察症状

| 症状 | 本地证据 | 当前解释 |
|---|---|---|
| `utt-1-1-1...`、重复 `0` 并打满 token | [`qnn-text-probe-20260718`](../build/qnn-spike/qnn-text-probe-20260718/) 与旧压力测试 | 旧 mixed pipeline 没有 `penalty`；加入后循环消失，属于已确认采样配置问题 |
| 开启 penalty 后短中文变成 `端 side 会 议...` | `c/d` probe；而 `f` 将 `penalize_prompt_tokens=false` 后恢复正常 | 全 prompt penalty 与复制 prompt 内容冲突，属于当前仍存在的配置风险 |
| `百分之九十七` 多次变成 `17%` 或 `7%` | [`realistic-18-budget-2700`](../build/qnn-spike/meeting-summary-stress-20260721/realistic-18-budget-2700/step-extract-1/raw.txt)、[`realistic-18-penalty-enabled`](../build/qnn-spike/meeting-summary-stress-20260721/realistic-18-penalty-enabled/job/step-extract-1/raw.txt) | 不能由 parser 修复；优先怀疑 W4 `lm_head`/校准、近贪心采样和 4B 复制能力，QNN 数值偏差待对齐 |
| JSON 缺引号、键名变体、数组变对象 | 同上及 [`qnn-run25-c64`](../build/qnn-spike/qnn-run25-c64/job/raw.txt) | 自由采样没有 grammar 约束；复杂 schema、prompt penalty、W4 token ranking 和模型能力共同作用 |
| 编造 mitigation/open question | [`realistic-18-penalty-enabled`](../build/qnn-spike/meeting-summary-stress-20260721/realistic-18-penalty-enabled/job/step-extract-1/raw.txt) | 生成式模型在“补全完整业务对象”时倾向补齐缺失字段；不是 JSON parser 问题 |
| 512 tokens 截断 | 历史 extract/repair artifacts | 输出预算不足是独立失败模式，但不能解释 286 tokens 时已出现的 `17%` 和坏语法 |
| 前台丢失后 QNN 加载停住 | [`mnn-llm-build-assets.md`](mnn-llm-build-assets.md) | vivo 后台冻结影响进度/超时，不改变已经生成 token 的事实正确性；本轮错误在持续前台下仍复现 |

## 2026-07-21 单变量真机对照

输入固定为 `realistic-18.json`，compact prompt 为 2,079 tokens，输出上限 256；四组 prompt 文件 SHA-256 均为 `4ff203ba...9e5c3`。CPU 与 QNN 复用同一个模型目录、`llm.mnn.weight`、`tokenizer.mtok` 和采样配置，唯一执行差异是根 `llm.mnn` 与 `qnn/llm.mnn`。

| 组别 | 唯一变量 | 结果 | 能说明什么 |
|---|---|---|---|
| C0：MNN CPU | 根 `llm.mnn`，同一 W4 权重 | 中文连贯，`utt-0...` 基本正常；`topics` 错为对象、编造 owner，256 tokens 截断 | W4 模型/4B/prompt 本身仍有 schema、幻觉、预算问题 |
| Q0：QNN 基线 | `qnn/llm.mnn` | evidence ID 变成 `0-1`、空项、`ut-4`，尾部重复 `ut-1` 并触发 detector | QNN 编译执行路径相对 CPU 明显退化 |
| Q1：QNN generated-only penalty | `penalize_prompt_tokens=false` | 不再触发循环检测，但出现游离数字、全角标点、坏数组/对象，repair 仍失败 | prompt penalty 是循环放大器，不是坏 JSON/数字的主因 |
| Q2：QNN 官方采样 | Q1 + `temperature=0.7` | 仍有 `-0,1,2...`、游离 `1`、缺失引号和错误 evidence ID | `temperature=0.2` 不是主要根因 |

产物位于 [`qnn-4b-root-cause-20260721`](../build/qnn-spike/qnn-4b-root-cause-20260721/)；其中每组保留 `prompt.txt`、`raw.txt`、`result.json`，Q1/Q2 还保留失败的 repair 输出。

## 2026-07-23 teacher-forced logits 对齐

新增真机探针绕过 sampler：CPU 根图和 QNN 图接收完全相同的 prompt，以及完全相同的逐 token reference 序列。每一步保存完整的 151,936 维 float32 logits，因此结果不再受 temperature、top-k、top-p、penalty 或随机数影响。

### 聚焦数字/JSON fixture

固定 108-token prompt、25 个 decode 位置：

| 指标 | 结果 |
|---|---:|
| CPU/QNN top-1 agreement | 96% |
| logits cosine mean / min | 0.949 / 0.688 |
| top-20 overlap | 17.12 / 20 |
| CPU logits 全局范围 | -40.944 ～ 52.806 |
| QNN logits 全局范围 | -19.891 ～ 29.969 |

在 `evidence_ids` 的 `utt-` 后预测数字 `1` 时出现首个 top-1 分叉：

- CPU：`1=52.534`，`0=30.272`，`2=30.249`，有清晰的 22.26 logit margin。
- QNN：`0/1/2/3` 全部被压到 `29.96875`，四个数字失去排序；按实际数组 argmax，top-1 变成 `0`。

这与历史 `utt-1` 变 `0/-1/ut-1` 的症状形成直接机制闭环：**不是 tokenizer 把 ID 编错，也不是 sampler 主动选择了低概率 token，而是 QNN logits 在采样前已经发生饱和和并列。**

### realistic-18 长 prompt

固定生产同款 2,079-token prompt，并使用 CPU 输出前缀作为 48-token teacher sequence：

| 指标 | 结果 |
|---|---:|
| CPU/QNN top-1 agreement | 89.58%（43/48） |
| logits cosine mean / min | 0.868 / 0.540 |
| top-20 overlap | 14 / 20 |
| QNN 相对 CPU top-1 分叉 | 5 / 48 |

首次分叉发生在 headline 内容的第一个 token：

- CPU：`评审=22.676`、`会议=21.472`，top-1 为 `评审`。
- QNN：`会议=16.609`、`评审=14.797`，top-1 翻转为 `会议`。

后续还观察到 `试点 -> 项目/方案`、`风险 -> 确认/控制`、`控制 -> 措施/JSON 结束符` 等排序翻转。这些位置并非都触及 `29.96875` 上限，说明除最终 logits clipping 外，QNN 图内部的静态量化/数值误差也在改变语义排序。

完整二进制与比较结果位于 [`qnn-4b-logits-align-20260723`](../build/qnn-spike/qnn-4b-logits-align-20260723/)。复现比较命令：

```bash
node tools/compare_teacher_forced_logits.mjs \
  build/qnn-spike/qnn-4b-logits-align-20260723/cpu-realistic \
  build/qnn-spike/qnn-4b-logits-align-20260723/qnn-realistic \
  build/qnn-spike/qnn-4b-logits-align-20260723/comparison-realistic.json
```

### logits 输出量程消融

在 `amphion-119` 上复用同一 C64 模型和同一份 W4 权重，只修改 lm_head 卷积输出、
post-convert 和最终 `logits` 三个 tensor 的静态 encoding：

- 原量程：`scale=0.000761227`、`offset=-26147`，对应约 `[-19.90, 29.98]`；
- 消融量程：`scale=128/65535`、`offset=-32768`，对应约 `[-64.00, 64.00]`；
- `llm.mnn.weight` 的 SHA-256 保持
  `b096a22a1d61fff84a77c2277202eeafba3802df941b7eef711f3faf570558db`。

使用 QAIRT/QNN 2.48、SM8850 `SOC_ID=87`、V81、C64 重新编译 38 个 graph 后，在同一台
Vivo V2505A 上复测：

| 指标 | 原 QNN | logits `[-64,64]` |
|---|---:|---:|
| 聚焦 fixture top-1 agreement | 96%（24/25） | **100%（25/25）** |
| 聚焦 fixture QNN logits 范围 | -19.891 ～ 29.969 | -43.750 ～ 55.594 |
| 聚焦 fixture cosine mean | 0.9486 | 0.9489 |
| realistic-18 top-1 agreement | 89.58%（43/48） | 89.58%（43/48） |
| realistic-18 cosine mean | 0.868317 | 0.868319 |

在原先失败的 `utt-` 后预测 `1` 位置，新图不再产生四个数字并列，QNN top-1 恢复为 `1`。
这证明最终输出 clipping 是 evidence ID 数字错误的一个直接根因。与此同时，长 prompt 的
`评审/会议`、`方案/项目`、`控制/措施` 等 5 个分叉的位置和数值几乎完全不变，证明放宽最终
logits 量程不能修复内部 A16 造成的语义排序误差。

本地产物为
[`Qwen3-4B-MNN-NPU-C64-LOGITS-WIDE64`](../local-assets/models/Qwen3-4B-MNN-NPU-C64-LOGITS-WIDE64/)，
对比结果为
[`comparison-wide64.json`](../build/qnn-spike/qnn-4b-logits-align-20260723/comparison-wide64.json)
和
[`comparison-wide64-realistic.json`](../build/qnn-spike/qnn-4b-logits-align-20260723/comparison-wide64-realistic.json)。
服务器生成目录为
`/home/ubuntu/meetnote-qnn-spike/ablation/Qwen3-4B-MNN-NPU-C64-LOGITS-WIDE64`。

为判断剩余误差是否也是统一的内部 clipping，又生成了一个保持 logits wide64、但将其余
4,111 个 INT16 tensor scale 全部放大 2 倍的 `A16X2` 消融。结果不支持该假设：

| 指标 | logits wide64 | A16X2 + logits wide64 |
|---|---:|---:|
| 聚焦 fixture top-1 agreement | **100%（25/25）** | 96%（24/25） |
| realistic-18 top-1 agreement | **89.58%（43/48）** | 87.50%（42/48） |
| realistic-18 cosine mean | 0.8683 | 0.8841 |
| realistic-18 RMSE mean | 1.7594 | 1.6269 |

统一放宽内部量程虽然改善了平均 cosine/RMSE，却增加了 top-1 分叉，并把聚焦 fixture 的
`":[" -> utt` 翻成 token `0`。这说明生成质量取决于候选 token 的局部 margin，平均向量指标
变好不等于生成更正确；剩余误差也不是简单地把所有内部 scale 放大即可修复。该变体只作为
失败边界保留在服务器
`/home/ubuntu/meetnote-qnn-spike/ablation/Qwen3-4B-MNN-NPU-C64-A16X2-LOGITS-WIDE64`
和本地
[`Qwen3-4B-MNN-NPU-C64-A16X2-LOGITS-WIDE64`](../local-assets/models/Qwen3-4B-MNN-NPU-C64-A16X2-LOGITS-WIDE64/)；
设备已恢复为 logits-wide64。

### 代表性会议校准消融

为直接验证“英文 WikiText、128-token 校准与中文会议分布不匹配”是否是剩余 5/48 分叉的
主因，新增了可复现的 128 条业务校准集：

- 80 条长会议 transcript 窗口；
- 24 条百分比、日期、金额和 utterance ID 样本；
- 16 条 JSON schema 样本；
- 8 条事实约束和输出规则样本。

生成器为
[`generate_qnn_meeting_calibration.mjs`](../tools/generate_qnn_meeting_calibration.mjs)，输出文件
SHA-256 为
`0a6c296d29b94063ff32d3b27b1596818fc624de8068e6b672202044e1edf42c`。
它避开了 realistic-18 的评测区间。MNN OmniQuant 的 custom loader 仍会把每行作为一个 user
样本，并截断到其固定的 128 tokens，因此本实验验证的是**校准内容域**，不是更长 calibration
sequence length。

在 `amphion-119` 上保持 W4 block 64、lm_head W4 block 64、HQQ、OmniQuant 1 epoch 和 A16
不变，得到业务校准 donor。随后做了两个单变量层级：

1. **A16 scale graft**：保留原始 W4 权重，只替换 4,114 个内部 tensor encoding，并保持
   logits wide64。4,104/4,114 个 encoding 发生变化，scale 比值中位数为 0.9885、p90 为
   1.1749、p99 为 1.9391。
2. **完整业务校准模型**：使用 donor 自己的 W4 权重和全部 encoding，再保持 logits wide64。
   donor W4 SHA-256 为
   `64256c3dcde087032cc30d4fe862ad070204a11795a81135229a2abfbba972a7`，
   与原始 W4 的 `b096a22a...` 不同。

真机结果：

| 变体 | 聚焦 top-1 | realistic top-1 | realistic QNN target top-1 | realistic cosine / RMSE |
|---|---:|---:|---:|---:|
| 原始 W4 + logits wide64 | 25/25 | 43/48 | **44/48** | 0.8683 / 1.7594 |
| 原始 W4 + 业务 A16 graft + wide64 | 25/25 | 43/48 | 43/48 | 0.8783 / 1.6006 |
| 完整业务 W4A16 + wide64 | 25/25 | 43/48 | 41/48 | 0.8789 / 1.6573 |

完整业务模型的 CPU/QNN 短 fixture 是 25/25，但长 prompt 的 5 个分叉没有减少，只是换了位置。
其新 CPU 与旧 CPU 在长 fixture 上仍有 46/48 top-1 一致，说明重新量化没有广泛破坏 CPU
能力；退化主要仍出现在 CPU 到 QNN 的边界。与 `A16X2` 结果一致，平均 cosine/RMSE 改善并
不保证局部 top-1 margin 改善。

因此可以排除：

> “默认 WikiText 校准分布不匹配”单独解释剩余 QNN 语义分叉。

不能排除的是少数敏感层/tensor 的 encoding、converter/compiler 对特定 graph 或算子的处理，
以及 HTP kernel 数值误差。设备最终恢复为当前表现最好的**原始 W4 + logits wide64**，其
`graph37.bin` SHA-256 为
`135248cfa9e4ece8de37c19298c9122876c8016932f409d33087da629fb4c20a`。

### layer/graph 边界对齐

QNN 根图把 36 个 transformer block 编译为 `graph0.bin` 到 `graph35.bin`，但 fused
Attention 仍作为 MNN op 留在 graph 之间；`graph36.bin` 是最终 hidden，`graph37.bin` 是
lm_head/logits。这提供了比最终 logits 更窄的对齐边界。

新增真机 layer trace 在同一个 teacher-forced decode step 中记录：

- 每层 QNN graph 输出的 hidden 和 Q/K/V；
- QNN 与 CPU 进入同一个 fused Attention op 的 Q/K/V；
- 每层 Attention 输出和最终 hidden。

探针位于
[`LlmTeacherForcedLogitsDeviceTest.kt`](../samples/meeting-demo/src/androidTest/java/ai/meetnote/demo/llm/LlmTeacherForcedLogitsDeviceTest.kt)，
比较器为
[`compare_layer_traces.mjs`](../tools/compare_layer_traces.mjs)，结果位于
[`qnn-4b-layer-align-20260723`](../build/qnn-spike/qnn-4b-layer-align-20260723/)。

在短 fixture 的 step 2，CPU/QNN 最终 token 排名仍一致，但误差已经逐层累积：

| 边界 | cosine |
|---|---:|
| layer 0 Attention 输出 | 0.9962 |
| layer 1 Attention 输出 | 0.9746 |
| layer 16 Attention 输出 | 0.9222 |
| layer 26 Attention 输出 | 0.8839 |
| 最终 hidden | 0.9132 |

在 realistic-18 首个语义翻转 step 32（CPU 为“评审”，QNN 为“会议”），长上下文把误差明显
放大：

| 层 | Q input | K input | V input | Attention 输出 |
|---:|---:|---:|---:|---:|
| 0 | 0.9458 | 0.9991 | 0.9994 | 0.9563 |
| 1 | 0.9382 | 0.9891 | 0.8926 | 0.8078 |
| 3 | 0.9011 | 0.9258 | 0.8105 | 0.7261 |
| 7 | 0.8820 | 0.8968 | 0.6677 | 0.5792 |
| 9 | 0.8455 | 0.8276 | 0.4796 | 0.4953 |
| 10 | 0.9067 | 0.9294 | 0.3854 | **0.4492** |
| 16 | 0.7606 | 0.7912 | **0.1362** | 0.5035 |
| 32 | 0.8948 | 0.9037 | 0.6883 | 0.6101 |
| 35 | 0.9286 | 0.9580 | 0.7616 | 0.7179 |

最终 hidden cosine 为 0.8789，但该 step 的最终 logits cosine 只有 0.6705，并发生 top-1 翻转。
因为两条路径在 graph 之间执行的是同一个 MNN fused Attention op，这组数据进一步缩小了范围：

1. 差值已经存在于 QNN graph 产生的 Q/K/V，而不是由 sampler 或 QNN Attention kernel 首次引入；
2. layer 0 的 K/V 基本一致，layer 1 开始 V 分支明显恶化，layer 7 到 16 又出现集中放大；
3. Attention 和 lm_head 会继续放大上游 hidden/QKV 的局部误差；
4. 当前数据不能单独把责任归给 `v_proj` kernel，因为 V 也继承了之前 block 的 hidden 误差。

### 单 tensor 因果注入

先在 `amphion-119` 上尝试了 projection CPU fallback。临时扩展 `compilefornpu`，让指定 op
成为 QNN graph break：

- 单独把 `v_proj` 留在 CPU 会产生不可用的内部 NC4 边界；
- 把 layer 1 的 Q/K/V projection 整组留在 CPU，L=64 prefill 可以执行，但第一次 L=1
  decode 没有 logits。当前 MNN 混合图会把 CPU 子图固化为 prefill shape，不能同时服务
  prefill/decode 两种 shape。

这只是当前 graph split 工具的限制，不是 projection 正确性的结论。为绕开它，在
teacher-forced step 32 的 debug callback 中加入单次 float32 tensor injection。CPU 和 QNN
仍使用同一模型、prompt、reference 和 KV 历史，只替换一个指定边界：

| step 32 变体 | QNN top-1 | logits cosine | RMSE | top-20 overlap | 最终 hidden cosine |
|---|---|---:|---:|---:|---:|
| QNN 基线 | 会议 | 0.6705 | 2.0193 | 5 | 0.8789 |
| layer 0 Q input = CPU | 会议 | **0.8047** | **1.5965** | 8 | **0.9363** |
| layer 1 Q input = CPU | 会议 | 0.7545 | 1.7576 | 9 | 0.9230 |
| layer 1 K input = CPU | 会议 | 0.6777 | 1.9917 | 7 | 0.9011 |
| layer 1 V input = CPU | overview | 0.6189 | 2.1576 | 2 | 0.8279 |
| layer 1 Attention output = CPU | 会议 | 0.7613 | 1.7366 | 8 | 0.9299 |
| layer 16 Attention output = CPU | 会议 | 0.6599 | 2.0548 | 5 | 0.8678 |
| final hidden = CPU | **评审** | **0.9480** | **0.8677** | **20** | 1.0000 |

最终 hidden 注入后的实际 trace 与 CPU cosine 为 `0.99999999`、RMSE 为 `0.00154`，证明注入
确实命中目标。它使整体 48-step top-1 从 43/48 变为 44/48，并把首次分叉从 step 32 后移到
step 39。剩余 logits cosine 0.9480 是 graph37/lm_head 自身的残余误差，但它已不足以翻转该
位置的 top-1，因此 **lm_head 不是 step 32 错误的主因**。

最重要的局部证据来自 layer 0：

- K/V input cosine 已为 0.9991/0.9994，Q input 只有 0.9458；
- Q 在 RoPE 输出边界为 0.945816，进入 fused Attention 时为 0.945814，说明两者之间的
  tensor 转换没有新增可见误差；
- 只修正 layer 0 Q，就把 Attention cosine 从 0.9563 提升到 0.9693，最终 hidden 从 0.8789
  提升到 0.9363。

因此当前证据支持 **第一层 Q 分支的 QNN 数值路径是主要早期误差源之一**。当前 trace 位于
RoPE 之后，仍不能继续区分 `q_proj`、Q norm、RoPE、静态 A16 encoding、converter 生成的
MatMul/Convolution、权重解量化或 HTP kernel；这些因素必须通过增加 Q 分支内部 trace，再做
layer 0 `q_proj` 的局部 W8/FP16 变体或独立 QNN/CPU op reference 继续拆分。

V/中层注入变差并不证明 V projection 正确。CPU 中间值来自 CPU hidden，把它单独塞进已经
偏离的 QNN residual/KV 状态会形成不一致的混合状态；这类负向结果只能说明该单点替换不能
独立修复，不能用来给对应 op 免责。完整产物位于
[`qnn-4b-tensor-injection-20260724`](../build/qnn-spike/qnn-4b-tensor-injection-20260724/)。

### layer 0 q_proj 局部权重量化消融

为继续区分 layer 0 Q 误差来自 W4 权重还是 QNN/A16 执行路径，在 `amphion-119` 上做了两个
只修改 `/layers.0/self_attn/q_proj/Linear` 的变体。两者均保持其余 252 个 Linear 权重、
4,111 个非 logits tensor encoding 和 logits wide64 不变，并使用设备实际目标 SM8850
`SOC_ID=87`、V81、C64 编译。

| 变体 | layer 0 Q cosine | layer 0 Attention cosine | final hidden cosine | step 32 logits cosine | 48-step top-1 |
|---|---:|---:|---:|---:|---:|
| W4 block 64 基线 | 0.9458 | 0.9563 | 0.8789 | 0.6705 | 43/48 |
| 仅 q_proj W4 block 32 | 0.9440 | 0.9548 | 0.8916 | 0.6984 | 42/48 |
| 仅 q_proj W8 block 64 | **-0.0173** | 0.1758 | 0.7330 | 0.3891 | 6/48 |

W4 block 32 的平均 logits cosine 从 0.8683 小幅升到 0.8715，RMSE 从 1.7594 降到 1.7374，
但 top-20 overlap 从 14.00 降到 13.69，且多出一个 top-1 分叉。它没有改善最早的 layer 0 Q
边界，因此 **W4 block 64 的分组粒度不是该边界误差的主因**。

W8 的失真只出现在被修改的 Q 分支：同一步 K/V cosine 仍为 0.9991/0.9994，而 Q 直接变成
-0.0173。该结果先被一个误用 `57/v75` 的编译复现，发现目标错误后作废；随后用正确的
`87/v81` 重新编译，得到完全相同的 teacher-forced 输出，排除了 SoC 配置这一混淆因素。
MNN 当前 `QNNConvolution::createWeightAndBias` 的 blockwise expansion 分支也带有
`Todo: result is wrong, need to verify`。所以 W8 结果应解释为 **QNN blockwise W8 路径错误或
未验证**，不能用来衡量 W8 权重本身的模型能力。

同一个 W8 根模型先在服务器 MNN CPU `llm_demo` 上通过自由生成 smoke：简单中文事实 prompt
回答正确；生产同款会议 prompt 的前 64 tokens 能连贯输出正确的 `schema_version`、会议标题
和 headline。随后新增 host 侧
[`mnn_teacher_forced_logits.cpp`](../tools/mnn_teacher_forced_logits.cpp)，用与真机探针相同的
状态转换补齐确定性对齐：

| CPU 对比 | top-1 | target top-1 | cosine mean/min | RMSE mean | top-20 overlap |
|---|---:|---:|---:|---:|---:|
| 服务器 W4 vs 仅 q_proj0 W8 | **48/48** | 45/48 vs 45/48 | 0.9794 / 0.8030 | 0.7022 | 18.54 |
| 真机 ARM CPU W4 vs 服务器 W4 | **48/48** | 45/48 vs 45/48 | 0.7628 / -0.3293 | 2.4267 | 15.44 |
| 真机 ARM CPU W4 vs 服务器 q_proj0 W8 | **48/48** | 45/48 vs 45/48 | 0.7629 / -0.2929 | 2.4237 | 15.60 |

服务器 x86 与真机 ARM 的完整 logits 数值并不逐值相同，因此不能把跨架构 cosine 当作 HTP
精度 reference；但三组 48-step top-1 完全一致，且服务器 W4/W8 同架构差分很小，足以通过
“局部 W8 根模型是否损坏”这一验收门。服务器 W4 首次运行时缺少
`embeddings_bf16.bin`，虽未报 load 失败却输出全 `FLT_MAX`；补入与真机、W8 模型
SHA-256 相同的 embedding `eabe5625...e8c0f8` 后才得到上述有效结果。以后 host parity 必须
把 embedding 哈希纳入前置检查，不能只看进程退出码。

有效产物位于
[`qnn-4b-qproj0-w4-b32-20260724`](../build/qnn-spike/qnn-4b-qproj0-w4-b32-20260724/) 和
[`qnn-4b-qproj0-w8-v81-20260724`](../build/qnn-spike/qnn-4b-qproj0-w8-v81-20260724/)。
W8 测试完成后设备从 USB 总线断开；wide64 基线仍保存在模型目录的
`llm.mnn.wide64-baseline` 与 `graph0.bin.wide64-baseline`。设备重新连接后确认 uptime
连续超过 6 天，排除本次现象为手机重启；它只是 USB 断连。当前模型已恢复 wide64 基线，
`llm.mnn`、`graph0.bin`、`graph37.bin` SHA-256 分别为 `2e336821...13`、
`839595ea...14a`、`135248cf...c20a`，应用也已重新拉到前台。

## 2026-07-24 QNN W8 blockwise 根因闭环

QAIRT 2.48 的 HTP 校验器和 SDK 实现给出了同一条合约：LPBQ 的
`blockScaleBitwidth` 在 HTP 上只能是 4，即 4-bit 权重扩展到 int8。三个单变量实验结果为：

1. 把 MNN 的硬编码从 4 改成 W8 的 `originBits=8`，图校验直接拒绝：
   `incorrect Value 8, expected equal to 4`。
2. 改用通用 `QNN_QUANTIZATION_ENCODING_BLOCK` 后，基础校验通过，但 HTP 无法为该
   Conv2d 准备 op，graph finalize 失败。
3. 对 W8 block 权重按 `value = int8_weight * block_alpha` 还原，再以每个输出通道的
   `maxAbs / 127` requantize，使用 axis 3 的 `AXIS_SCALE_OFFSET`；HTP graph optimize、
   context 生成和真机执行全部通过。`127` 是有符号 int8 的对称正量程，不是经验参数。

该 fallback 会丢失 W8 的精确 per-block scale，代价是一次 W8 block 到 W8 per-channel 的
requantize；但它使用 HTP 明确支持的编码，且相对于继续误用 LPBQ，failure domain 从“数值
完全失真”缩小为可测的量化误差。真机结果如下：

| 变体（CPU reference） | 48-step top-1 | target top-1 | logits cosine | layer 0 Q cosine | layer 0 Attention cosine |
|---|---:|---:|---:|---:|---:|
| W4 block 64 基线 | 43/48 | 44/48 | 0.8683 | 0.9458 | 0.9563 |
| 旧 W8 + 错误 LPBQ | 6/48 | 6/48 | 0.3153 | -0.0173 | 0.1758 |
| 新 W8 per-channel fallback | **43/48** | **44/48** | **0.8715** | **0.9436** | **0.9557** |

新 W8 与 W4 真机输出自身为 47/48 top-1 一致、logits cosine 0.9768；layer 0 Q 的两者
cosine 为 0.9973。这同时闭环了“W8 根模型能力、权重重排还是 QNN encoding”的问题：
W8 根模型和 HWIO 重排有效，错误位于 blockwise encoding 合约。

可复现实验产物位于
[`qnn-4b-qproj0-w8-per-channel-v81-20260724`](../build/qnn-spike/qnn-4b-qproj0-w8-per-channel-v81-20260724/)；
项目补丁由 [`apply_mnn_patches.sh`](../tools/apply_mnn_patches.sh) 幂等重放。服务端生成必须使用
本项目历史产物一致的 `max_history_token=0`；误用 4096 会在 `compilefornpu` 的第二个 shape
转换阶段崩溃，不能把它误判成 W8 encoding 问题。

## 2026-07-24 layer 0 RoPE 边界定位

为排除服务器 x86 与手机 ARM 数学库差异，本轮在同一台 Vivo V2505A 上使用同一份
`libMNN.so` 做 ARM CPU/QNN 对照。输入固定为生产同款 2,079-token prompt，reference
连续 40 个 token 330，trace 固定在 step 32；因此两条路径的 token、RoPE position 和历史
长度完全相同。只加载 layer 0 最小依赖图，单次运行约 1.4 至 1.8 秒，不加载 38 个 HTP
context，也没有触发重启。

| 边界 | cosine | relative L2 | RMSE | max abs |
|---|---:|---:|---:|---:|
| `q_proj_input` | 0.99999995 | 0.049422 | 0.0016335 | 0.0134237 |
| `q_norm` | 0.99975918 | 0.022775 | 0.0334831 | 0.2117395 |
| RoPE 后 `attn_q` | **0.94581558** | **0.328214** | **0.4825364** | **7.3687043** |

`q_proj_input` 的 relative L2 看似不小，是因为该 tensor RMS 只有约 0.033；方向几乎完全
一致。`q_norm` 的 CPU/QNN RMS 分别为 1.4702/1.4609，RoPE 后仍为 1.4702/1.4609，
QNN 也没有碰到约 `[-12, 11.9]` 的 A16 边界。因此这里不是向量被 clipping，而是长度近似
保持、方向被错误旋转。

从每个 head 的 `(dimension i, i+64)` 反解旋转角，可以直接看到误差随相位大小变化：

| RoPE 维度 | ARM CPU 角度 | QNN 角度 |
|---:|---:|---:|
| 0 | 0.849737 | 1.570799 |
| 1 | -0.743218 | 1.570789 |
| 2 | 2.265603 | 2.356224 |
| 4 | -1.712314 | -0.785384 |
| 24 | -0.691371 | -0.693399 |
| 32 | 2.113281 | 2.112575 |

前几个高频维度的相位仍在数百到两千量级，QNN 角度接近 π/2、3π/4、-π/4 等错误值；
相位降低后两条路径重新接近。生成的 QNN C++ 进一步确认：

1. `position_ids` 先 cast 到 FP16；
2. 与 `inv_freq` 相乘后的 `t40` 仍是 FP16；
3. `t40` 未做约减，直接进入 HTP `ElementWiseCos`/`ElementWiseSin`；
4. 角度误差随相位减小而消失，说明问题集中在大相位计算，而非 Q 向量 clipping。

尝试在图内增加 `ElementWiseFmod(t40, 2π)` 时，QAIRT V81 在 context 生成阶段明确报
`OpValidator not found`，所以该路径未上手机。当前最小修正改用 `compilefornpu` 的
`cpu_ops` 分图控制。第一版只将共享的 64×64 `/rotary/Cos_output_0` 和
`/rotary/Sin_output_0` 留在 MNN CPU，其余 projection、norm 和 phase multiply 仍在 HTP。
服务器生成的两个最小 context 为：

- `graph0.bin`：HTP 执行到 FP16 phase，SHA-256 `10f75080...6552`；
- CPU 执行 Sin/Cos；
- `graph1.bin`：HTP 从 cos/sin 接回 RoPE，SHA-256 `e8130629...4ea7`；
- 最小 wrapper：`dbd5ce1e...d202`。

本地产物位于
[`qnn-4b-rope-cpu-trig-20260724`](../build/qnn-spike/qnn-4b-rope-cpu-trig-20260724/)；
真机结果只部分改善：

| 变体 | layer 0 Q cosine | relative L2 | phase RMSE |
|---|---:|---:|---:|
| HTP multiply + HTP trig 基线 | 0.94581558 | 0.328214 | 0.354973 |
| HTP multiply + CPU trig | 0.96101531 | 0.278420 | 0.318499 |
| **CPU multiply + CPU trig** | **0.99975932** | **0.022759** | **0.000121** |

CPU-trig 第一版修复了维度 0、1、4，但维度 2、3、5 分别仍残留约 -1、-2、-1 rad 的误差。
这不是随机量化噪声，而是 HTP phase multiply 输出在大相位上的 1/2 ULP 级差值。将
`/rotary/Mul_output_0` 也留在 CPU 后，这些整数弧度残差全部消失；抽检的维度
0/1/2/3/4/5/8/16/24/32 均恢复到 ARM CPU reference，最大角度误差小于 0.0003 rad。

CPU multiply + trig 最小图真机运行 1.398 秒，没有设备重启。产物位于
[`qnn-4b-rope-cpu-mul-trig-20260724`](../build/qnn-spike/qnn-4b-rope-cpu-mul-trig-20260724/)。
以下命令是确定性的红/绿反馈回路；第一条在阈值 0.99 下返回 1，第二条返回 0：

```bash
node tools/check_rope_trace_parity.mjs \
  build/qnn-spike/qnn-4b-layer0-boundary-20260724/phone/arm-cpu-layer0-rope-token330-step32 \
  build/qnn-spike/qnn-4b-layer0-boundary-20260724/phone/qnn-graph0-token330-step32 \
  0.99

node tools/check_rope_trace_parity.mjs \
  build/qnn-spike/qnn-4b-layer0-boundary-20260724/phone/arm-cpu-layer0-rope-token330-step32 \
  build/qnn-spike/qnn-4b-rope-cpu-mul-trig-20260724/phone/qnn-rope-cpu-mul-trig-token330-step32 \
  0.99 \
  --candidate-qnorm-prefix \
    build/qnn-spike/qnn-4b-layer0-boundary-20260724/phone/qnn-graph0-token330-step32 \
  --candidate-attn-key attn_k:1
```

完整模型第一次使用带 debug 终端输出的临时 hybrid wrapper 时，在 `Session::resize()` 发生
应用进程 SIGSEGV；设备 uptime 连续，手机没有重启。该 wrapper 插入新 graph1 后只替换了
graph0/graph1，却错误地继续让新 graph2 加载旧 graph2；正确映射应是新 graph2..38 分别复用
旧 graph1..37。服务器已逐项比较 `inputs`、`outputs`、`allInputShape` 和全部输出 metadata，
确认 37 组后续 context 除 graph 序号外完全相同，并生成修正 `allGraphName` 的映射 wrapper。

分图工具本身还暴露了两个独立问题：forced CPU binary op 的静态常量不应作为运行时输入；
Attention placeholder 必须按原始 query 输入而不是排序后的第一个输入推导
`[B, L, H×D]`。两点均已修复，并生成无 debug 输出的完整 wrapper、graph0/graph1
context 和后续 context 映射。

最终验收在同一台 Vivo V2505A 上完成。部署时只传新 wrapper、graph0、graph1，并在设备内
将原 graph1..37 顺延为 graph2..38；部署后首尾 context 哈希与服务器映射逐项一致。固定
2,079-token 真实会议 prompt、48 个 teacher-forced reference token，结果如下：

| 变体 | top-1 agreement | target top-1 | cosine mean/min | RMSE mean | top-20 overlap |
|---|---:|---:|---:|---:|---:|
| QNN wide64 基线 | 43/48 | 44/48 | 0.8683 / 0.5403 | 1.7594 | 14.00 |
| **CPU phase multiply + trig** | **47/48** | **45/48** | **0.9480 / 0.5839** | **1.0689** | **16.58** |

基线在 step 32、39、40、42、43 分叉；修正后只剩 step 39。该位置 CPU 的 top-3 是
`，` 22.360、`方案` 22.239、`项目` 21.944，修正 QNN 的 top-3 是 `项目` 23.125、
`，` 22.953、`方案` 21.406。修正 QNN 的 top-1/top-2 margin 只有 0.172，top-20 与
CPU 重合 18/20；这更像后续 A16/HTP 小残差放大了本来就接近的排序，而不是仍有一个与
step 32 同量级的 RoPE 相位错误。

本次 instrumentation 用时 369.2 秒，应用进程未崩溃，设备 uptime 连续超过 6 天。测试后
已逆向还原 graph 映射，并用 wrapper、graph0、graph1、graph37 的 SHA-256 确认手机恢复
wide64 基线；应用重新拉到前台。测试过程中主进程一度无 CPU 活动，再次拉前台后约 15 秒
完成，说明前台状态会影响验收时延，但不会改变上述 teacher-forced 数值结论。

生产资产不再手工改 graph 文件名。仓库的
[`assemble_qnn_cpu_rope_model.py`](../tools/assemble_qnn_cpu_rope_model.py) 会用
`MNNConvert` 读取 baseline 与 hybrid wrapper，逐图比较 `inputs`、`outputs`、
`allInputShape` 和全部 `o_*` 输出 metadata，只在合约一致时替换 `allGraphName`，然后组装
新 graph0/1 与 baseline graph1..37。服务器真实产物验证命令为：

```bash
python3 tools/assemble_qnn_cpu_rope_model.py \
  --mnn-convert work/MNN/build_qnn_host_llm/MNNConvert \
  --baseline-qnn-dir \
    ablation/Qwen3-4B-MNN-NPU-C64-LOGITS-WIDE64/qnn \
  --hybrid-qnn-dir \
    ablation/cache-rope-cpu-mul-trig-nodebug/qnn \
  --output-qnn-dir \
    ablation/assembled-rope-cpu-mul-trig/qnn \
  --link-contexts
```

组装结果为 39 个 graph，wrapper SHA-256 为
`71052b83ecb988c9db2a62d3b310b7bf83c1aae32974cf2d4c8526231ccf185f`，与本次真机
48-step 验收使用的 wrapper 逐字节一致。`--link-contexts` 只用于同文件系统服务器节省空间；
发布目录应省略该参数生成独立副本。

### 修复模型的产品链路验收

修复版已作为 `cpu-rope-v1` 部署到 Vivo V2505A 并保留，不再恢复旧 wide64 基线。设备上的
wrapper、graph0、graph1、graph38 SHA-256 分别为 `71052b83...185f`、
`487163f2...1773`、`73d97778...3d38`、`135248cf...20a`，共 39 个 graph。

验收首先刻意运行旧 APK 的自由 JSON 路径，结果在第 1/23 段失败：首轮和 repair 都打满
512 tokens 后截断。首轮已经正确保留“百分之九十七”，没有复现旧 QNN 的 `17%/7%`，但
repair 仍产生坏引号。单次 compact prompt 也在 545 tokens 主动结束时留下坏 JSON。这个反例
确认了两个边界：

1. CPU RoPE 修复恢复 logits 排序，但不能为自由生成增加 JSON grammar；
2. 4B 模型不应承担完整结构化纪要的唯一事实和语法边界。

随后用当前 grounded pipeline 重新构建并覆盖安装 APK。该路径确定性抽取证据和组装结构，
QNN 只用最多 128 tokens 生成 headline/overview envelope；模型失败或格式不完整时回退到
确定性结果。最终提交范围从干净 worktree 独立构建，JVM 单测、debug APK 和 instrumentation
APK 共 102 个 Gradle 任务通过。相同 `long-96.json`（96 句话、约 45 分钟会议）真机通过：

- instrumentation：`OK (1 test)`；
- QNN 模型时间：27.260 秒，其中 prefill 17.667 秒、decode 9.572 秒；
- prompt/decode：2,296 / 128 tokens；
- RSS：503,500 KB；
- JSON 必需字段完整，27 个 evidence ID 全部合法；
- 保留“百分之九十七”、`utt-94`、`utt-95` 和末尾决议；
- 应用进程未崩溃，设备 uptime 连续超过 6 天 15 小时。

instrumentation 墙钟为 73.712 秒，其中包含前台被微信抢走后的 Vivo 冻结等待，不能作为纯模型
延迟。重新拉起 MeetNote 后测试继续完成。真机产物保存在
`build/qnn-spike/qnn-4b-rope-cpu-mul-trig-20260724/e2e-long96-release-v3/`。

当前手机已完成修复模型和 grounded APK 的组合验收。39-graph 资产也已发布到独立的不可变前缀：

- manifest：
  `https://meetnote.tos-cn-guangzhou.volces.com/qnn/v81/c64-cpu-rope-2026-07-24/manifest.tsv`；
- manifest SHA-256：
  `af02ce83916424c80a7927e9c82bb889612affac27330d12e3adeaacbc20203f`；
- 48 个对象共 5,087,437,969 bytes，其中 39 个 graph；wrapper SHA-256 为
  `71052b83ecb988c9db2a62d3b310b7bf83c1aae32974cf2d4c8526231ccf185f`。

发布采用新前缀而非覆盖旧 manifest，因此已发布 APK 仍可固定旧 SHA 回滚。匿名 HTTPS GET 和 Range
均已验证；manifest 最后上传，固定 5 分钟缓存，版本对象使用 immutable 长缓存。后续 QNN 构建应固定
上述 URL 与 SHA，不再使用旧的 38-graph `c64-2026-07-12` manifest。

## 各候选根因判断

### 1. 采样器与 prompt 的交互：已证实影响循环，但不是总根因

旧的无限重复已经被单变量实验定位到 `penalty` 未进入 mixed pipeline。把 `penalty` 加进去是正确方向；本轮又确认 `"penalize_prompt_tokens": false` 能避免长 prompt 下的尾部循环，因此产品配置应保留 generated-only penalty。

结构化抽取要求模型反复输出相同的键：`evidence_ids`、`title`、`summary`、`owner`；也要求精确复制 prompt 中的数字和 utterance ID。对 prompt token 施加 1.1 repetition penalty，等于系统性降低这些目标 token 的 logits。这能解释：

- 固定字段漏引号或变名；
- `utt-3` 变 `utt3`、`-3`；
- 复制文本被不必要地同义改写；
- 短 probe 的逐字空格/中英混杂。

不过，Q1/Q2 在关闭 prompt penalty 后仍有明显数字、标点和 JSON 错误，因此不能再把这些症状主要归因于 penalty 或温度。

### 2. W4A16 量化与校准：重点是 QNN 静态 A16，而非只有 W4 标签

量化本身不是必然不可用。Qwen 官方 AWQ-int4 结果显示，相比 BF16，原版 4B non-thinking 的 LiveBench 从 48.4 到 46.1、GPQA 从 41.7 到 40.4、MMLU-Redux 从 77.3 到 73.4，说明一个正确实现的 INT4 可以保留大部分平均能力，但仍有可测损失。[Qwen3-4B-AWQ 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-AWQ#performance)

同一 W4 权重在 CPU 上明显好于 QNN，说明 **W4 权重本身不足以解释 QNN 的新增退化**。MeetNote 当前 QNN 产物比“INT4”这个标签包含更多具体风险：

- `lm_head` 也为 W4，最终 151,936 词表排序直接受 4-bit 权重误差影响；数字、引号、连字符通常只是相邻候选之间很小的 logit 差。
- QNN 图的激活不是浮点 A16，而是通过业务外校准得到的静态 INT16 scales/zero points；这是 CPU 根图与 QNN 图的重要执行差异。
- 默认校准是英文 WikiText、128 tokens；生产是中文、长 prompt、数字、JSON 标点和重复键。
- `omni_epochs=1`，而 MNN 默认值为 20；这是显著压缩的优化预算，尚无该设置对 Qwen3-4B W4A16 的准确率报告。
- Qualcomm 自己的端侧 Llama 3.2 3B 方案也不是所有层统一 W4A16，而是部分层使用 W8A16，说明敏感层混合精度是实际部署手段。[Qualcomm Llama-v3.2-3B-Instruct-SSD](https://aihub.qualcomm.com/compute/models/llama_v3_2_3b_instruct_ssd)

历史 `lm_head=8` 变体和本轮局部 `q_proj=W8` 均不能证明“8-bit 更差”。本轮已经确认 W8 的
权重引用只改变一个 op，但 QNN blockwise W8 执行仍使对应 Q 分支崩坏；正确实验必须增加
MNN CPU W8 parity，并在 QNN 后端修复或绕开该未验证分支后再比较质量。

### 3. 模型能力与模型版本：中等概率

原版 Qwen3-4B 不是专用信息抽取模型，也没有结构化输出正确性保证。官方给出的原版 non-thinking IFEval 81.2、BFCL-v3 57.6 已说明：它的指令遵循不错，但远非 100%。[Qwen3-4B-Instruct-2507 官方对比表](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507#performance)

同时，官方明确把 `Qwen3-4B-Instruct-2507` 描述为原版 4B non-thinking 的更新版，在指令遵循、逻辑推理、文本理解和工具使用上有明显提升；其 IFEval 为 83.4。[官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

所以模型升级可能提高上限，但在验证 BF16 reference 之前，不应把当前坏 JSON 直接定性为“4B 天生不行”。如果 BF16 原版在同一 prompt 下 20 次都能正确复制 97% 且 JSON 合法，问题就不在参数量本身。

### 4. Prompt/schema 设计：中等概率

原链路要求一次生成多种数组、嵌套对象、证据 ID、状态、owner、due、mitigation 等字段。很多字段在原文中没有值，却要求模型填满完整对象，这会诱发“合理补全”。完整 few-shot JSON 还会造成 copy bias；repair 又把已损坏 JSON 和错误事实作为新上下文。

MNN 当前生成路径只有自由采样 pipeline，没有 JSON grammar/schema decoder；源码列出的步骤是 penalty、top-k、top-p、min-p、TFS、typical 和最终 selection。[MNN sampler source](https://github.com/alibaba/MNN/blob/0bff03cbef43c783f44e41484b9f8a0b28bd758d/transformers/llm/engine/src/sampler.cpp)

因此 prompt 最多提高概率，不能提供语法保证。对于需要证据可追溯的纪要，抽取层应优先输出最小事实单元，且不存在于原文的字段不要要求模型“补齐”。

### 5. Token budget：确定存在，但不是所有错误的根因

Qwen 官方强调需要充足输出长度，并对原版/2507 推荐远高于 512 的通用上限；这不是说纪要必须生成上万 tokens，而是说明 512 并非模型训练时的可靠完整性边界。[Qwen3-4B Best Practices](https://huggingface.co/Qwen/Qwen3-4B#best-practices)

MeetNote 已观察到 512 token 刚好耗尽并截断，所以预算不足可以解释 JSON 尾部缺失。但 `realistic-18-budget-2700` 在约 286 decode tokens 时已经产生 `17%` 和坏引号，说明增大上限只能解决截断，不能解决事实错误和自由 JSON 语法错误。

### 6. QNN 编译图/静态 A16/HTP 执行：已确认主要根因并完成最终排序验收

本轮 C0/Q0 已完成相同权重、tokenizer、prompt 和采样配置的最终输出对照。QNN 的 evidence ID、数字和 JSON token 明显比 CPU 根图差，说明 QNN 编译执行路径不是低概率旁支，而是主要贡献层之一。该结论与是否出现 QNN fatal 无关：数值执行成功不等于 token ranking 与 reference 足够接近。

最终 logits clipping 已通过单变量消融确认并修复；业务校准 donor 和 A16 scale graft 都没有
减少长 prompt 的 5/48 top-1 分叉。tensor injection 先把 step 32 的主要差值定位到 layer 0
Q；同手机 ARM CPU/QNN 的前后边界又确认 `q_norm` cosine 仍为 0.99976，而 RoPE 后骤降到
0.94582。只迁移 Sin/Cos 后 Q cosine 为 0.96102；把 phase multiply 一并迁移后恢复到
0.99976。因此首个主要差值已经因果定位为 HTP FP16 大相位 multiply + trig，而不是
`q_proj`、Q norm 或统一 A16 clipping。

因此下一步不再制作更多 q_proj 权重量化变体。layer 0 Q 门槛和无 debug 完整 wrapper 的
最终排序门槛都已通过：48-step top-1 从 43/48 恢复到 47/48。若产品目标要求解释最后
1/48，再从 step 39 沿后续层追踪残余 A16/HTP 差值；它不再阻塞当前主根因结论。
MNN 已支持 Qwen3 + QNN + OmniQuant，但该能力较新。[MNN 3.4.0 release notes](https://github.com/alibaba/MNN/releases/tag/3.4.0)

### 7. Tokenizer、chat template、资产和设备状态：低概率

- chat template 中包含官方 Qwen3 Jinja，JNI 显式关闭 thinking；简单中文正常，说明模板没有整体失效。
- 缺 embedding 时历史表现是重复 `<`，与当前连贯但事实错误的症状不同。
- vivo 冻结只在应用失去前台时停止推进；错误在强制前台状态下复现。

不过仍应做一次 HF tokenizer 与 `tokenizer.mtok` 的 token ID golden parity，避免中文数字和 JSON 标点被 tokenizer 转换差异污染。

## 根因概率排序

以下是基于当前证据的工程先验，不是统计置信区间；候选因素会相互作用，但为便于安排实验，强制归一为 100%。

| 排名 | 候选层 | 概率 | 已有证据强度 |
|---:|---|---:|---|
| 1 | HTP FP16 大相位 multiply + Sin/Cos / RoPE 执行路径 | 64% | 同手机单变量修复后 RoPE Q 从 0.94582 恢复到 0.99976，最终 top-1 从 43/48 恢复到 47/48 |
| 2 | Prompt/schema 过复杂且无 constrained decoding | 14% | CPU 对照仍有对象/数组错误、幻觉与截断，架构上没有语法保证 |
| 3 | W4 权重、全 W4 `lm_head`、OmniQuant 优化预算 | 5% | 官方 BF16 与 MNN W4 CPU 47/48；W4 block32 未改善最早 Q 边界 |
| 4 | 原版 Qwen3-4B 能力与版本上限 | 5% | BF16/W4 排序接近，但 CPU 自由生成仍编造 owner，复杂结构能力非 100% |
| 5 | 其他内部 A16 tensor 或后续 HTP 算子的局部敏感性 | 4% | RoPE 是首个大差值，但后续层仍可能存在独立残差 |
| 6 | 采样器配置/`penalize_prompt_tokens`/温度 | 4% | 对循环影响已证实；teacher forcing 绕过采样后仍有明确分叉 |
| 7 | 256/512 token 输出预算 | 3% | 截断已确认；无法解释未到上限时的数字错误 |
| 8 | tokenizer/template/资产/设备状态 | 1% | 相同 tokenizer/prompt 的 CPU 明显更好；前台状态只影响时延 |

按症状拆分时排序会不同：

- **无限重复**：旧 mixed pipeline 缺 `penalty` 是第一主因；长 prompt 下惩罚 prompt token 与 QNN token 排名劣化会继续放大尾部循环。
- **固定 JSON 语法错误**：QNN 静态 A16/执行路径 + 自由采样无 grammar > prompt/schema > W4/模型能力；prompt penalty 不是必要条件。
- **`97% -> 17%/7%`**：QNN 静态 A16/执行路径 > W4 `lm_head`/校准 > 4B 复制能力 > 采样温度。
- **无依据 mitigation**：prompt 要求填完整对象 > 4B 生成倾向 > 采样/量化；这不是后端 crash 类问题。

## 可证伪实验矩阵

### 统一数据与指标

固定三个输入集，每个配置至少跑 20 个 seed：

1. `copy-numbers`：100 条中文数字、百分比、日期、金额、utterance ID 的精确抽取。
2. `json-minimal`：仅 4 个 utterances，输出 2 个字段和 `evidence_ids`。
3. `meeting-realistic-18`：当前真实感会议压力输入。

统一指标：

- `number_exact_match`：数字/单位逐字一致；
- `evidence_id_exact_match`；
- `json_parse_rate` 与 `schema_valid_rate`；
- `unsupported_claim_rate`：输出陈述无任一证据句支持；
- `loop_rate`；
- `eos_before_limit_rate`；
- teacher-forced 每步 top-1 agreement、top-5 overlap、logits cosine/KL；
- 延迟、RSS 只作为次级指标，不能代替质量指标。

### P0：先验证采样配置，不重新量化

| 实验 | 唯一变量 | 预测 | 可证伪结论 |
|---|---|---|---|
| S0 | 当前配置 | 基线 | - |
| S1（已执行 1 次 realistic-18） | 增加 `penalize_prompt_tokens=false` | 循环消失，但 JSON/ID 仍坏 | prompt penalty 影响循环，不是结构错误主因 |
| S2 | S1 + `temperature=0.7` | 多样性增加；若低温在固化错误，数字 exact match 提升 | 若错误率不变，低温不是主因 |
| S3（已执行 1 次 realistic-18） | S1 + 官方 non-thinking 参数 `0.7/0.8/20/min_p=0` | JSON/ID 仍坏 | 官方采样参数不能修复当前 QNN 路径 |
| S4 | S1 + `presence_penalty` 0/0.5/1.5 | 测循环与语言混合的 trade-off | 只保留能降低 loop 且不损害 exact match 的值 |
| S5 | 无 penalty，仅 top-k/top-p/temp | 判断当前事实错误是否由 penalty 引入 | 若 `97%` 恢复但循环变多，需 generated-only penalty 而非全关闭 |

通过门槛建议：`copy-numbers` 100%，`json-minimal` schema >= 95%，20 次无循环。先完成该组，避免把错误采样参数带入昂贵的重新量化实验。

### P1：三阶 reference 对齐，严格分离量化与 QNN

同一 prompt、同一 chat template、同一 teacher-forced token 序列：

| 阶段 | 执行对象 | 隔离因素 |
|---|---|---|
| R0 | HF BF16 `Qwen/Qwen3-4B` | 模型能力 + prompt 的 reference |
| R1 | MNN CPU，转换前根 `llm.mnn` W4A16 | R0 -> R1 只看量化/MNN 导出 |
| R2 | QNN C64 context graphs | R1 -> R2 只看 QNN 转图/HTP runtime |

不要先比较最终自由生成文本。对固定正确答案做 teacher forcing，在每个位置记录 top-20 logits，重点观察：`九十七`、`97`、`"`、`:`、`,`、`-`、`utt`、`]`、`}`。

判定规则：

- R0 已失败：模型版本/prompt/schema 是主因。
- R0 通过、R1 明显掉点：量化/导出是主因。
- R1 通过、R2 明显掉点：QNN 转图/runtime 是主因。
- 三者 teacher-forced 接近，但自由生成分叉：采样 RNG/pipeline 是主因。

### P2：量化消融

在 P1 确认 R0 -> R1 有明显质量损失后再生成以下模型：

| 变体 | 目的 | 预测 |
|---|---|---|
| Q0 | 当前 W4A16、全 W4、WikiText、1 epoch | 基线 |
| Q1 | 仅 `lm_head` 改 W8 或 FP16 | 若数字/标点 top-k 排名恢复，定位 `lm_head` |
| Q2 | layer 0 `q_proj` 单独 W8/FP16 | 验证已由 injection 定位的第一层 Q 数值路径 |
| Q3 | 当前精度，OmniQuant 20 epochs | 验证 1 epoch 是否欠优化 |
| Q4 | 当前精度，使用中文会议/JSON/数字代表集校准，seq len 512/1024 | 验证校准域与长度 |
| Q5 | W8A16 reference | 若显著恢复，说明全 W4 压缩过激 |
| Q6 | HQQ only 与 OmniQuant only | 分离两种优化组合是否存在实现交互 |

每个变体先做模型文件引用完整性、CPU logits parity，再编译 QNN。历史 LM8 乱码变体在 parity 之前不得作为质量结论。

### P3：模型能力与 prompt 消融

只在 BF16 reference 上做，避免被量化噪声干扰：

| 实验 | 变量 | 目标 |
|---|---|---|
| M0 | 原版 Qwen3-4B non-thinking | 当前模型上限 |
| M1 | Qwen3-4B-Instruct-2507 | 验证官方更新版是否降低 schema/hallucination 错误 |
| M2 | 同系列 8B/Instruct reference | 判断 4B 参数量是否是主要瓶颈 |
| P0 | 当前完整 schema | 基线 |
| P1 | 最小 `{facts:[{text,evidence_ids}]}` | 分离 schema 复杂度 |
| P2 | 两阶段：先抽取原文 span，再确定性组装 JSON | 验证生成式组装是否为失败源 |
| P3 | 不要求缺失的 mitigation/owner/due | 测无依据补全率 |
| P4 | 128/512/1024 output tokens | 只判断截断，不混淆事实准确率 |

如果 BF16 原版在最小 schema 下已经达到目标，而完整 schema 失败，则不需要先换更大模型；应先缩小模型承担的状态边界。

### P4：tokenizer/chat template golden test

固定 20 条包含中文数字、百分号、JSON 标点和 `utt-319` 的字符串：

1. HF tokenizer 输出 token IDs；
2. MNN `tokenizer.mtok` 输出 token IDs；
3. 比较 `enable_thinking=false` 后最终 prompt 文本和完整 token IDs；
4. 分别 decode，要求 round-trip 一致。

若不一致，先修 tokenizer/template 导出；若一致，可将该层概率降到接近 0。

## 推荐执行顺序

1. **产品配置先显式设 `penalize_prompt_tokens=false`，并恢复官方 non-thinking 采样温度。** 这能降低循环，但不要把它当成内容质量修复。
2. **局部 W8 后续实验统一走已验证的 per-channel fallback。** 不再把原生 W8 block 塞进
   LPBQ，也不使用 HTP Conv2d 不支持的通用 BLOCK encoding；每个敏感层变体先在服务器完成
   CPU parity 和 context 生成，只在最终 logits/layer trace 门槛上使用手机。
3. **采用 CPU multiply + trig 分图，不采用仅 CPU-trig。** layer 0 Q 已通过 0.99 门槛，
   无 debug 完整 wrapper 也已把最终 48-step top-1 从 43/48 恢复到 47/48。下一步是把
   该分图规则固化到可复现的生产资产构建，不再重复跑同一真机验收。
4. **R0 BF16 reference 已补齐。** 官方 BF16 与 MNN W4 CPU 为 47/48 top-1；后续只需扩大
   fixture 覆盖，不再把当前 layer 0 RoPE 差值归因于 4B 模型能力。
5. **不再优先尝试全局 A16 scale、仅换校准文本或更多 q_proj 精度变体。** 如果完整
   CPU-RoPE 仍未修复最终排序，再根据新的 layer trace 选择下一个敏感边界。
6. **产品链路继续采用最小事实抽取 + 确定性组装/校验 + fallback。** 完整自由 JSON 不能依赖 prompt 获得语法和证据保证。

## 最终判断

目前可以有把握地说：

- **旧式无限重复主要是采样器配置问题；generated-only penalty 能继续降低循环。**
- **QNN 编译执行路径相对同权重 CPU 有明确质量退化；teacher forcing 已排除 sampler，并直接复现数字 logits 饱和和 top-1 翻转。**
- **代表性业务校准和 q_proj W4 block size 都只带来轻微混合变化，已不是剩余问题的首要解释；tensor injection 已确认 layer 0 Q 路径是主要早期贡献点之一。**
- **局部 W8 的灾难性失真已修复并闭环为 MNN/HTP blockwise encoding 合约错误；新路径回到 W4 的 43/48 top-1 水平。**
- **同手机 ARM CPU/QNN 已把首个主要差值闭环到 HTP FP16 大相位 multiply + Sin/Cos；将两者移到 CPU 后 layer 0 Q cosine 从 0.94582 恢复到 0.99976。**
- **修正 graph2..38 到旧 graph1..37 的映射后，无 debug 完整模型已通过 48-step 验收：CPU/QNN top-1 从 43/48 恢复到 47/48，cosine mean 从 0.8683 恢复到 0.9480。**
- **临时 wrapper 的 SIGSEGV 是 graph 序号错位导致的应用进程崩溃，不是手机重启；验收后设备 uptime 连续超过 6 天，wide64 基线也已恢复并校验。**
- **坏 JSON 和无依据字段还包含独立的 prompt/schema 与 4B 能力问题；即使 QNN parity 修好，仍需要确定性组装和验证。**

下一阶段应先把 CPU RoPE 分图固化为可复现的生产模型资产，再量化三个独立差值：
BF16 到 W4A16、修复后 CPU W4A16 到 QNN，以及自由采样到受控结构化抽取；不再把已闭环的
W8 encoding 或 RoPE 问题混入模型能力判断。
