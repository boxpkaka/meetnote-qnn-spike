# MOSS Qualcomm-native 静态 Decoder PoC（2026-08-25）

## 结论

本轮已经推进到性能硬门禁，结论是：**保留 hybrid v23 正确性基线，不继续把当前
FP16 Qualcomm-native 静态 decoder 扩展到 28 层产品集成。**

静态 Attention/KV 的数值正确性成立，`position` 全程保持 INT32，位置 `2048` 后没有
断崖；但按真机逐层实测外推，case 2 的端到端 RTF 为 `1.166`（同次 profile 基线）或
`1.344`（按原始 `562.59 s` 基线保守缩放）。后者已经超过方案约定的 `RTF 1.3`
止损线，二者都达不到 `RTF <= 1.0`。

因此，继续完成 28 层、自由生成和 10 个 AliMeeting case 只会验证一个已经由真机数据
否定的性能方向。后续若重启该方向，必须先引入不同的核心变量，例如可用的 grouped-GQA
HTP kernel，或经过真实 teacher trajectory 校准的官方 KV INT8 recipe；它们应作为新的
PoC，而不是继续扩大本实现。

## 环境与边界

- 服务器：`mlu-cpu:/data/mingdong/workspace/meetnote-qnn-spike`
- 分支：`perf/qnn-static-decoder-poc`
- ExecuTorch：`v1.4.1`，commit `e4d02...`
- QNN SDK：`2.48.40.260702`
- 模型 checkpoint SHA256：`9a0ceb...`
- 真机：SM8850，Android 16，设备序列号仅保存在本机验证环境
- 量化、导出和编译只在服务器执行；USB 真机运行和 artifacts 只在本机执行

ExecuTorch 的 MHA-to-SHA pass 缺少 Qwen3 per-head RMSNorm 支持。本轮只增加了
`aten.rms_norm.default` 的逐 head 拆分，补丁保存在
`third_party/patches/executorch/0001-support-qwen3-rmsnorm-mha2sha.patch`。

## Profile 门禁

case 2 实测：

| 指标 | 时间 |
|---|---:|
| Total | 490.511 s |
| Decode | 418.550 s |
| CPU Attention | 302.881 s（decode 的 72.36%） |
| QNN execute/sync | 100.244 s（decode 的 23.95%） |
| Unaccounted | 14.697 s |

CPU Attention 明显超过 55%，所以通过第一道门禁，值得做静态 Attention PoC。profile
显示每 token 有 28 次 CPU attention 和 31 次 QNN graph 调用，符合“28 层 graph +
3 个外围 graph”的现有结构。

原始方案中的 `562.590 s` 总耗时、`474.140 s` decode 作为保守基线继续保留。达到
RTF 1.0 时，保守 decode 预算只有 `211.550 s`。

## 实现与正确性

实现了三个可复现工具：

- `tools/build_moss_static_attention_poc.py`：单层 Qwen3 GQA Attention、静态 FP16 KV、
  预计算 RoPE table、INT32 position。
- `tools/build_moss_static_decoder_layer_poc.py`：完整 layer 0，包括 input RMSNorm、Q/K/V、
  Attention、residual、post-attention RMSNorm、gate/up/down MLP。
- `tools/build_moss_static_decoder_shard_poc.py`：连续 2/4 层 shard，用于观察融合收益和
  数值累积。

CPU reference 在 `0/1/63/64/2047/2048/2049/3995/3996/7000/10239` 全部通过；真机
FP16 graph 输出全部 finite，KV 更新位置正确。关键结果：

| Graph | Context / position | 真机延迟 | QNN vs FP16 hidden cosine |
|---|---:|---:|---:|
| 1 layer | 4096 / 3996 | 1.4122 ms/layer | 0.99999946 |
| 1 layer | 6144 / 4096 | 2.1476 ms/layer | 0.99999946 |
| 1 layer | 7168 / 7000 | 2.6573 ms/layer | 0.99999934 |
| 1 layer | 8192 / 7000 | 3.3533 ms/layer | 0.99999934 |
| 1 layer | 10240 / 10239 | 4.4419 ms/layer | 0.99999958 |
| 4 layers | 4096 / 3996 | 1.3331 ms/layer | 0.99998420 |
| 2 layers | 8192 / 7000 | 3.3930 ms/layer | 0.99999714 |
| 4 layers | 8192 / 7000 | 3.5699 ms/layer | 0.99998820 |

4 层在 c4096 比单层快 5.6%，但在 c8192 比单层慢 6.5%；2 层在 c8192 也慢 1.2%。
说明长 context 下 shard 融合增加了内存压力或 spill，不能套用官方短 context 示例中的
“每 4-7 层一个 shard”作为固定最优解。c5120 的单层延迟 `2.2451 ms` 还慢于 c6144 的
`2.1476 ms`，也表明 QNN 编译结果并非随容量单调变化，必须逐容量真机测量。

4 层 c4096 的 final hidden cosine `0.99998420` 高于 v23 sparse trajectory 的最低值
`0.99978399`，且接近其均值 `0.99998193`。因此本轮止损原因是性能，不是已发现的数值
错误。PoC 没有进行自由生成，所以没有生成结果可做 transcript/evidence 语义审核。

## 量化与替代布局消融

### W8A16 weights

W8A16 的 PTE 约减半，但没有加速：

| Context | FP16 | W8A16 | hidden cosine | new K cosine |
|---|---:|---:|---:|---:|
| c4096 | 1.4122 ms | 1.4580 ms | 0.99998975 | 0.99984229 |
| c8192 | 3.3533 ms | 3.5216 ms | 0.99999249 | 0.99984461 |

因此 W8A16 同时降低速度和 KV 数值质量，不进入投影最优组合。

### Grouped GQA

尝试保留 8 个 KV group、每组 2 个 query head，以避免 MHA-to-SHA 将 KV 展开成 16 head。
该 graph 能编译，但在 c4096 和 c8192、shared buffer 开关两种配置下都确定性触发 DSP
SSR。SSR 后重新运行原 FP16 graph 能正常恢复，说明故障由 grouped graph 结构触发，而
不是设备持续异常。该布局在当前 ExecuTorch/QNN 组合上止损，不再重试。

随后补做了两种不依赖 5D grouped tensor、也不启用 MHA-to-SHA 的稳定表达，并加入与
expanded GQA reference 的 CPU 单测：

| c4096 Attention-only graph | 真机延迟 | DSP SSR | QNN vs FP16 cosine |
|---|---:|---|---:|
| 显式展开 16 个 KV head 的 control | 12.1522 ms | 否 | 0.99999976 |
| 8 组 3D `bmm` | 4.8439 ms | 否 | 0.99999976 |
| 8 组显式 unrolled matmul | 3.3254 ms | 否 | 0.99999976 |

两种非展开表达都解决了 SSR，并保持与 control 完全相同的真机数值输出；其中 unrolled
是稳定实现里最快的。但它仅 Attention 就需要 `3.3254 ms`，已经显著慢于现有优化完整
layer graph 的 `1.4122 ms`，因此在 c4096 即触发性能硬门禁，不再扩大到 c8192 或完整
layer。

unrolled + MHA-to-SHA 也做了编译判别。pass 只拆分了第一个 group，随后生成 0-sized
non-batch dimension，QNN context binary generation 以 error `6005` 失败，未进入真机。
结论是：在当前 ExecuTorch/QNN lowering 中，已验证的 grouped 表达要么 5D graph 编译后
SSR，要么稳定但失去 optimized MHA 的性能；继续排列 reshape/matmul 不能改变投影结论。
若后续重启 grouped-GQA，边界应改为 Genie/官方 HTP context recipe、直接 QNN graph，或
修复 MHA-to-SHA pass，而不是继续增加等价 ATen 表达。

### INT8 KV

显式 QDQ 的 uint8 KV graph 被拆成两个 QNN context，c8192 真机约 `9.42-9.47 ms/layer`，
输出 buffer contract 也没有被 `TagQuantIO` 正确改成 uint8，因而既慢约 3 倍又无法形成
有效正确性比较。该实验源文件已删除，失败 artifacts 保留。

默认 `QuantDtype.use_8a8w` 也不会自动量化当前 KV graph I/O：`k_cache/v_cache` placeholder
仍是 FP16，`cat` 直接消费 FP16，`new_k/new_v` 由无 quant config 的 `aten.clone` 以 FP16
输出。要得到真正 INT8 KV，需要采用官方 static-LLM quant recipe、可被 annotation 的
cache update/output 结构，并使用真实 teacher trajectory 校准。

随后又按官方源码补做了 `annotate_kv_8bit`、shape-based `TagQuantIO` 和无 clone cache
output 的完整尝试。它证明这不是一个可在当前 PoC 上直接补齐的配置开关：

- c4096 编译被拆成 **15 个 QNN context**，编译期出现多处 FP16/UFIXED8/UFIXED16
  datatype mismatch，已经失去单 graph 加速前提；
- 仅用诊断 fixture 校准时，QDQ hidden cosine 只有 `0.91743`，new K/V 分别为
  `0.99747/0.99870`，远低于正确性 operating point；
- ExecuTorch 通用 QNN runner 仍按逻辑 FP16 tensor size 装载 cache，不能直接承载被
  `TagQuantIO` 改为 uint8 的物理 I/O；为了判别 runtime 边界而做的 padding 运行触发
  DSP SSR；
- SSR 后原 c4096 FP16 graph 立即恢复，单次为 `1.49 ms`，设备没有持续故障。

这与官方 static LLM wrapper 的专用量化、metadata、IO manager 和 calibration pipeline
是一个整体相吻合。单独抽取 annotation/TagQuantIO 不能形成有效 INT8 KV runtime。
该失败实现不保留在源码中，只保留 server report/PTE 与本机 run log。

此外，layer 0 K norm 权重存在约 `97` 的最大值，随机 fixture 的 new K 绝对值最大约
`128`、p99 约 `18.6`；简单 per-tensor INT8 K 已显示明显风险。不能使用拍脑袋 scale，
也不能把该实验结果外推为产品质量结论。

## 性能投影

投影工具：`tools/project_moss_static_decoder_profile.py`。输入和结果保存在：

- `work/moss-static-decoder-projection/case2-profile-result.json`
- `work/moss-static-decoder-projection/plan.json`
- `work/moss-static-decoder-projection/result.json`

使用可观测到的最优组合：c4096 用 4-layer shard，其余长 context 用单层 graph；最后一个
bucket 在 c7168/c8192 间按实际 layer call 数拆分。结果：

| 组成 | 投影时间 |
|---|---:|
| 28 层静态 decoder graph | 252.800 s |
| 剩余 3/31 QNN execute/sync | 9.701 s |
| 保留的 QNN copy/state | 0.728 s |
| 原 profile unaccounted | 14.697 s |
| **Projected decode** | **277.926 s** |

同次 profile 的 fixed non-decode 为 `71.961 s`，得到 total `349.887 s`、RTF `1.1663`。
按原始慢基线的 decode 比例 `1.132816` 保守缩放，得到 total `403.289 s`、RTF `1.3443`。

这个投影已经对静态路径有利：没有额外计入 28 层完整集成的 shard 切换、cache 生命周期、
应用 runtime glue 和潜在 copy。因此不能用“完整模型也许更快”覆盖当前真机证据。

## 决策

1. 保留 W16 MNN CPU 与 hybrid v23，不删除 CPU `FusedAttention` 基线。
2. 保留静态 Attention、单层 decoder、shard PoC 和 profile/projection 工具，作为后续新
   kernel 或新 quant recipe 的可复现基准。
3. 不建设本 FP16 路线的 28 层资产，不进入 case 2 自由生成、case 9/case 5 和 10-case
   AliMeeting 回归，因为已触发性能止损线。
4. 下一次只有在单层 c8192 延迟从 `3.3533 ms` 显著下降，且投影先通过 RTF 1.0 门禁后，
   才重新开放全模型迁移。
5. 真正有希望改变结论的候选只有：稳定的 grouped-GQA HTP kernel，或官方 recipe 驱动、
   真实轨迹校准且不拆成多 context 的 INT8 KV。二者都需要新的可证伪 PoC。

## Artifacts

本机真机验证根目录：

- `build/device-validation/20260825-moss-decode-profile/case2-profile`
- `build/device-validation/20260825-moss-static-decoder-layer`
- `build/device-validation/20260825-moss-static-decoder-shard2`
- `build/device-validation/20260825-moss-static-decoder-shard4`

grouped-GQA SSR 和 KV INT8 失败 artifacts 也保留在 static decoder layer 目录中。服务器端
所有导出 report/PTE 位于 `work/moss-static-decoder-*`；它们是实验产物，不纳入源码提交。
官方 annotation 尝试额外保存在服务器
`work/moss-static-decoder-layer-c4096-official-kv-int8` 和本机
`run-c4096-official-kv-int8-position-3996`，其中包含 SSR 后的 FP16 recovery log。
