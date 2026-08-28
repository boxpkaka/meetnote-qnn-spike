# MOSS MNN CPU Attention PoC

日期：2026-08-25

## 结论

现有 MNN `FusedAttention` 的配置调优和局部内存优化路线停止。候选只有在 256-token 门禁显示显著收益时才进入完整 case 2；其中 KV 扩容 512 是唯一短测显著候选，但完整生成质量失败，已拒绝。

在相同 case 2、相同 3996-token prompt、固定 256-token 诊断窗口下，没有候选同时满足正确性和有效加速：

- FP32 flash 基线：decode 22.67 s，CPU Attention 15.04 s。
- FP32 non-flash：decode 22.93 s，无有效收益。
- K INT8 / K+V INT8：分别退化为“手机”和“嗯”重复，并且更慢。
- 8 CPU threads：decode 27.89 s，线程同步开销放大。
- CPU 4-7 affinity：decode 33.28 s，同时限制 QNN/音频线程，明显更慢。
- 全局 FP16：QK/QKV 数学核约快 1.8 倍，但输出时间戳和 speaker token 结构损坏；整个 Attention 仅从 15.04 s 降到 13.86 s，decode 反而为 25.06 s。
- TQ4 K：退化为“对”重复，QK 从 4.12 s 恶化到 8.96 s。
- decode-only grouped GQA：输出前缀与基线一致，但 Attention 仅降到 14.75 s，decode 为 23.21 s；实验代码已撤销。

因此，不能把现有 CPU Attention 通过 mode、线程数、绑核、现成 KV quant 或 2:1 GQA batching 优化到 RTF 1.0。

## 性能门槛

完整 case 2 profile：

- decode：418.55 s
- CPU Attention：302.88 s
- 其余 decode：115.67 s
- RTF 1.0 对应 decode 预算约 213 s

即使其余 decode 完全不变，CPU Attention 也必须降到约 97.33 s，相当于至少 3.11 倍加速。256-token 分段 profile 显示：

| 阶段 | 时间 |
| --- | ---: |
| KV update | 1.56 s |
| Q pack | 0.01 s |
| QK | 4.12 s |
| softmax | 1.27 s |
| QKV | 3.96 s |
| output update/unpack | 0.39 s |
| Attention 总计 | 15.04 s |

QK 与 QKV 是主要数学核，但临时 buffer、线程唤醒、分块准备和同步等未细分开销仍约 3.7 s。仅把数学核换成 FP16 不能达到 3.11 倍总 Attention 加速。

## 正确性审核

所有 256-token 运行都是诊断窗口，因主动耗尽 token budget，`status=error` 和 `transcript_parse_failed` 不作为产品失败判据；人工语义审核直接比较 raw prefix：

- mode 8、mode 0、8 threads、affinity、grouped GQA：截断前缀语义与基线一致。
- mode 9：从首 token 开始重复“手机”。
- mode 10：连续重复“嗯”。
- global FP16：时间戳、小数点和 speaker token 结构异常。
- mode 13：连续重复“对”。

未发现 crash、SSR、OOM 或 non-finite。由于没有候选通过 256-token 门禁，未运行 teacher-forced sparse trajectory，避免无意义的长时间设备实验。

## 实现状态

- 手机已恢复 release v7 基线：
  - config：`3ee6878785790c1c3ea413e937b739610f0a2a8a67a1ae6a78c1a1841d849194`
  - runner：`e18218c2a824d70c36b963ff15a0542ff517c3c1a9951f1d641e3cd0cb9c9435`
  - libMNN：`75a7ea0d234809d16003c294e0cb036f0dad981f735231e6bce8c591dddac4ce`
- grouped GQA mode 15 已从服务器源码撤销。
- 服务器保留分阶段 profiler，用于后续架构实验；它只增加观测，不改变 mode 8 算法。
- profiler 复现补丁：`third_party/patches/mnn/0016-profile-moss-decode-breakdown.patch` +
  `third_party/patches/mnn/0017-profile-moss-attention-phases.patch`；`0017` 已通过 reverse apply check。
- phase profiler 构建 hash：libMNN `45f4488a77648d38b4c8e53a72426d40a5c7b4ac98ed331016672a934ce1996b`，
  runner `04e117f3e3bd140bb7f8f726caee9c5f5268312ac3a5ce7bf5b6d81d5f73ac2d`。
- 完整本机证据：`build/device-validation/20260825-moss-cpu-attention-sweep/`
- 汇总：`comparison-summary.json`
- 文件 hash：`artifacts-sha256.txt`

## 下一步建议

停止小范围 CPU 调参。若继续追 RTF 1.0，只剩架构级工作：

1. 构建 Attention-only mixed-precision backend：position 始终 INT32，RoPE 保持当前 FP32/正确路径，只把 Attention/KV 改为 FP16，并消除每层临时 buffer 与线程同步开销。
2. 在单层 microbenchmark 上先证明总 Attention 至少 3.1 倍，而不是只证明 QK/QKV kernel 更快。
3. 若单层达不到 3.1 倍，停止 CPU Attention 开发，转为 Attention 与 QNN projection/FFN 的联合架构优化，不再做单变量 mode sweep。

全局 `precision=low` 不能作为 mixed precision 的替代品，因为它会同时降低 RoPE/position 相关计算精度，复现长 position 结构性错误。

## 2026-08-25 补充：内存与 KV 扩容实验

在同一 3996-token prompt、256-token decode 窗口上又完成三组实现级 PoC：

| 候选 | 256-token 结果 | 完整 case 2 | 决策 |
| --- | --- | --- | --- |
| 复用 Attention 临时 buffer，并删除未使用的 unpack buffer | Attention `15.819 -> 15.761 s`，仅 `0.37%`；decode 反而慢 `0.72%` | 未运行 | 拒绝并撤销 |
| KV 扩容步长 `64 -> 512` | 两次 decode `18.038/18.197 s`，相对约快 `20%`；raw prefix 完全一致 | decode `418.150 s`，与 `418.550 s` 基线基本相同；约 257 秒后重复“对”，遗漏充电宝、手机壳、贴膜和品牌限量赠品讨论 | 质量失败，拒绝并撤销 |
| V cache 容量不变时跳过重分配，并用 4 线程复制 K/V head | decode `22.512 s`，相对历史同 case 基线仅约 `0.7%`；Attention 处于测量噪声范围 | 未运行 | 收益不足，拒绝并撤销 |

KV 512 的短测收益没有转化为完整生成收益，且改变 cache relocation 轨迹后出现长序列语义退化。该候选不能作为性能优化保留，也不能用 parser、repetition penalty 或结果后处理掩盖。完整人工语义审核和 artifacts 位于本机
`build/device-validation/20260825-moss-kv-expand-512/`。

## 2026-08-25 补充：QNN 图级分解

通过临时、环境变量控制的 QNN graph wall-time profiler，在 case 2 的 256-token 窗口捕获 1200 次 graph execute：

| 图组 | 每 token 平均 | QNN 占比 |
| --- | ---: | ---: |
| `graph1_0..1`，layer 0 Attention 前的 prefix | `1.234 ms` | `4.8%` |
| `graph1_2..29`，28 层 projection/FFN | `19.294 ms` | `74.4%` |
| `graph1_30`，final norm/lm_head | `5.422 ms` | `20.9%` |
| 合计 | `25.950 ms/token` | `100%` |

这证明 QNN 侧 95.3% 时间确实在 projection/FFN/lm_head，W8A16 有明确优化面；但完整 case 2 的 QNN 总预算约 `107 s`，即使理想地加速 2 倍也只节省约 `53 s`。CPU Attention 仍约 `303 s`，所以 W8 不能单独把 decode 压到约 `213 s`。图级 summary 和 graph/model 映射位于本机
`build/device-validation/20260825-moss-qnn-graph-profile/`。

## 2026-08-25 补充：W8A16 长序列复核

历史 `release-w8-pure-fp16-normal-v1` 已通过 30/60/90/120 秒语义审核，180-step 同机 CPU/QNN top-1 为 `99.44%`、cosine mean/min 为 `0.997432/0.966427`，因此不重复量化构建。首次直接运行该旧 release 的 case 2 使用了缺少 prompt 末尾换行的旧 runner，4096 token 耗尽且不可作为量化结论。

随后只替换为已修复 newline contract 的 runner，并增加固定 300 秒 prompt contract，保持 W8 decoder graph、权重、`precision=normal` 和 8192 context 不变。完整 case 2 结果：

- `status=ok`，3730 generated tokens，122 segments，4 speakers；
- 约 257 秒后没有重复退化；充电宝、手机壳、屏幕贴膜、品牌 logo 和限量版赠品讨论全部保留；
- decode `439.453 s`，total `554.283 s`，RTF `1.848`，peak PSS `3.512 GB`；
- 本次设备由前序长测升温，不能用于精确冷机速度排名，但它没有显示足以改变架构判断的性能收益。

人工审核为 `pass_with_recorded_differences`：相对 W16 v23 存在时间边界和少量措辞差异，最后完整解析片段结束更早，但实质尾部讨论完整。证据位于本机
`build/device-validation/20260825-moss-w8-case2-fixed-prompt/`。

结论更新为：W8 可以保留为 projection/FFN/lm_head 的混合精度候选，但不能取代 W16 正确性基线，也不能单独解决 RTF。下一轮若继续实现，应把资源投入到可独立验证的 Attention kernel/backend；达到单层总 Attention `>=3.1x` 后，再与 W8 QNN 图联合评估。

## 2026-08-26 补充：MNN OpenCL Attention

为排除“GPU 只差构建开关”的可能性，在独立构建目录启用 `MNN_OPENCL=ON`，同时保留现有
QNN plugin，使 projection/FFN 继续走 HTP、`FusedAttention` 走 Adreno OpenCL。手机使用
独立目录 `/data/local/tmp/meetnote-moss-opencl-attention-v1`，大资产只链接到 release v7，
没有覆盖正确性基线。

真机结果：

| 模式 | 32-token smoke | 256-token 门禁 | 决策 |
| --- | --- | --- | --- |
| OpenCL `precision=normal` | decode `2.159 s`，CPU 为 `2.558 s` | decode `8.710 s`，但 121 token 后时间戳与 speaker token 结构损坏 | 正确性失败 |
| OpenCL `precision=high` | 32-token raw output 与 CPU 逐字一致 | raw prefix 与 CPU mode 8 一致，但 decode `23.758 s`，CPU 为 `18.038 s` | 性能失败 |

`normal` 的速度显示 Adreno 上的 FP16 Attention 数学核仍值得作为自定义 mixed-precision
kernel 的线索，但不能直接使用整个 MNN OpenCL FP16 路径。`high` 证明 OpenCL 算子本身能维持
正确性，却比 CPU 慢 `31.7%`。后续如果重启 GPU 路线，必须把 INT32 position、RoPE table
lookup 和必要的归一化保持在正确精度，只将 QK/softmax/QKV 与 FP16 KV 放到专用 kernel；不能再
切换全局 backend precision。

构建 SHA256：libMNN `a9f0e1cc...`、libMNN_CL `5533e5f5...`、runner
`2b304c9c...`。完整真机日志、配置、logcat、result summary 和 hash 位于本机
`build/device-validation/20260825-moss-opencl-attention/`。首次 smoke 因
`ADSP_LIBRARY_PATH` 错用冒号而失败；修正为分号后正常运行，未出现 SSR、重启或 OOM。
