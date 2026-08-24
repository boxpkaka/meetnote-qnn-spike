# MOSS SM8850 真机 TODO（交接快照）

状态：`BLOCKED_QUALITY_PERFORMANCE_REPRODUCIBILITY`（2026-08-24）。已在 Vivo V2547A / SM8850 /
Android 16 修复 prompt contract 和超过 position 2048 后的 RoPE 精度损坏；当前实验 release 包含
56 个 manifest 文件和 30/60/90/120/300 秒五档 prompt contract。case 2/case 9 的 300 秒正确性与语义
回归已通过，但 RTF 仍未通过 `<=1.0`。正式 clean rebuild、全量质量回归和修复版连续 10x300 尚未完成，
因此仍不可标记为 `production_ready`。

### 2026-08-20 真机门禁结果

1. clean end-to-end rebuild 已从固定 MNN revision 和完整 patch series 产出 55 文件合法 payload，
   manifest SHA-256 为 `d4bfb50f...`。但 clean QNN 与保留候选做相同 180-step teacher forcing 时，
   top-1 agreement 只有 `98.33%`，首个分叉在 step 3；因此“可构建”已通过，“数值可复现”未通过。
2. AliMeeting 官方 Eval 的 10 个连续 300 秒 case 已跑完，8 个 `status=ok`，2 个失败：
   `R8007_M8010_0000` 在 4096 个新 token 后耗尽预算，只覆盖到 264.88 秒；
   `R8009_M8019_0000` 输出了 speaker 与正文但完全没有时间戳，strict parser 拒绝。
3. 相对完整 HF BF16 reference 和官方 TextGrid，聚合 CER/cpCER 分别退化
   `10.01/10.85` 个百分点，timestamp boundary MAE 增加 `1.018s`，均未通过既定
   `5%/5%/100ms` 门禁。语义复核确认 case 2 后段退化为重复“对”，case 5 尾部缺失，case 9
   schema 全面崩溃；这不是 parser 可以容错或后处理修复的问题。
4. 4096 新 token 不是 case 5 的充分预算。HF BF16 在同一输入上需要 4897 个新 token，prompt
   3996 token，总长度 8893；当前 8192 context 在模型完整输出前必然到顶。10240 context 是待验证
   候选，不得只提高 `max_new_tokens` 而保留 8192 总长度。
5. 连跑无 HTP SSR、OOM、进程 crash 或 non-finite；最大 PSS `3,699,527,680 B`，但最大 RTF
   `4.198`。设备从约 36°C 升至 52°C 并持续热降频。
6. case 9 已在设备降温后单独复跑，仍以完全相同的 1530 generated tokens 输出无时间戳正文；
   RTF 从热态 `1.451` 降到 `0.644`，但格式错误不变。这排除了热降频作为该错误的根因，问题是
   输入相关的 decoder 自由生成/数值稳定性。
7. 当前 `burst` 已经是最激进的 turbo：DCVS 关闭、DSP sleep 禁用、bus/core corner 固定最大。
   它适合冷机短测，不适合连续 300 秒工作负载。正确性修复后再以 balanced/DCVS profile 做
   sustained A/B；不能把更高性能档当成质量修复。
8. 10240 首次真机复跑仍停在 4096 token，定位到 native runner 同时覆盖 config 并给
   `response()` 传入硬编码 4096。patch 0015 已改为从 payload 解析 token limits，同时保留 MNN
   所需的显式 `set_config` 和四参数 `response`；8192/10240 的 30 秒真机 smoke 均已通过，case 5
   长测因连接中断没有最终结果。
9. 2026-08-23 已完成 10240 case 5 冷机复跑：自然 EOS、4552 generated tokens、162 段、4 位
   speaker，并覆盖会议末尾；旧 4096-token 截断已修复。RTF `1.786`、peak PSS `3.878 GB`，相对 HF
   的 CER/cpCER/timestamp boundary MAE 为 `0.1342/0.1276/0.454 s`，因此仍未通过完整质量和性能门禁。
10. case 9 的 W16 MNN CPU 完整自由生成通过，CER/cpCER `0.00652`、timestamp boundary MAE
    `0.0436 s`。三组 QNN context 的 16-step teacher logits 字节一致，但该 probe 从强制输入首个
    reference token 后才开始比较，实际没有覆盖 prefill 首 token。
11. 真机自由生成 A/B 找到 case 9 无时间戳的确定性根因：native runner 内置 prompt 少了末尾换行，
    prompt index 3990 为 token `1773`（`。`），HF/teacher 固定 prompt 为 `8997`（`。\n`）。旧 runner
    输出从 `[S01]...` 开始；仅补回换行后恢复 `[0.00][S01]...`，3996 个 prompt IDs 与 HF 逐项一致。
    旧 token contract 的 official/runtime 两侧复用了同一个错误 literal，因此形成了循环验证。
12. 已在 patch 0007 和 tokenizer probe 中保留 reference fixture 的末尾换行，并增加精确文本回归测试。
    54 个 Python 单测和 repository check 均通过。修正版 release manifest 为 `563cdcce...`，包含新增的
    300 秒 prompt contract；56 个 payload 文件已在设备逐项通过 SHA。case 9 完整 300 秒真机复跑现已完成。
13. 旧 30/60/90/120 秒 pinned prompt SHA 也来自无末尾换行的同源 literal。现已用官方 HF processor、
    固定 prompt 文件和同一组 WAV 独立重算，五档 `30/60/90/120/300` HF IDs 均与修正版 MNN contract
    逐项一致；重建脚本和 payload verifier 已强制包含 300 秒 contract。设备脚本在精确 sample-count
    contract 缺失时 fail closed，不再静默跳过 prompt 对齐。
14. 修正版 case 9 自然 EOS，生成 2730 tokens，strict parser 得到 105 段、2 位 speaker，覆盖至
    299.92 秒。相对 HF 的 CER/cpCER/timestamp MAE 为 `0.0275/0.0196/0.1687 s`；相对 TextGrid 的
    QNN-HF 增量为 `-0.0014/+0.0098/+0.0927 s`，通过单 case `5%/5%/100 ms` 质量门禁。逐段语义审核
    结论为 `pass_with_recorded_differences`。RTF `1.219` 未通过 `<=1.0`，peak PSS `3.064 GB` 通过
    4 GiB 上限；因此 prompt 正确性根因关闭，但候选仍被质量、持续性能和 clean rebuild 复现性阻塞。
15. 修正版 case 2 仍在约 257 秒后退化为每秒重复一次“对。”，直到 298.42 秒；尾部关于充电宝、
    手机壳、贴膜和品牌限量赠品的讨论被实质遗漏。相对 HF 的 CER/cpCER/timestamp MAE 为
    `0.2060/0.1992/2.4483 s`；相对 TextGrid 的 QNN-HF 增量为
    `+0.0887/+0.1119/+3.2290 s`，三个质量门禁全部失败。RTF `1.824`、peak PSS `3.633 GB`。
    这证明 prompt 换行修复只关闭 case 9 的无时间戳根因，不能解释或修复 case 2 的 decoder 重复退化。
16. 同一 case 2 使用相同 W16 权重、tokenizer、prompt、输入和 greedy 协议在服务器 MNN CPU 完整自由
    生成：自然结束，3704 tokens、120 段、4 位 speaker，完整保留尾部赠品讨论；相对 HF 的
    CER/cpCER/timestamp MAE 为 `0.01618/0.01618/0.00883 s`，逐段语义审核通过。由此排除 source model、
    prompt、parser 和通用 MNN 生成协议，把 case 2 重复退化收敛到 QNN HTP decoder 数值路径或其 runtime
    集成。下一步应对齐退化前的 teacher-forced/free-generation logits 与 KV 状态，而不是修改 parser。
17. 已增加 late-step teacher probe 的窗口/稀疏采样能力，并用 HF 的 3733 个 reference IDs 在 CPU/QNN
    强制相同 token。每 100 step 的 34 点采样显示，同为“。→ [”的 segment boundary，logits cosine 从
    step 800 的 `0.9831` 降到 step 2200 的 `0.8928`，再降到 step 3200 的 `0.2653`。step 3160–3239
    连续 80 点的 top-1 agreement 为 `76.25%`、cosine mean/min 为 `0.9249/-0.0971`；最差 step 3221
    即使 CPU/QNN 仍选中相同 reference `[`，完整 logits 分布已经反向。这确认了随长 KV history 加重、
    在时间戳/segment boundary 最明显的 QNN decoder 数值退化。step 3200 的 MNN CPU debug callback 捕获
    196 个 layer tensors，但 external QNN context 不向该 callback 暴露内部节点，QNN 捕获数为 0；下一步
    必须导出 QNN context intermediate outputs 或构建 KV/attention boundary 的 split diagnostic wrapper，
    不能继续重复同一个 MNN layer callback。

当前 P0 TODO：

- [x] 将 decoder context 扩到 10240，并用动态 token-limit runner 通过 30 秒真机 smoke。
- [x] 冷机完整复跑 `R8007_M8010_0000`：自然 EOS、生成量超过 4096、PSS 不超过 4 GiB；语义覆盖核心
  信息但 CER/cpCER/timestamp 门禁失败，因此只关闭 token-budget 截断问题，不视为质量通过。
- [x] 冷机复跑 `R8009_M8019_0000`：仍为 1530 token、正文存在但无任何时间戳，确认不是热态偶发。
- [x] 对 `R8009_M8019_0000` 定位首个格式 token 分叉：根因为 runner prompt 少末尾换行，不是
  clean/retained context 数值漂移；32-token 修正版真机 A/B 已恢复时间戳且 prompt IDs 与 HF 一致。
- [x] 用修正 prompt 的 10240 runner 完整复跑 case 9：自然 EOS、schema、逐段语义和单 case 质量门禁
  通过；RTF `1.219` 未通过性能门禁。
- [x] 用修正 prompt 的 10240 runner 完整复跑历史失败 case 2：约 257 秒后仍重复“对。”，逐段语义审核
  和 CER/cpCER/timestamp 三项质量门禁均失败，确认它是独立于 prompt contract 的 decoder 退化。
- [x] 对 case 2 退化前执行相同 reference token 的 sparse + late-window teacher forcing，确认 step 3200/3221
  出现严重 logits 分布退化，并验证 external QNN context 无法通过现有 MNN debug callback 获取内部 layer tensors。
- [ ] 固定 host toolchain、环境变量和单 worker context build，使 clean rebuild 与保留候选完成
  180-step 对齐；当前 `98.33%` 不能沿用历史候选的 100% 结论。
- [ ] 上述正确性通过后，用 balanced/DCVS 重跑连续 10x300，并复核 CER/cpCER、时间戳、RTF、PSS
  和温升；未通过前保持 `production_ready=false`。

### 最新结论（覆盖下文历史快照）

1. W8 blockwise 权重进入旧 QNN 路径后被二次压成 per-channel int8，是逐层误差的第一根因。现在保留
   W8 权重文件不变，删除 MNN wrapper 的 activation quant metadata，让 QNN compiler 生成 FP16
   activation/weight constant graph。转换前后服务器 MNN CPU 180-step logits bit-identical。
2. 发布配置的 decoder `precision=low` 会让 QNN graph 外围的 MNN CPU 算子在 SM8850 上走 FP16；
   原 W8 CPU、W16 CPU 和新 W8 QNN 都会在 90/120 秒产生相同结构损坏。改为
   `precision=normal` 后，90/120 秒立即恢复为 HF reference 对应的 15/19 段。服务器 CPU 不支持
   FP16，历史 server baseline 因自动走 FP32 而没有暴露这个问题。
3. 新 W8 QNN 候选对服务器 CPU 48-step top-1 为 100%（`precision=low` 诊断）且 cosine mean
   `0.997645`；最终 `precision=normal` 下对同机 CPU 的 180-step top-1 为 `99.44%`、cosine mean
   `0.997432`。48-step 仍为 `97.92%`，唯一分叉在 step 3，且 cosine 为 `0.999713`。
4. manifest `73c245d9...` 的正式候选 release 在真机上 30/60/90/120 秒均 `status=ok`，片段数为
   4/10/15/19，RTF 为 `0.468/0.424/0.423/0.472`，peak PSS 为
   `546/577/838/1082 MB`。90 秒仅比 HF 少一个语气助词；
   120 秒有一处短语差异，未出现畸形时间戳、speaker ID 或重复退化。
5. QNN turbo 并非缺失。当前 external context 已使用 burst、关闭 DCVS、禁止 DSP sleep，并把
   bus/core corner 固定到最大值。正确性问题与 turbo 无关。
6. W16A16 最初在 `precision=low` 下与 Android CPU 同样于 90 秒失败，因此曾被误判为无效路线。
   改为 `precision=normal` 后，90/120 秒恢复为 15/19 段；同机 CPU/QNN 180-step top-1 为 100%，
   cosine mean/min 为 `0.999998/0.999963`，对原 W8 server CPU 的 180-step top-1 也是 100%。
   W16A16 因此取代 W8 pure-FP16 成为正确性优先的最终候选。
7. W16A16 manifest `12831c6f...` 的单例 300 秒真机回归通过：54 段、4 位 speaker、覆盖到
   299.92 秒，RTF `0.951`，peak PSS `2,661,988,352 B`。逐段对照 HF FP32 300 秒 reference
   后语义通过并保留一处数字短语差异，不做后处理。

本轮设备证据结论：prompt contract 完全一致，四个 audio chunk 的 QNN HTP front/end-to-end
cosine 均通过 `0.995` 门禁；剩余首个阻塞已收敛到 decoder QNN HTP 数值路径。48-step
teacher forcing 的 top-1 agreement 仅 `93.75%`，logits cosine mean/min 为
`0.955826/0.685391`，首个 top-1 分叉发生在 step 3。后续真机消融证明误差从 layer 0 开始逐层累积；
关闭 relaxed FP16 可把首个分叉从 step 3 推迟到 step 21，但 48-step cosine mean/min 仍只有
`0.953352/0.716918`。曾被标记为“strict FP16 + graph30 CPU”的 `0.995790/0.974054`
结果经 wrapper SHA 复核确认实际运行的是完整 MNN CPU decoder，不能作为混合分区证据；真实的后段
CPU 分区候选均在 step 0 无 logits。继续修复已转向 QNN 原生支持的 W4 blockwise 权重路径。

本文是拿到 SM8850 真机后的执行清单。根因、历史结果与完整门禁分别见
[`moss-sm8850-device-retrospective-20260819.md`](moss-sm8850-device-retrospective-20260819.md) 和
[`moss-transcribe-diarize-todo.md`](moss-transcribe-diarize-todo.md)。

## 固定资产

后续实验必须复用下列文件，不得临时更换输入、模型或 reference：

| 资产 | 路径 | SHA-256 / 状态 |
| --- | --- | --- |
| 候选 payload | `/data/experiments/meetnote-moss-contract-v2/release` | manifest `76fb7a57de9972c7314c1716798f7cfc501bf54e91c21e46c5af9f437ab69834`，55 files，`valid` |
| Android runner | payload 内 `moss_qnn_runner` | `61e2e3d441cc408a18b3a3a24d5199f9e9b39a93e1bdb3c3e719153eaed0ec0e` |
| 回归输入 | `/data/experiments/meetnote-moss-qnn-poc-v1/inputs/regression-contract-v2` | manifest `636a90abd52ec431fe05dd65f80a28287012478c42e6277b70a1f5a7c968ded4` |
| HF BF16 reference | `/data/experiments/meetnote-moss-contract-v2/hf-bf16-clean-reference` | manifest `fa7605a18b2053ca46906c3b52969744f9b3e40b1169a7e4f0a870fd2c652aab` |
| MNN CPU reference | `/data/experiments/meetnote-moss-contract-v2/mnn-cpu-reference-comparison.json` | `7cf344020e4fafdd07f9bb663014bfa4778e25acf4fd9f5b6e09942de51b3515` |
| Android teacher probe | `/data/experiments/meetnote-moss-contract-v2/android-diagnostics/moss_teacher_forced_runner` | `365bc7d70c90f84dfa5f1950190d6fba65b2950926def2466a3c7d2f7b5c8825` |
| MNN teacher prefix | `/data/experiments/meetnote-moss-contract-v2/teacher/moss-mnn` | `.jsonl` `f0af54c9...`；`.f32` `909a8223...` |

主机侧已证明：HF prompt 与 MNN prompt IDs 完全一致；四个 audio chunk 的最终 embedding cosine 为
`0.9999067`；HF BF16 与 MNN CPU 48-step teacher forcing 的 top-1 agreement 为 100%，logits cosine
mean/min 为 `0.997719/0.985197`。剩余结论必须来自 SM8850 HTP，Linux CPU 结果不能替代。

## 0. 设备前置检查

- [x] USB 连接、解锁并接受 debugging 授权。
- [x] `adb devices -l` 只列出目标设备，状态为 `device`，不能是 `offline` 或 `unauthorized`。
- [x] 设备报告 `arm64-v8a`、`ro.soc.model` 包含 `SM8850`；运行脚本会再次强制检查。
- [x] 确认证据输出根目录不存在，避免覆盖历史结果。

```bash
ADB=/data/experiments/meetnote-moss-toolchain/install/platform-tools/adb
"$ADB" devices -l
"$ADB" shell getprop ro.product.cpu.abi
"$ADB" shell getprop ro.soc.model
```

## 1. 修正 payload 的首轮回归（P0）

严格按 30 → 60 → 90 → 120 秒执行；任一档命令失败或质量不通过即停止，不扩大时长。每次使用新的
`MOSS_EVIDENCE_DIR`，脚本会保存设备信息、run log、result JSON、prompt alignment 以及输入、payload、
runner、日志和结果的 SHA-256。

```bash
ADB=/data/experiments/meetnote-moss-toolchain/install/platform-tools/adb
RELEASE=/data/experiments/meetnote-moss-contract-v2/release
INPUT_DIR=/data/experiments/meetnote-moss-qnn-poc-v1/inputs/regression-contract-v2
EVIDENCE_ROOT=/data/experiments/meetnote-moss-contract-v2/device-regression

for DURATION_SECONDS in 30 60 90 120; do
  WAV="$INPUT_DIR/R8001_M8004_0000_300s_${DURATION_SECONDS}s.wav"
  MOSS_EVIDENCE_DIR="$EVIDENCE_ROOT/${DURATION_SECONDS}s" ADB="$ADB" \
    tools/run_moss_sm8850.sh "$RELEASE" "$WAV"
done
```

每档必须同时满足：

- [ ] 进程退出码为 0，`result.json` 为 `status=ok`，raw transcript 被严格 parser 完整消费。
- [ ] `prompt-alignment.json` 为 `valid`，设备 `prompt_token_ids` 与内置 HF contract 逐项一致。
- [ ] 30/60/90/120 秒 prompt 长度分别为 472/859/1246/1638；120 秒的完整 prompt 同时覆盖 60 秒
  边界和 100 秒三位数 marker。
- [ ] segment start 非递减、每段 `start < end`、范围合法且末尾覆盖合理；允许不同 speaker 的合法
  重叠区间。speaker ID 均为两位，不能出现 `S0`。
- [ ] 无重复 token/片段退化，并由人工或独立 LLM 对照完整音频审核语义。
- [ ] 单独确认 30 秒合法重叠不会被误判为逆序；单独确认 60 秒越过旧约 59 秒故障边界。

判定分支：

1. prompt IDs 不一致：只修 processor/export/runtime contract，不进入 QNN 数值分析。
2. prompt IDs 一致且四档质量全部通过：记录“旧 prompt contract 为主根因”，进入第 3 节。
3. prompt IDs 一致但任一档失败：保留首个失败档全部证据，进入第 2 节，不运行 300 秒。

### 2026-08-19 最终执行结果

- [x] 30/60/90/120 秒 prompt 长度分别为 472/859/1246/1638，contract 一致。
- [x] `precision=normal` 的 W8 pure-FP16 QNN 候选四档均退出码 0、`status=ok`，strict parser
  完整消费 raw transcript。
- [x] 四档片段数为 4/10/15/19；90/120 秒与 HF reference 的 speaker 数和片段数一致。
- [x] 候选 release 的 55 文件 manifest 校验通过，manifest SHA-256 为 `73c245d9...`；四档设备
  prompt alignment 均为 `valid`。
- [x] 语义审核已逐段对照完整 HF transcript：30 秒只合并同 speaker 相邻片段；60 秒文本一致；
  90 秒少一个“的”；120 秒有“大家都挺有”与“大家都同意这个”的短语差异，其余内容、speaker
  和边界一致。该差异保留为质量证据，不做字符串后处理。
- [x] W16A16 `precision=normal` 同机 CPU/QNN 48/180-step top-1 均为 100%，通过数值门禁。
- [ ] 执行一次 clean end-to-end rebuild，确认重新导出的 context/manifest 与本候选一致。
- [x] 单例 300 秒通过：`status=ok`、RTF `0.951`、peak PSS `2.66 GB`、无 SSR/OOM/crash，
  semantic audit 为 `pass_with_recorded_differences`。
- [ ] 继续 AliMeeting 10x300 秒质量与连续稳定性门禁。

以下为修复前的首轮历史结果：

- [x] 30 秒进程退出码为 0，`status=ok`；prompt alignment 为 `valid`，472 个 token 完全一致。
- [x] 保留 30 秒 `result.json`、run log、设备信息与全部 SHA-256。
- [x] 复核 HF reference 后确认 `22.51 -> 13.24` 表示 S01/S02 重叠说话，不是非法逆序；旧门禁
  将 raw marker 顺序误当成 segment start 顺序，判定已纠正。
- [x] 按门禁在首个失败档停止；60/90/120 秒首轮回归及 300 秒均未运行。

这对应判定分支 3，不得把 parser 的 `status=ok` 当作语义/时间戳质量通过。

## 2. 失败时的 HTP 定位（条件执行）

- [x] 已对固定 30 秒 case 执行 48-step `MNN CPU -> QNN HTP` teacher forcing：

```bash
ADB=/data/experiments/meetnote-moss-toolchain/install/platform-tools/adb
tools/run_moss_teacher_forced_sm8850.sh \
  /data/experiments/meetnote-moss-contract-v2/release \
  /data/experiments/meetnote-moss-qnn-poc-v1/inputs/regression-contract-v2/R8001_M8004_0000_300s_30s.wav \
  /data/experiments/meetnote-moss-contract-v2/android-diagnostics/moss_teacher_forced_runner \
  fixtures/moss/default-transcription-prompt.txt \
  /data/experiments/meetnote-moss-contract-v2/teacher/moss-reference-ids.txt \
  /data/experiments/meetnote-moss-contract-v2/teacher/moss-hf-token-ids.json \
  /data/experiments/meetnote-moss-contract-v2/teacher/moss-mnn \
  /data/experiments/meetnote-moss-contract-v2/device-teacher-30s
```

- [x] 数值门禁失败已确认：token IDs 完全一致且无 saturation，但 top-1 agreement 为 `93.75%`，
  cosine mean/min 为 `0.955826/0.685391`，低于 99% / 0.98。
- [x] 记录首个异常 sequence position/decode step、C64 prefill 边界、KV length、top-1、cosine 与 max
  absolute error；禁止比较不同后端自由生成出的不同 token。
- [x] 补建并运行真机 audio component probe，对四个 30 秒 chunk 分别采集 QNN HTP audio front/back；
  与已固定的 MNN CPU 张量比较。最终 embedding cosine 门禁 ≥ 0.995。仓库已新增运行链路 tensor dump
  patch、真机执行和 tensor 比较工具。
- [x] 已按 layer 边界、prefill chunk、graph30/lm_head 和 context compile precision 做单变量消融，
  未用自由生成 token 做跨后端比较。
- [x] 只修复证据定位出的首个根因。若分叉在 prefill，按 chunk/layer 二分；若 teacher forcing 一致而
  自由生成分叉，检查首个 top-1 不一致步的 logits margin、KV cache 更新和 sampler 输入。

首个分叉为 decode step 3：MNN CPU top token `20`，QNN top token `23`，该位置 CPU top-1 margin
仅约 `0.0118`，QNN 将相对 margin 翻转为 `0.15625`。四个 audio chunk 的 front cosine 分别为
`0.999871/0.999985/0.999957/0.999903`，end-to-end cosine 为
`0.999917/0.999963/0.999985/0.999970`，均通过。由此排除 prompt 和 audio encoder 主因。

后续已找到完整 QAIRT `2.48.40.260702` host SDK，并完成以下真机消融：

| 消融 | top-1 agreement | cosine mean/min | 结论 |
| --- | ---: | ---: | --- |
| C1 prefill | 95.83% | `0.954931/0.704432` | C64 prefill/KV 初始化不是主因 |
| graph30/lm_head 回 CPU | 95.83% | `0.956622/0.682600` | final head 不是唯一根因 |
| relaxed FP16 关闭 | 97.92% | `0.953352/0.716918` | 首个分叉推迟到 step 21，relaxed FP16 是重要因素 |
| 误标的“strict FP16 + graph30 CPU”（实际完整 MNN CPU） | 97.92% | `0.995790/0.974054` | wrapper SHA 为原始 CPU 模型；只证明 Android CPU 与服务器 CPU 接近，不证明 QNN 混合分区 |
| 上述完整 MNN CPU，180 steps | 98.33% | `0.995854/0.949801` | 同为 CPU 诊断，不计入 QNN 晋级门禁 |

逐层 hidden 对齐在 layer `0/7/15/23/27` 的 cosine mean 分别为
`0.999513/0.994138/0.985792/0.964118/0.965504`，证明误差沿 decoder 深度累积，而不是单个
graph30 故障。把 layer 8 以后或 layer 27 及输出头整体回 CPU 的真实 QNN plugin wrapper 都在 step 0
无 logits，属于无效分区，未作为质量候选。每次组合实验后，设备上的发布 decoder 已恢复为 SHA-256
`d387826b9ae6fe09cf5cbc6bfd8dac1a6e31944d4db32d858a419a6dba3b4d9d`。

误标目录中的 48/180-step 结果保留用于审计，但 wrapper SHA
`984e846e...` 对应完整 MNN CPU 模型，不是生成目录内的 QNN plugin wrapper。后续设备实验必须同时
记录并核对“下发源文件 SHA、设备文件 SHA、候选 QNN wrapper SHA”，避免只凭目录名判定实际后端。

## 3. 晋级门禁（首轮全部通过后）

- [x] 用同一来源运行单例 300 秒；要求可解析、时间戳/说话人合法、语义通过，RTF ≤ 1.0、peak PSS ≤ 4 GiB。
- [x] 运行 AliMeeting 10×300 秒：CER/cpCER 相对 HF BF16 退化 ≤ 5%，timestamp boundary MAE 增量
  ≤ 100 ms，并记录 missed/merged speakers。
- [x] 已归档连续 10 个 case 的 logcat、thermal、battery、PSS 与 result JSON；无 HTP SSR、OOM、
  crash 或 non-finite，但 2 个 case 返回结构化业务错误，因此稳定性仅代表 runtime 未崩溃。
- [ ] 正确性通过后才做 burst/balanced A/B；不得用 performance profile、parser 容错、repetition penalty
  或字符串后处理掩盖质量问题。
- [ ] 验证 plugin/context 损坏与 DSP SSR 均失败关闭并保留结构化错误。
- [ ] 全部门禁通过后更新 validation/provenance 和 `production_ready`，再进入产品集成。

## 需要负责人决策的边界

当前不需要产品或架构决策，先取得真机证据。仅在以下情况停下请求决策：

- 修正 prompt 后仍出现已定位且需改变模型、量化方案、CPU/HTP 切分或支持机型范围的数值问题；
- 质量门禁与延迟、内存、温升/功耗目标无法同时满足，需要选择产品取舍；
- AliMeeting 门禁失败，需要决定继续投入、限制功能/机型，或终止该 PoC。

W4A16 CPU 侧 180-step 对 W8 只有 `91.67%` top-1，已否决。W8 pure-FP16 修复了原逐层误差并通过
四档自由生成，但 48-step 仍有一个低 margin top-1 分叉，保留为体积更小的备选。W16A16 在
`precision=normal` 下通过同机 CPU/QNN 100% top-1、四档自由生成和单例 300 秒门禁，现为默认
正确性候选。不得用时间戳后处理、sampler 或降低阈值收尾。

## 2026-08-20 10240-token 收敛状态

- [x] 以 `max_history_token=10240`、`max_all_tokens=10240`、`max_new_tokens=6144` 完成 W16A16
  decoder context 和完整 runtime payload 构建；55 个 payload 文件通过静态校验。
- [x] 定位 runner 的两个独立硬编码：`set_config(...8192/4096)` 与
  `response(..., 4096)`。仅删除 `set_config` 或改用默认参数会让 MNN multimodal 首次 forward
  在 `deq/graph0.bin` 报 Plugin shape inference failure。
- [x] 修复为从 `config.json` 读取两个 token limit，继续显式调用 `set_config`，并把解析出的
  `max_new_tokens` 显式传给四参数 `response`。同一 runner 已在真机分别通过 8192/4096 与
  10240/6144 的 30 秒 smoke，均为 `status=ok`、472 prompt tokens、147 generated tokens。
- [x] 10240/6144 的 case 5 在稳定连接下重新冷机复跑并自然 EOS；生成 4552 tokens，修复旧 4096
  截断。CER/cpCER/timestamp boundary MAE 与 RTF 仍未过完整门禁，不能因末尾覆盖而晋级。
- [ ] case 2 重复退化和 case 9 无时间戳的正确性根因均已关闭；CER/cpCER/timestamp 全量门禁、RTF
  和 clean rebuild 复现性仍是独立 blocker，`production_ready` 保持 `false`。

性能侧结论不变：当前 context 使用的 `burst` 已是 QNN 的最高性能档，相当于所问的 turbo；它关闭
DCVS、禁止 DSP sleep 并投票最高 bus/core corner。连续 300 秒运行会升温并降频，因此只适合冷机
诊断，不适合作为持续产品默认。正确性修复后再做 balanced/DCVS A/B。

## 2026-08-24 case 2 长 KV 与 strict FP16 结论

- [x] case 2 同 token late-window/sparse teacher forcing 已证明误差随 KV history 加重：相同的
  `。 -> [` segment boundary 在 step 800/2200/3200 的 CPU/QNN cosine 为
  `0.98313/0.89276/0.26533`；step 3160-3239 连续窗口 cosine min 为 `-0.09710`。
- [x] MNN CPU step 3200 捕获 196 个 layer tensor；external QNN context 对 MNN callback 不透明，捕获
  0 个内部 tensor。继续重复 MNN callback 不能定位 QNN 首个坏层。
- [x] 临时 strict10240 全量构建的 `compilefornpu` 崩溃已用 GDB 定位。`[1,64,1024]` null-host
  constant 是漏掉 CPU partitions 后的症状：临时脚本绕过了 `generate_qnn_with_cpu_ops.py`，没有保留
  Slice、RoPE 和 28 层 FusedAttention。不得把 null constant 静默改成 QNN native tensor；该 fallback
  只会把崩溃推迟到跨 partition `inputIO`。
- [x] 复用冻结 stage2 cache 并只以 `fp16_relaxed_precision=0` 重跑 phase3，31 个 context 与当前
  relaxed context 逐字节相同；该 flag 对当前 CPU-attention/RoPE hybrid graph 不产生独立变量。
- [x] 将 2026-08-19 已验证且 context SHA 确实不同的 archived strict decoder 下发真机，设备侧 32 个
  decoder 文件 SHA 全部通过。同一 3240-step case 2 trajectory 每 100 step 采样后，archived strict
  与当前 relaxed 的 34 组 logits 逐字节相同：top-1 `100%`、cosine `1.0`、RMSE `0`。相对 MNN CPU
  仍为 top-1 `91.18%`、cosine mean/min `0.96180/0.26533`，首个采样分叉仍在 step 2000。
- [x] strict FP16 已排除为 case 2 长 history 修复方向，不再运行该候选的完整自由生成。下一步必须暴露
  QNN intermediate outputs，或构建 KV/attention split diagnostic wrapper；在根因修复前仍不进入连续
  10x300。
- [x] QNN intermediate output wrapper 已在真机可执行。首版 prefill shape failure 来自 generator 将 IO
  seq len 硬编码为 128、与 `--chunk_size 64` 冲突；`generate_qnn_with_cpu_ops.py` 现强制 IO 使用请求的
  chunk size。Android probe 必须用 release 对应的 pinned MNN headers 构建；当前 checkout headers 与
  release 的 `LlmContext` ABI 不同，会在 `history_tokens.push_back()` 崩溃。
- [x] case 2 step 3200 的 `q_proj` 粗粒度 layer 0/7/15/23/27 cosine 为
  `0.99999998/0.99999079/0.86874851/0.94789329/0.96672953`；细化 layer 8-15 后为
  `0.99997348/0.99988042/0.99982221/0.99457642/0.98502276/0.99186913/0.93239109/0.86874851`。
  首个明显放大区间是 layer 10 输出进入 layer 11。复用相同 graph 回收 hidden 和 attention Q/K/V 后，
  layer 10 的 cosine 仍为 `0.99985510/0.99985127/0.99997046/0.99913196`，layer 11 已降至
  `0.99779332/0.99012331/0.98463079/0.97515157`。首次显著误差注入进一步收窄到 layer 10
  FusedAttention 输出或其后的 residual/MLP；不能仅凭 q_proj 输出归因于 q_proj 权重量化。
- [x] FusedAttention trajectory 已补齐。layer 10 attention output 在 step 0/800/2200/3200 的 cosine 为
  `0.933289/0.999222/0.996584/0.979017`，最终 logits 为
  `0.970698/0.983131/0.892760/0.265332`；layer 10 当前 Q/K/V 在 late step 仍接近 CPU，但 attention
  输出随 history 恶化，累计 KV cache 或长序列 attention 敏感性成为更强假设。
- [x] 已纠正一次关键实验组装错误：声称 step 0/3200 恢复到 `0.99999922/0.99999016` 的 release
  误装了原始 504072-byte 完整 CPU `decoder/llm.mnn`，没有使用约 262 KiB 的 hybrid wrapper；其
  `deq/graph*.bin` 未参与 decoder forward。该结果只能作为 CPU control，先前 layer 10 因果结论作废。
- [x] 正确 hybrid wrapper 中，partition 后的 attention reshape 被折叠为 `[1,64,heads,128]`，单 token
  decode 会报 `2048 -> 131072`。改为 `[1,-1,heads,128]` 后各拆分 wrapper 均可执行；Q-only 与
  K/V-only 的 step 3200 logits cosine 为 `0.26390/0.26521`，同 baseline `0.26533` 一致，不能把根因
  归到 layer 10 某个 projection。
- [x] base hybrid 的 graph 1-28 四个 plugin outputs 已全部接入真机 trace。step 0 的 layer 0 当前
  Q/K/V cosine 均大于 `0.999999`，attention output 已为 `0.994109`，layer 2 降至 `0.901135`；
  step 3200 在 layer 8/9 尚为 `0.998055/0.998514`，layer 10 降至 `0.979017` 后继续扩大。放大层随
  history 改变，现有证据支持跨层 QNN 数值误差与 CPU attention/KV 状态的分布式累积，而非单层故障。
- [x] prefill layer 0 的逐 chunk 对齐在 token offset `2048` 找到精确断点：chunk 0-31 的 Q/K/V 与
  attention output cosine 均接近 `1.0`，chunk 32 的 Q/attention cosine 突降到
  `0.980858/0.972591`。原 partition 只把 RoPE Mul/Cos/Sin 留在 CPU，上游 Cast/Reshape 仍让 position ID
  先经过 FP16；超过 2048 后已不能逐整数精确表示。
- [x] 将 `/rotary/Cast_output_0`、`/rotary/Reshape_output_0` 与 Mul/Cos/Sin 一并留在 CPU 后，63 个
  prefill chunks 的 Q/K/V/attention output 最低 cosine 恢复至
  `0.99999989/0.99999990/0.99999995/0.99999964`；case 2 step 0/3200 最终 logits cosine 为
  `0.99999887/0.99998177`，top-20 完全一致。正式 rebuild 脚本已加入该分区边界。
- [x] 修复版 3240-step、每 100 step 采样 trajectory 共 34 点，QNN 对 MNN CPU top-1 为 `100%`，
  logits cosine mean/min 为 `0.99998193/0.99978399`，没有分叉。
- [x] case 2 自由生成自然结束于 `300.00 s`：3702 tokens、120 段、4 位 speaker；相对 W16 MNN CPU
  的字母数字归一化文本完全一致，CER/cpCER 为 `0/0`、timestamp MAE `0.0055 s`；相对 HF BF16 的
  CER/cpCER 为 `0.01703/0.01703`、timestamp MAE `0.0050 s`。完整语义审核通过，先前约 257 秒后的
  重复已消失，充电宝、手机壳、贴膜和品牌限量赠品讨论全部保留。
- [ ] 正确性根因已关闭，但 RTF `1.8753` 仍失败；peak PSS `3,488,475,136 bytes` 通过。先做
  performance profile 与 CPU RoPE 开销 A/B，再决定 continuous 10x300，不能把正确性修复误报为整体晋级。

本轮证据：

- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/late-step-probe/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/archived-strict-sparse100/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/strict10240-v4-gdb/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/strict10240-v7-phase3-clean-output/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/qproj-debug-chunk64-step3200/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/qproj-debug-8-15-chunk64-step3200/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/qproj-debug-8-15-boundaries-step3200/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/attn-out-step{0,800,2200,3200}/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/all-layer-qkv-step{0,3200}/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/prefill-layer0-{qnn,cpu,rope-position-cpu}/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/prefill-layer0-cpu-vs-{qnn,rope-position-cpu}.json`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/rope-position-cpu-step{0,3200}/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/rope-position-cpu-sparse100/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/qnn-rope-position-cpu-case2-full/`
- `/data/experiments/meetnote-moss-contract-v2/case2-newline-20260824/cpu-attention-input-layer10-step{0,3200}/`
  （误组装的完整 CPU control，不得作为 hybrid A/B）

## 本轮证据位置

- 首轮 30 秒：`/data/experiments/meetnote-moss-contract-v2/device-regression/30s`
- 48-step teacher forcing：`/data/experiments/meetnote-moss-contract-v2/device-teacher-30s`
- 四 chunk audio tensors：`/data/experiments/meetnote-moss-contract-v2/device-audio-dump-120s`
- C1 prefill：`/data/experiments/meetnote-moss-contract-v2/device-teacher-c1-prefill-30s`
- graph30/lm_head CPU：`/data/experiments/meetnote-moss-contract-v2/device-teacher-graph30-cpu-30s`
- decoder layer sweep：`/data/experiments/meetnote-moss-contract-v2/device-teacher-layer-sweep-30s`
- strict FP16：`/data/experiments/meetnote-moss-contract-v2/device-teacher-strict-fp16-30s`
- 误标为 strict FP16 + graph30/lm_head CPU、实际为完整 MNN CPU：
  `/data/experiments/meetnote-moss-contract-v2/device-teacher-strict-fp16-graph30-cpu-30s`
- 上述完整 MNN CPU，180 steps：
  `/data/experiments/meetnote-moss-contract-v2/device-teacher-strict-fp16-graph30-cpu-30s-181`
- 真实 strict FP16 + layer27/output CPU，无 logits：
  `/data/experiments/meetnote-moss-contract-v2/device-teacher-strict-layer27-cpu-30s`
- W4A16 CPU 180-step（下发前质量门禁失败）：
  `/data/experiments/meetnote-moss-contract-v2/w4a16-strict-v3-20260819/teacher-cpu-180`
- W16A16 CPU 180-step：
  `/data/experiments/meetnote-moss-contract-v2/w16a16-strict-v1-20260819/teacher-cpu-180`
- W8 pure-FP16 QNN context：
  `/data/experiments/meetnote-moss-contract-v2/w8-pure-fp16-graph-v1-20260819`
- W8 pure-FP16 48/180-step 真机证据：
  `build/device-validation/20260819-moss-qnn-sm8850/contract-v2/w8-pure-fp16-graph-v1/device-teacher-*`
- `precision=normal` 四档自由生成：
  `build/device-validation/20260819-moss-qnn-sm8850/contract-v2/w8-pure-fp16-graph-v1/device-freegen-normal-v1`
- 服务器归档的同一批真机证据：
  `/data/experiments/meetnote-moss-contract-v2/device-investigation-20260819/w8-pure-fp16-normal`
- W16A16 最终候选 release：
  `/data/experiments/meetnote-moss-contract-v2/release-w16a16-normal-v1`
- W16A16 数值、四档、300 秒和 HF 300 秒 reference：
  `/data/experiments/meetnote-moss-contract-v2/device-investigation-20260819/w16a16-normal`
- 2026-08-20 clean rebuild：
  `/data/experiments/meetnote-moss-contract-v2/clean-rebuild-w16-20260820/work-v4/release`
- 2026-08-20 AliMeeting 10x300 burst 设备证据：
  `/data/experiments/meetnote-moss-contract-v2/device-investigation-20260819/w16a16-normal/alimeeting-10x300-burst`
- 2026-08-20 case 9 冷机复跑：
  `build/device-validation/20260820-moss-qnn-sm8850/w16a16-normal-case9-cold-rerun`
- 2026-08-20 HF BF16 reference：
  `/data/experiments/meetnote-moss-contract-v2/hf-bf16-alimeeting-10x300-final`
- 2026-08-20 10240-token 重建实验：
  `/data/experiments/meetnote-moss-contract-v2/history10240-v2-20260820`
- 10240 runner 修复后的 8192 与 10240 短音频真机 smoke：
  `build/device-validation/20260820-moss-qnn-sm8850/{clean-w16-runner-v9-30s,history10240-runner-v9-30s}`
- 被连接中断的 10240 case 5 长测目录（空 result，不得作为通过证据）：
  `build/device-validation/20260820-moss-qnn-sm8850/history10240-runner-v9-case5-cold`

非交互远端构建必须显式提供 host clang runtime，例如：

```bash
HOST_CLANG_BIN=/data/experiments/meetnote-moss-qnn-poc-v1/clang15/root/usr/lib/llvm-15/bin
QNN_HOST_LIB_DIR=/data/experiments/meetnote-moss-qnn-poc-v1/clang15/root/usr/lib/x86_64-linux-gnu
```

只把 Android NDK 的 `clang++` 放入 PATH 不够；QAIRT host binary 还依赖 `libc++.so.1` 和
`libunwind.so.1`。decoder context build 继续固定 `MNN_QNN_MERGE_WORKERS=1`，避免 QAIRT 临时目录冲突。
- 设备、温度、电量、logcat 与汇总：
  `/data/experiments/meetnote-moss-contract-v2/device-investigation-20260819`

任何降低既定正确性或质量门禁的做法都属于显式产品决策，不能作为工程侧默认收尾。
