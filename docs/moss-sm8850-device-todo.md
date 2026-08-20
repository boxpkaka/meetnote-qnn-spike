# MOSS SM8850 真机 TODO（交接快照）

状态：`BLOCKED_CORRECTNESS`（2026-08-20）。已在 Vivo V2505A / SM8850 /
Android 16 修复原 decoder QNN 累积误差和长序列损坏，30/60/90/120 秒实验候选均可完整解析；
候选 release 已按 55 文件 manifest 组装并完成四档真机回归，但仍不可标记为 `production_ready`，
W16A16 候选已通过历史 48/180-step 数值门禁和单例 300 秒门禁。clean rebuild 和
AliMeeting 10x300 秒已执行，但重建复现性与长音频质量门禁均失败。

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

当前 P0 TODO：

- [x] 将 decoder context 扩到 10240，并用动态 token-limit runner 通过 30 秒真机 smoke。
- [ ] 冷机完整复跑 `R8007_M8010_0000`，确认自然 EOS、生成量超过 4096、schema/语义通过且
  PSS 不超过 4 GiB；本轮连接中断的空结果不计入验收。
- [x] 冷机复跑 `R8009_M8019_0000`：仍为 1530 token、正文存在但无任何时间戳，确认不是热态偶发。
- [ ] 对 `R8009_M8019_0000` 按首个格式 token 分叉做 teacher forcing，定位 clean/retained context
  数值漂移或自由生成稳定性，而不是放宽 parser。
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
- [ ] 只修复证据定位出的首个根因。若分叉在 prefill，按 chunk/layer 二分；若 teacher forcing 一致而
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
- [ ] 10240/6144 的 case 5 长测在约 8 分钟时因设备和开发机的 ADB/网络连接同时丢失而中断，
  runner 未产出最终 JSON；不能据此宣称 4096-token 截断已解决。恢复设备后只需重跑这一条，
  验收生成 token 大于 4096、自然 EOS、完整时间戳 schema 和逐条语义审核。
- [ ] 即使 case 5 通过，case 2 重复退化、case 9 无时间戳、CER/cpCER/timestamp 全量门禁以及
  clean rebuild 数值不可复现仍是独立 blocker，`production_ready` 保持 `false`。

性能侧结论不变：当前 context 使用的 `burst` 已是 QNN 的最高性能档，相当于所问的 turbo；它关闭
DCVS、禁止 DSP sleep 并投票最高 bus/core corner。连续 300 秒运行会升温并降频，因此只适合冷机
诊断，不适合作为持续产品默认。正确性修复后再做 balanced/DCVS A/B。

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
