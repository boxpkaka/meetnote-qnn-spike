# MOSS Transcribe-Diarize TODO

> 真机恢复执行请从 [`moss-sm8850-device-todo.md`](moss-sm8850-device-todo.md) 开始；本文保留完整问题树、
> 验证依据与晋级门禁。

## 当前判断

- Release payload 已能在 Vivo V2505A（SM8850、Android 16）稳定加载和执行；可运行性不再是主阻塞。
- 30 秒结果 `status=ok`、RTF 0.487，但存在时间戳逆序，只能作为 smoke test，不能算质量通过。
- 120 秒结果 RTF 0.274，约 59 秒后出现重复 token、畸形时间戳和非法 `S0`，严格 parser 正确返回
  `transcript_parse_failed`，因此不是 parser、schema 或设备算力问题。
- clean 30/60/90/120 秒 HF BF16 reference 已补齐；120 秒为 19 段、4 位说话人、覆盖到 119.98 秒，
  没有 59 秒后退化。它确认旧真机失败不是 clean 输入本身或模型长音频能力导致。
- 120 秒四个 30 秒 chunk 的 HF FP32 -> MNN CPU audio 链路已全部通过：拼接 log-mel cosine
  `0.999996`、MAE `0.000586`，最终 embedding cosine `0.999907`；主机 frontend/export 不再是阻塞，
  下一步只需补 QNN HTP component 对齐。
- 修正 prompt 后，MNN CPU W8 自由生成 120 秒得到 19 段、4 位说话人并覆盖到 119.98 秒，与 HF BF16
  transcript 字符相似度 `0.998496`，未复现 59 秒后退化。根因范围已缩小到旧 prompt contract 和/或
  QNN HTP decoder；MNN CPU decoder 不再是主要嫌疑。
- 静态 contract 核对已发现失败 payload 的 runtime 每 2 秒插入 marker，且遗漏 audio start/end token；
  固定 HF processor 的权威配置是每 5 秒插入 marker，并包含 151669/151670 boundary token。先验证该
  确定性输入差异是否为主根因，再进入 QNN decoder 数值排查。
- 进一步的完整 prompt 对比发现旧 `tokenizer.mtok` 没有导出独立的 `chat_template.jinja`，真机实际还
  丢失了 19 个 system/user/assistant 框架 token。修复后 120 秒 prompt 从 1619 恢复为 HF 基准 1638；
  旧 probe 因两侧共用同一坏模板而出现循环验证，现已增加独立 HF token 长度与 SHA 门禁。
- 当前已使用 burst/max voltage corner。正确性通过前不调 performance profile，也不使用 parser 容错、
  repetition penalty 或字符串后处理掩盖错误。

## P0：定位并修复质量阻塞

按以下顺序执行；前一阶段未通过时，不进入后续长时质量或性能测试。

### 1. 固化可复现基线

- [ ] 为本轮 30 秒和 120 秒输入、release manifest、runner、raw output、result JSON 与完整运行日志记录
  SHA-256；补齐 generated token 数和 audio front、audio back、prefill、decode 分阶段耗时。
- [x] runner 输出完整 prompt token IDs，`run_moss_sm8850.sh` 在设置 `MOSS_EVIDENCE_DIR` 时自动保存
  run log、result JSON、设备信息及 input/payload/runner/log/result SHA-256；crash 时也保留无 result 的证据。
- [x] 从验收源生成固定 30/60/90/120 秒 clean 前缀和 manifest；120 秒 SHA `80cf2405...` 与现有
  HF case 记录的 clean source 一致。
- [x] 直接使用上述 clean WAV 补跑 HF BF16 reference，记录完整 raw transcript、segment 数、speaker 数和
  末尾时间戳。30/60/90/120 秒分别为 6/10/15/19 段，末尾 29.94/59.88/89.29/119.98 秒；reference
  manifest SHA 为 `fa7605a1...`。后续对齐与回归不得更换输入或 reference。
- [x] 增加由同一 120 秒音频截取的 60 秒和 90 秒用例。60 秒用于复现约 59 秒边界，90 秒用于确认问题
  是否在跨过边界后持续存在。

完成标准：任一同版本重复运行都能根据输入、payload 和日志 SHA 唯一复现实验，且能判断首个坏 token
所在的 decode step。

### 2. 先排除 processor 与多 chunk 拼接错误

- [x] 对照固定模型 `processor_config.json`、`config.json` 和 `tokenizer.json`，定位 marker interval 与
  audio boundary token 漂移；修复导出/runtime 补丁，并增加组装和 payload 校验门禁。
- [x] 新增 `verify_moss_audio_token_contract.py`，输出 30/60/90/120 秒 audio span 的 embedding 数、
  marker token 和稳定 SHA-256；旧 export 已被确认缺少 151669/151670 boundary token。
- [x] 修复 MNN `.mtok` exporter：当 `tokenizer_config.json` 不内嵌模板时读取
  `chat_template.jinja`；新完整 prompt 30/60/90/120 秒分别为 472/859/1246/1638 token，与独立 HF
  基准的 token 数及 ID SHA 完全一致。
- [x] 复用未变化的 QNN context/权重，更新 Android runtime payload；候选包包含四档 prompt contract，
  共 55 个文件，完整性与 contract verifier 均为 `valid`。manifest SHA 为 `76fb7a57...`，综合
  contract 报告 SHA 为 `84977c2b...`。
- [ ] 在 SM8850 上用新 payload 回归 30/60/90/120 秒；当前构建机 `adb devices` 无连接设备。
- [ ] 将 Android runner 回传的 `prompt_token_ids` 与上述 HF 锚定结果逐项比较，确认设备实际执行路径没有
  偏离；payload 已内置四档 contract，runner 脚本会按 WAV sample count 自动选择并失败关闭。
- [x] 在主机侧逐项检查 58/60/62 秒附近的 marker 与 embedding offset，并验证 100 秒后三位数 marker；
  末个 marker 分别为 55/60/60/100 秒，100 秒 token IDs 为 `[16,15,15]`。机器报告 SHA 为
  `78af3e97...`。
- [ ] 在 SM8850 回传的 `prompt_token_ids` 中复核上述 60 秒边界和 100 秒三位数 marker，确认设备执行路径
  不存在丢失、重复或错位。
- [x] token/offset 与完整 prompt 对比已产出机器可读报告，不再只依赖最终文本。

完成标准：所有离散 token、位置和计数完全一致；否则先修 processor/runtime contract，不进入 decoder
数值分析。

### 3. 对齐 audio front/back

- [x] 用固定 120 秒输入完成四个 30 秒 chunk 的 `HF FP32 -> MNN CPU` log-mel、audio front 和 audio back
  对齐，并比较拼接后的完整 embedding。四档均 finite 且逐 chunk 通过；报告 SHA 为 `651bd72b...`。
- [ ] 在 SM8850 对四个 chunk 补齐 `MNN CPU -> QNN HTP` 的 audio front/back 中间张量对齐，沿用门禁：
  log-mel cosine >= 0.9999、MAE <= 1e-3，最终 audio embedding cosine >= 0.995。
- [ ] 若首次偏差出现在 MNN CPU，修 frontend/export；若 MNN CPU 正确而 HTP 偏差，定位到 audio front
  或 audio back 的首个异常输出后再修对应 QNN graph。

完成标准：四个 chunk 及拼接结果均通过阈值，排除 audio 链路后才进入 decoder 排查。

### 4. 定位 decoder 首个数值分叉

- [x] 用修正后的完整 prompt 在 MNN CPU W8 上回归 30/120 秒自由生成；两档 prompt IDs 均与 HF contract
  完全一致，120 秒结构完整且与 HF BF16 高度一致。机器报告 SHA 为 `7cf34402...`。
- [x] 在相同 30 秒 audio embedding、prompt IDs 与 HF reference token 上完成 48-step
  `HF BF16 -> MNN CPU W8` teacher-forced 对齐：top-1 agreement 100%，logits cosine mean/min
  `0.997719/0.985197`，top-20 overlap `19.125/20`；报告 SHA 为 `84a1cc8f...`。
- [ ] 使用上述完全相同的 fused audio embedding、prompt token、position id 和 HF target token，在真机补齐
  `MNN CPU -> QNN HTP` teacher-forced 对齐；禁止用各后端自由生成的不同 token 互相比 logits。
- [ ] 分别记录每个 C64 prefill chunk 结束处和每个 decode step 的 top-1、logits cosine、max absolute error、
  saturation 及 KV length，先确定分叉发生在 prefill 还是 decode。
- [ ] 在 30/60/90/120 秒输入上定位首个失败的 sequence position/decode step；若 60 秒可稳定复现，围绕
  首个失败点缩短 trace，不直接采集整段全层输出。
- [ ] 若首个分叉已出现在 QNN prefill，按 chunk、layer 二分定位；若 teacher-forced 一致而自由生成分叉，
  检查首个 top-1 不一致步的 logits margin、KV cache 更新和 sampler 输入。
- [ ] 达到现有 decoder 门禁：QNN/MNN top-1 agreement >= 99%、logits cosine >= 0.98、无 logits
  saturation，并保留 HF BF16、MNN CPU、QNN HTP 三层证据。

完成标准：给出首个异常组件、序列位置和数值证据；没有该证据前不尝试调量化范围、采样参数或性能档位。

### 5. 最小修复与递进回归

- [ ] 仅修复已定位的首个根因，重新构建 payload，并保存新 manifest 与旧/新对照结果。
- [ ] 按 30 -> 60 -> 120 -> 300 秒顺序回归；任一档失败即停止扩大时长。每档同时检查完整可解析、
  时间戳单调且位于音频范围内、speaker id 合法、末尾覆盖、无重复退化和语义忠实度。
- [ ] 单独关闭 30 秒时间戳逆序问题；120 秒不再损坏不能替代这一门禁。
- [ ] 每次回归均由人工或独立 LLM 对照完整音频证据审核，不能只看 parser `status=ok`。
- [ ] 单例 300 秒通过后执行 AliMeeting 10x300 秒质量门禁：CER/cpCER 相对 HF BF16 退化不超过 5%，
  timestamp boundary MAE 增量不超过 100 ms，并逐例记录 speaker count、missed/merged speakers。
- [ ] 全部门禁通过后才更新 `production_ready` 并进入 MeetNote 产品集成。

## P1：正确性通过后的性能与稳定性

- [ ] 对 burst 与 balanced 做固定输入 A/B，记录 RTF、PSS、温度、电量和连续运行降频；选择档位时同时
  考虑延迟、功耗和温升。
- [ ] 连续运行至少 10 个 300 秒 case，保存 logcat、crash、thermal、battery 和每次 result JSON，确认
  p95 RTF <= 1.0、peak PSS <= 4 GiB，且无 HTP SSR、OOM、crash 或 non-finite output。
- [ ] 验证 MNN plugin 初始化失败、QNN context 缺失/损坏、DSP SSR 时 runner 均失败退出并保留结构化错误。

## 已完成：运行与验收基础

- [x] 发布 payload 包含 `llm.mnn.weight`、三个 wrapper、全部 context、Android runtime 和 v81 skel。
- [x] 消除 decoder、audio front、audio back 的 `qnn/graph*.bin` 路径冲突。
- [x] Android 构建开启 MNN plugin，并让 plugin/context 加载失败可诊断。
- [x] 固定 decoder `chunk_limits=[64,1]`，修复 multimodal chunked prefill 状态丢失。
- [x] 下发前校验精确文件集与 SHA-256，按 manifest SHA 隔离设备目录。
- [x] 修复 PSS 采集，并让畸形 raw transcript 返回错误。

## 当前门禁

当前状态是“旧真机 payload 质量失败、修正候选的 HF/MNN CPU 路径通过、待真机验证”。在 QNN HTP
component alignment、decoder teacher-forced 对齐和 AliMeeting 10x300 秒全部通过前，禁止集成到
MeetNote 产品路径。
