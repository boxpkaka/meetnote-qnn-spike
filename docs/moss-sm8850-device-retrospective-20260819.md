# MOSS SM8850 真机验证复盘（2026-08-19）

## 结论

`moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1` 已从“发布目录不可运行”推进到修复候选在
SM8850 真机完成 30/60/90/120 秒完整转写。旧版约 59 秒后的劣化已定位并修复，55 文件候选
release 已完成 manifest 校验和四档真机回归；但 clean end-to-end rebuild 尚未执行，48-step 严格
top-1 门禁已由 W16A16 最终候选清零，单例 300 秒也已通过。clean end-to-end rebuild 与
AliMeeting 10x300 秒质量/稳定性门禁尚未执行，因此暂不进入产品集成。

## 2026-08-20 复验补充

上述“尚未执行”已由新证据覆盖：clean rebuild 和 AliMeeting 10x300 均已执行，但结果不支持晋级。

- clean build 从固定 revision 和完整 patch series 生成 55 文件合法 payload，解决了 patch series 重复应用、
  非交互 shell 找不到 host `clang++`、W16 无 activation quant metadata 时仍执行 logits widening 等
  构建阻塞。新 manifest 为 `d4bfb50f...`。
- clean QNN 与保留候选的同输入 180-step teacher forcing 仅 `98.33%` top-1 一致，首个分叉在
  step 3。QAIRT context 二进制本身不要求 bit-identical，但这种可改变 greedy token 的漂移已超出
  发布复现门禁，不能只比较文件数、尺寸或 manifest 合法性。
- AliMeeting 10x300 连跑 8/10 成功。case 5 因 8192 总 context/4096 新 token 上限而截断；HF BF16
  完整输出需要 prompt 3996 + generated 4897 = 8893 token。case 9 生成正文却丢失全部时间戳，属于
  decoder 格式退化。case 2 后段还出现重复“对”的语义退化。
- 相对 HF BF16 和官方 TextGrid，CER/cpCER 分别退化 `0.1001/0.1085`，时间戳 MAE 增加
  `1.018s`。这些结果超过既定门限，不能通过 parser 放宽、字符串修补或调整验收阈值解决。
- runtime 层没有出现 SSR/OOM/crash/non-finite，最大 PSS 为 `3.70 GB`；但 burst 连跑使电池温度
  从约 36°C 升至 52°C，RTF 从首例 `0.800` 恶化，最差到 `4.198`。
- case 9 在降温后单独复跑仍产生完全相同的 1530 个 token 且无任何时间戳；RTF 从热态 `1.451`
  恢复到 `0.644`，证明热降频只影响速度，不是该格式崩溃的根因。
- 10240 context 的首次设备复跑仍在 4096 token 停止，进一步定位到 runner 在加载 payload 后又
  硬编码覆盖 8192/4096，并给 `response()` 显式传 4096。不能直接删除这些调用：MNN multimodal
  路径会在首次 decoder forward 发生 Plugin shape failure。最终 patch 0015 从 release config 解析
  token limit，并动态执行显式 `set_config` 和四参数 `response`；这也解释了为什么只改 payload
  config 没有生效。

解决顺序必须是：先用 10240 context 冷机复跑 token-budget case；再对无时间戳 case 做冷机复现和
首个格式 token 的 teacher-forced 定位；固定 clean-build 数值复现；最后才将 sustained profile 改为
balanced/DCVS 并重跑 10x300。`burst` 就是当前 turbo 实现，不应进一步加码。

验证设备为 Vivo V2505A、SM8850、Android 16。输入为 AliMeeting
`R8001_M8004_0000` 的 30/60/90/120 秒切片。

## 最终根因与修复

原阻塞由两个互相独立的精度问题叠加：

1. MNN QNN converter 对 W8 blockwise 权重没有原生 LPBQ 表达，旧路径把它二次转换为
   per-channel int8，误差从 layer 0 开始逐层累积。修复候选删除 wrapper 的 activation quant
   metadata，让 decoder QNN graph 使用 FP16 activation 和由原 W8 blockwise 权重解量化得到的 FP16
   constant；外部 W8 权重文件和 CPU 语义不变。服务器上转换前后 180-step logits bit-identical。
2. `config.json` 的 decoder `precision=low` 在手机上启用了 MNN CPU FP16。QNN graph 外围仍有
   attention、RoPE、KV/cache 等 CPU 算子，因此 90/120 秒会产生畸形时间戳。原 W8 CPU、W16 CPU
   和新 W8 QNN 在手机上都复现同样失败；改为 `precision=normal` 后全部恢复。服务器不支持 FP16，
   自动走 FP32，故旧 server CPU reference 没暴露该问题。

仓库构建链路现加入 `strip_mnn_activation_quant.py`，payload 组装和安装前验证均强制 decoder
`precision=normal`。这不是 parser 容错或输出后处理，而是从 graph encoding 和 runtime 精度配置修复。

修复候选真机结果：

| 输入 | 状态 | segments | RTF | peak PSS | 与 HF BF16 对照 |
| --- | --- | ---: | ---: | ---: | --- |
| 30 秒 | `ok` | 4 | 0.468 | 546,489,344 B | 内容和 speaker 一致，同 speaker 片段被合并 |
| 60 秒 | `ok` | 10 | 0.424 | 576,571,392 B | 文本一致，仅边界有 10-70 ms 浮动 |
| 90 秒 | `ok` | 15 | 0.423 | 838,252,544 B | 少一个语气助词“的”，其余一致 |
| 120 秒 | `ok` | 19 | 0.472 | 1,081,573,376 B | 一处短语差异，其余内容、speaker 和边界一致 |

正式候选位于 `/data/experiments/meetnote-moss-contract-v2/release-w8-pure-fp16-normal-v1`，
artifact manifest SHA-256 为 `73c245d9c24cba146ed21639c2a56957ed812aefefe8152fed8689c16440deb0`。

数值门禁方面，最终配置相对同机 MNN CPU 的 180-step top-1 为 `99.44%`、cosine mean/min 为
`0.997432/0.966427`；48-step 为 `97.92%`，唯一分叉在 step 3，分叉步 cosine 为 `0.999713`。
这组 W8 pure-FP16 数据证明根因修复有效，但仍未通过严格短前缀门禁。

修正 `precision` 后重新打开 W16A16 路线，结论发生改变：此前 W16 QNN 与 CPU 的 90 秒共同失败
是外围 MNN CPU FP16 所致，并非 W16 模型退化。`precision=normal` 下 W16A16 对同机 CPU 的
180-step top-1 为 100%，cosine mean/min 为 `0.999998/0.999963`；对原 W8 server CPU 的
180-step top-1 同样为 100%。30/60/90/120 秒分别输出 4/10/15/19 段，均完整解析。

W16A16 正式候选 release 的 artifact manifest SHA-256 为
`12831c6fd9bc687b2db20a82f9fdad1c835f5f711de207598766e26ccc7d6b1c`。单例 300 秒真机结果为
`status=ok`、54 段、4 位 speaker、末尾 299.92 秒、RTF `0.950522`、peak PSS
`2,661,988,352 B`，无 SSR/OOM/crash。HF FP32 300 秒 reference 为 56 段并覆盖 299.99 秒；逐段
语义审核确认主题、speaker 和完整讨论流程保留，候选少量合并/省略 backchannel，并有一处
“十八到五”对“十八到五十”的数字短语差异，已原样记录。W16A16 现为正确性优先候选，但在 clean
rebuild 和 AliMeeting 10x300 秒门禁完成前仍不得标为 production ready。

## 阻塞与修复

本轮发现的问题不是单一模型问题，而是发布组装、Android 构建和运行时状态管理叠加：

1. 发布目录没有完整 payload，且缺失 `llm.mnn.weight`。现在由
   `assemble_moss_runtime_payload.py` 从 export 与三个 QNN component 统一组装，缺文件立即失败。
2. 三个 wrapper 都引用 `qnn/graph*.bin`，在同一个 external root 下发生同名 context
   冲突。现在分别重写为 `deq/`、`afq/`、`abq/`，且保持二进制路径等长。
3. FastRPC skel 放在 `dsp/`，设备实际从 `dsp/cdsp/` 查找。Android 打包脚本已修正目录。
4. Android MNN 没有开启 plugin，MOSS multimodal op 无法注册。构建已增加
   `MNN_WITH_PLUGIN=ON`，plugin 初始化失败也不再继续执行空 kernel。
5. external context 的未指定 size 被编码成 `UINT64_MAX`，QNN 收到无效 buffer size。
   runtime 现在把该 sentinel 还原为 `0`，由文件大小决定，并把 mmap、binary metadata、
   context create 错误改为可诊断失败。
6. decoder 配置遗漏 `[64, 1]` chunk limit。组装工具现在强制写入并在下发前验证。
7. chunked prefill 对每个 chunk 重复调用 multimodal `embedding()`，第一个 chunk 又会清空
   全部 audio embedding 状态。现在先对完整 prompt 生成一次 embedding，再沿 sequence 轴切片。
8. PSS 读取把 `smaps_rollup` 第一行误当字段后提前结束，始终得到 0。现在按行寻找 `Pss:`。
9. parser 会从损坏的 raw output 中摘取少数合法片段，并误报 `status=ok`。现在要求 raw
   output 被完整消费、speaker id 为两位、每段结构合法；120 秒损坏输出会明确失败。
10. 下发目录固定，旧文件可能污染新实验。现在使用 manifest SHA 生成设备目录，并在
    `adb push` 前校验精确文件集和 SHA-256。

## 真机结果

修正后的 Release Android runtime 与 payload 通过 51 个文件的完整性检查。

| 输入 | 状态 | total | RTF | peak PSS | 质量结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 30 秒 | `ok` | 14.625 s | 0.487 | 539,787,264 B | 可转写；后续确认所谓“逆序”是合法重叠说话 |
| 120 秒 | `error` | 32.922 s | 0.274 | 566,681,600 B | 约 59 秒后出现畸形时间戳、`S0` 和重复 token，parser 正确拒绝 |

语义审核：现有 120 秒 BF16 结果能覆盖到 119.94 秒并产生 18 个有效片段、4 位说话人；
QNN 输出只在前半段保持可读，后半段与完整音频证据明显不一致。后续核对发现，该 BF16 run 虽记录
clean source SHA `80cf2405...`，实际推理使用的是 `+30 dB + tanh` 后 SHA `7c9170f9...` 的音频，
因此它只能证明模型具备长音频覆盖能力，不能作为真机 clean 输入的严格同输入 reference。
当前结果仍应拒绝，但根因不能据此直接限定为 QNN 数值/decoder。不得用字符串规则修补或接受该结果。

原始日志保存在 MeetNote 工作区：
`build/device-validation/20260819-moss-qnn-sm8850/artifacts/30s-clean-fixed-run.log` 和
`build/device-validation/20260819-moss-qnn-sm8850/artifacts/120s-clean-fixed-run.log`。

## Turbo 模式

当前 external-context 执行路径在 `RawExecutorWrapper` 构造时已经调用
`setPowerConfigBurst()` 和 `setRpcLatencyAndPolling()`。这个 burst 配置会关闭 DCVS、禁止
DSP sleep，并把 bus/core 的 min、target、max 全部设为
`DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER`；它比 balanced 配置使用的 `TURBO` corner 更激进。

因此本轮不再新增一个“turbo”开关，也不能把长音频质量失败归因于没有开 turbo。当前
manifest 明确记录 `profile=burst`。后续只有在正确性门禁通过后，才做 burst/balanced 的
时延、温升与功耗 A/B，避免以最高功耗模式掩盖错误。

## Decoder 真机数值定位补充

在修正 prompt contract 后，30 秒 raw marker 出现 `22.51 -> 13.24`。后续与 clean HF reference
逐段复核，确认这是 S01/S02 重叠说话的 end/start 交错，不是非法时间戳；合法性应按每段
`start < end` 和 segment start 非递减判断，不能要求 raw marker 全局单调。固定相同
472-token prompt、audio embedding 和 48 个 reference token 的 teacher forcing 显示，原始 QNN HTP
decoder top-1 agreement 为 `93.75%`，cosine mean/min 为 `0.955826/0.685391`。四个 audio chunk
均通过 `0.995` cosine 门禁，所以阻塞不在 prompt 或 audio encoder。

逐层 probe 进一步显示 layer `0/7/15/23/27` 的 hidden cosine mean 从 `0.999513` 逐步下降至
`0.965504`。graph30/lm_head 单独回 CPU 没有消除 step 3 分叉，C1 prefill 也没有改善，说明这是
decoder HTP 计算误差的逐层累积，不是 prefill 边界或单个 lm_head graph 故障。

QAIRT host SDK 实际位于仓库 `sdk/qairt/2.48.40.260702`。补齐 NDK host runtime、串行 context merge
和可配置 `fp16_relaxed_precision` 后，成功重建并在同一台 SM8850 上测试严格 FP16 context。仅关闭
relaxed FP16 时，top-1 agreement 提升到 `97.92%`，首个分叉从 step 3 推迟到 step 21，但 cosine
mean/min 仍为 `0.953352/0.716918`。

后续复核发现，曾标记为“strict FP16 + graph30/final norm/lm_head CPU”的 48/180-step 实验下发了
原始完整 MNN CPU wrapper（SHA `984e846e...`），而非生成目录中的 QNN plugin wrapper。对应的
`0.995790/0.974054` 和 `0.995854/0.949801` 只能说明 Android CPU 与服务器 CPU 接近，不能证明混合
分区改善了 QNN。证据文件保留但已重新标注，避免继续传播错误结论。

用真实 QNN plugin wrapper 重跑后，把 layer 8 以后或 layer 27 及输出头放回 CPU 均在 step 0 无
logits，说明当前 compiler/runtime 不支持这种细粒度后段分区作为可运行候选。因此 relaxed FP16 是
已证实的重要误差来源，但不是完整修复；发布 decoder 已在每次诊断后恢复并校验 SHA-256。

进一步核对转换补丁确认：QNN HTP 的 LPBQ 只原生支持 4-bit blockwise 权重；现有 W8 blockwise 输入
会回退为 per-channel requantization。这能解释投影/FFN 误差从 layer 0 开始逐层累积。当前修复路线
改为先验证 QNN 原生支持的 W4A16。W4A16 CPU 对 W8 CPU 的 180-step top-1 agreement 只有 `91.67%`，
cosine mean/min 为 `0.944675/0.608280`，首个 top-1 分叉在 step 5；因此它在下发前就被质量门禁否决，
不能以更好的 HTP 可实现性换取明显模型退化。W16A16/FP16 权重的 CPU baseline 对 W8 CPU 在 180
steps 达到 `99.44%` top-1、`0.997284/0.977352` cosine mean/min，仅 step 132 有一个 top-1 分叉，
已明显优于 W4，允许继续进入 QNN 数值验证。该路线优先用存储和内存换正确性，再评估设备资源边界。
项目状态继续保持 `BLOCKED_DECODER_QNN_NUMERICAL_ALIGNMENT`，通过前不做自由生成晋级。

## 根因边界

本轮已修复可运行性、可诊断性和错误验收问题，但没有证明 QNN 模型数值正确。

复盘后的静态 contract 核对又发现，固定模型的 `processor_config.json` 要求每 5 秒插入 time marker，
且 audio span 两侧必须包含 `<|audio_start|>`（151669）和 `<|audio_end|>`（151670）。失败 payload
对应的 runtime 源码却每 2 秒插入 marker，导出配置也没有写入两个 boundary token。这会确定性改变
prompt token 序列，必须优先修复、重建并回归，不能直接把 59 秒后的劣化归因于 QNN 数值误差。

仓库补丁现已改为从 `llm_config.json` 读取 5 秒间隔、写入 boundary token，并在组装和下发前拒绝
contract 漂移。若修正后的 payload 仍失败，再使用同一 audio embedding 和 token 序列做
teacher-forced 对齐，按 audio front、audio back、decoder prefill、decode step 缩小首个数值分叉点；
Linux QNN CPU 结果不能替代 SM8850 HTP 结论。

完整 prompt 对比随后发现第三项确定性漂移：模型把 chat template 单独保存在 `chat_template.jinja`，
旧 MNN `.mtok` exporter 只读取 `tokenizer_config.json` 的内嵌字段，导致 native runtime 的
`apply_chat_template()` 原样返回用户内容。真机 prompt 因而缺少 system、user header、user end 和
assistant header，共 19 个 token。旧主机 probe 的“逐项一致”是 official/runtime 两侧共同使用同一个
缺模板的 `.mtok` 所造成的循环验证，不能作为 HF 对齐证据。

固定回归输入已从 `R8001_M8004_0000_300s.wav` 生成 30/60/90/120 秒前缀，manifest 位于外部构建目录
`inputs/regression-contract-v2/manifest.json`。其中 120 秒 SHA 为 `80cf2405...`，与现有 BF16 case 记录的
clean source 一致。后续 HF、MNN CPU 和真机必须直接使用这些相同 SHA 的 WAV。

clean HF BF16 reference 现已补跑完成。30/60/90/120 秒分别生成 6/10/15/19 个片段，覆盖到
29.94/59.88/89.29/119.98 秒；120 秒识别出 4 位说话人。四档片段范围均合法、start timestamp 单调、
speaker id 均为两位格式，且 60 秒结果能正常越过旧真机约 59 秒的故障边界。reference manifest 位于
`/data/experiments/meetnote-moss-contract-v2/hf-bf16-clean-reference/manifest.json`，SHA 为
`fa7605a1...`。这排除了 clean 输入和官方模型长音频能力，但仍不能替代修正 payload 的真机验证。

主机 audio 链路也已扩展到完整 120 秒的四个 30 秒 chunk，不再只验证首块。实际 MNN
`whisper_fbank`、MNN CPU audio front/back 与 HF FP32 的拼接结果分别为：log-mel cosine
`0.9999959`、MAE `0.0005859`，最终 audio embedding cosine `0.9999067`；四个 chunk 的最差最终
cosine 为 `0.9998102`，全部通过 `0.995` 门禁。机器报告位于
`/data/experiments/meetnote-moss-contract-v2/audio-mnn-alignment-120s.json`，SHA 为 `651bd72b...`。
因此 host frontend/export 已排除，仍需真机补齐 QNN HTP component 对齐。

修正后的相同 prompt 又在 MNN CPU W8 decoder 上完成 30 秒和 120 秒自由生成。两档设备前同构 prompt
分别为 472/1638 token，IDs 与 HF contract 全量一致。120 秒 MNN CPU 输出 19 段、4 位说话人并覆盖到
119.98 秒，与 clean HF BF16 transcript 的字符相似度为 `0.998496`（719 vs 720 generated tokens），
没有约 59 秒后的结构损坏。机器报告位于
`/data/experiments/meetnote-moss-contract-v2/mnn-cpu-reference-comparison.json`，SHA 为 `7cf34402...`。
因此剩余根因已收缩为旧 prompt contract 和/或 QNN HTP decoder 执行路径。

decoder logits 也已使用相同 30 秒 audio embedding、472 个 prompt IDs 和 181 个 HF reference token 做
48-step teacher forcing。HF BF16 与 MNN CPU W8 的 top-1 agreement 为 100%，logits cosine mean/min 为
`0.997719/0.985197`，top-20 overlap 平均 `19.125/20`，没有首个 top-1 分叉。比较报告位于
`/data/experiments/meetnote-moss-contract-v2/teacher/hf-vs-mnn.json`，SHA 为 `84a1cc8f...`。这进一步
排除 MNN W8 导出本身；真机连接后应直接以该 MNN CPU probe 为 QNN HTP 对照。

修正候选已在外部目录 `/data/experiments/meetnote-moss-contract-v2/release` 组装完成。它复用数值内容未变化
的三个 QNN context 和权重，更新 Android runtime、`llm_config.json` 与 `tokenizer.mtok`，并内置四档
prompt contract；55 文件 payload 完整性校验为 `valid`。artifact manifest SHA 为 `76fb7a57...`，runner
SHA 仍为 `61e2e3d4...`。当前构建环境 `adb devices` 无连接设备，尚未产生修正后的真机结果。

独立 HF tokenizer 已复现历史 120 秒 `promptLen=1638`，并生成四档 token ID SHA。修复后的
`tokenizer.mtok` 对 30/60/90/120 秒分别得到 472/859/1246/1638 个 token；MNN official/runtime 构造逐项
一致，且各自均命中独立 HF 长度与 SHA。综合 contract 报告 SHA 为 `84977c2b...`。runner 已增加
`prompt_token_ids` 输出，下一次真机执行会按 WAV sample count 自动选择 payload 内的 contract，比对设备
实际 token 序列，并归档 input、payload、runner、log、result 的 SHA-256。

marker 边界另以 58/60/62/100/120 秒输入做了主机机器校验。末个 marker 分别落在
55/60/60/100/120 秒；100 秒 marker 正确编码为三位 token IDs `[16,15,15]`，其 audio embedding position
为 1240（遵循官方 `int(12.5 * 5)` 的每段 62-token 插入规则）。报告位于
`/data/experiments/meetnote-moss-contract-v2/audio-token-boundaries.json`，SHA 为 `78af3e97...`。仍需在
真机回传的完整 prompt IDs 中复核设备执行路径。

## 2026-08-20 补充：token limit runner 回归

10240-token context 首次真机运行暴露的 `Plugin op infer shape failed` 不是 QNN graph 本身损坏。
对同一 8192 graph 做 runner A/B 后确认：删除 runner 的 `set_config` 并依赖 `response` 默认参数会让
multimodal 首次 decoder forward 失败；恢复显式 `set_config` 与四参数 `response` 后，同一 graph
立即恢复。最终实现从 `config.json` 解析 `max_all_tokens/max_new_tokens`，不再写死 8192/4096，
同时保留 MNN 这两个必要的显式调用。

教训是 token limit 同时存在于 QNN context、MNN runtime config 和 runner 调用三个层级。静态 manifest、
context binary utility 和模型 load 成功都不能证明三层合同一致；发布检查必须至少增加一个真实 decoder
forward smoke，并同时覆盖默认 8192 档和扩展 context 档。

修复后的 30 秒真机结果在 8192 与 10240 档均为 `status=ok`。原 case 5 的 10240 长测因 ADB 与远端
连接中断而没有最终 JSON，所以本轮只证明 runner 阻塞已修复，不证明长输出截断或全量质量门禁已通过。
