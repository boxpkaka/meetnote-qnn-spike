# MNN LLM Build Assets

本文件记录 meeting-demo 接入 MNN/Qwen 端侧纪要时的本地资产布局、打包命令和设备验证教训。每次处理 MNN native、Qwen 模型或 USB 安装前，先看本文件。

## 本地资产约定

MNN 源码通过 `third_party/MNN` Git submodule 引用官方仓库，主仓库固定一个已验证 commit；MNN Android
构建产物和 Qwen 模型放在项目内 `local-assets/`，但整个目录被 `.gitignore` 忽略，不能提交到仓库。

当前验证过的 MNN commit：

```text
0bff03cbef43c783f44e41484b9f8a0b28bd758d
Merge pull request #4570 from LudovicoYIN/llm-qnn-raise-stack
```

推荐布局：

```text
  local-assets/
    mnn-android-build/
      build_llm_stdout_64/
  models/
    Qwen3-0.6B-MNN/
    Qwen3-4B-MNN/
third_party/
  MNN/  # Git submodule: https://github.com/alibaba/MNN.git
```

`local.properties` 可以使用项目内路径：

```properties
meetnoteMnnRoot=/Users/mingdongyu/Documents/workspace/MeetNote/third_party/MNN
meetnoteMnnAndroidBuildDir=/Users/mingdongyu/Documents/workspace/MeetNote/local-assets/mnn-android-build/build_llm_stdout_64
```

Android demo 运行时默认优先找 app 私有目录 `files/meetnote-llm/models/<model>`，调试时也会查 `/data/local/tmp/meetnote_mnn_eval/models/<model>`。设备上没有模型时，即使 APK 含 MNN native，也只能显示模型缺失。

## MNN Native 打包

普通 debug 构建不会包含 MNN LLM native runtime。要构建可端侧生成纪要的 APK，必须显式打开：

```bash
./gradlew --no-daemon --no-watch-fs :samples:meeting-demo:assembleDebug \
  -PmeetnoteLlmMnn=true
```

若没有写 `local.properties`，需要显式传：

```bash
./gradlew --no-daemon --no-watch-fs :samples:meeting-demo:assembleDebug \
  -PmeetnoteLlmMnn=true \
  -PmeetnoteMnnRoot=/Users/mingdongyu/Documents/workspace/MeetNote/third_party/MNN \
  -PmeetnoteMnnAndroidBuildDir=/Users/mingdongyu/Documents/workspace/MeetNote/local-assets/mnn-android-build/build_llm_stdout_64
```

安装前必须检查 APK 内是否真的包含 MNN native 库：

```bash
unzip -l samples/meeting-demo/build/outputs/apk/debug/meeting-demo-debug.apk \
  | rg 'lib/arm64-v8a/(libmeetnote_mnn_llm|libMNN).*\.so'
```

至少应看到：

```text
lib/arm64-v8a/libMNN.so
lib/arm64-v8a/libmeetnote_mnn_llm.so
```

设备安装建议用 `pm install`，避免 Vivo 安全安装弹窗影响：

```bash
adb push samples/meeting-demo/build/outputs/apk/debug/meeting-demo-debug.apk /data/local/tmp/meeting-demo-debug-mnn.apk
adb shell pm install -r -t -g /data/local/tmp/meeting-demo-debug-mnn.apk
```

## 这次问题的根因

这次 USB 设备上看到“当前安装包未包含本地纪要引擎”，不是模型目录问题，而是安装了普通 debug APK。普通包里没有 `libmeetnote_mnn_llm.so` 和 `libMNN.so`，因此 app 正确降级为“未包含本地纪要引擎”。

正确修复过程：

1. 用 `-PmeetnoteLlmMnn=true` 重新构建当前源码。
2. 构建完成后先用 `unzip -l` 验证 APK 内存在 `lib/arm64-v8a/libMNN.so` 和 `lib/arm64-v8a/libmeetnote_mnn_llm.so`。
3. 再安装到 USB 设备。
4. 用 logcat 验证 native loader 输出 `libmeetnote_mnn_llm.so ... ok`。
5. 截图确认 UI 显示“大模型已就绪：Qwen3-4B-MNN”。

## Gradle 卡住教训

本机曾在 `:samples:meeting-demo:mergeDebugResources` 长时间无输出。线程栈显示根因不是 aapt2 死锁，而是 Gradle 在任务前后对大资源/输出目录做文件 hash 和 snapshot。旧的 `build/intermediates/merged_res_blame_folder` 中还出现了 `values-sl 2.json` 这类历史残留文件，会放大快照开销。

处理方式：

```bash
rm -rf samples/meeting-demo/build/intermediates \
       samples/meeting-demo/build/outputs \
       samples/meeting-demo/build/generated \
       samples/meeting-demo/build/tmp
./gradlew --stop
./gradlew --no-daemon --no-watch-fs :samples:meeting-demo:assembleDebug -PmeetnoteLlmMnn=true
```

不要在 APK 时间戳未更新、`unzip -l` 未确认 native 库存在时安装；否则容易把旧普通包误装到设备上。

## 设备侧验证清单

检查 APK native 加载路径时，`nativeLibraryDir` 为空或目录下没有 so 不一定是错误。当前 APK 使用 `extractNativeLibs=false`，Android 可以直接从 `base.apk!/lib/arm64-v8a` 加载 native 库。

关键 logcat：

```text
Load ... base.apk!/lib/arm64-v8a/libmeetnote_mnn_llm.so ... ok
```

检查模型目录：

```bash
adb shell ls -lh /data/local/tmp/meetnote_mnn_eval/models/Qwen3-4B-MNN
```

调试时如需推送项目内模型：

```bash
adb shell mkdir -p /data/local/tmp/meetnote_mnn_eval/models
adb push local-assets/models/Qwen3-4B-MNN /data/local/tmp/meetnote_mnn_eval/models/
```

## JSON 解析教训

历史上 Qwen3-1.7B 在短会议 `FAST_SINGLE_SHOT` 路径上可能输出接近正确但不完全合法的 JSON，例如：

- 外层包了 ```json 代码块。
- `topics`、`decisions`、`action_items`、`risks`、`open_questions` 这些顶层字段之间漏逗号。

这种情况下 native 生成和模型加载都成功，`raw.txt` 中也有可用内容，但严格 `JSONObject` 解析会失败，UI 曾显示 `no JSON object found`。修复原则是只做窄范围 repair：先严格解析；失败后只针对 `meetnote.summary.v1` 已知顶层字段之间的缺失逗号做修复；随后仍必须通过 `schema_version` 和 `evidence_ids` 校验。

不要把 parser 放宽成“任意文本都生成纪要”。如果模型输出无法修成带合法证据链的 summary JSON，仍应失败并保留 `raw.txt`。

`FAST_SINGLE_SHOT` 还需要模型级 repair retry：当 native 生成成功但 `meetnote.summary.v1` 解析或 evidence 校验失败时，remote service 会写 `repair-prompt.txt`，复用同一个 `:meetnote_llm` 进程再生成一次 `raw.repair.txt`。repair 成功才写 `summary.json`；repair 后仍失败则保留原始输出并报错。不要对 native crash、模型目录缺失、超时做无限重试，这些不是格式问题，重试只会拖慢 UI 并扩大失败域。

## 4B 默认策略

2026-07-09 起，meeting-demo 丢弃 Qwen3-1.7B 作为运行候选；默认且唯一产品 profile 是 `Qwen3-4B-MNN`。

- 不再从 4B fallback 到 1.7B；设备缺少 4B 模型时必须显式失败并提示模型缺失。
- 1.7B 仅作为历史实验记录保留，不参与 app 内模型选择。
- USB 验证和调试模型目录统一使用 `/data/local/tmp/meetnote_mnn_eval/models/Qwen3-4B-MNN`。

2026-07-09 USB 验证记录：

- `meeting-demo-debug.apk` 已用 `-PmeetnoteLlmMnn=true` 构建，并确认 APK 内包含
  `lib/arm64-v8a/libMNN.so` 和 `lib/arm64-v8a/libmeetnote_mnn_llm.so`。
- Vivo `V2505A/PD2505` 上模型目录 `/data/local/tmp/meetnote_mnn_eval/models/Qwen3-4B-MNN`
  完整存在，`llm.mnn.weight` 约 2.5 GiB。
- 首次安装后 UI 能显示“大模型已就绪：Qwen3-4B-MNN”，说明 4B 默认选择和 native loader 成功。
- 120s AISHELL-4 转写走 quality-agent 时，第一段 extract 在 150s step timeout 内未完成：
  `LLM summary timed out after 150000ms`。这是 4B CPU 端多阶段链路的时延问题，不是 fallback。
- 同日修正短文本路由：AISHELL-4 这类 120s/约 500 字但 20+ utterance 的输入应走 single-shot；
  utterance 数只是碎片度，不应单独把短文本推入多阶段链路。
- 修正后的 APK 再次安装时被 Vivo 安全确认拦截，`pm install` 返回
  `INSTALL_FAILED_ABORTED: User rejected permissions`。继续 USB single-shot 验证前，需要在设备侧确认安装。

当前 MeetNote 的 MNN/Qwen 路径仍使用 CPU backend。MNN 仓库文档显示 QNN backend 可调用部分高通 NPU/HTP
能力，但需要 `-DMNN_QNN=ON`、QNN SDK、设备对应 SOC ID / HEXAGON ARCH、QNN HTP 运行库，以及在线或离线构图产物。
本项目当前 APK 未打包 QNN 库，也没有为 Qwen3-4B-MNN 生成 QNN 离线产物；不能把当前 4B 测试结果理解为高通 NPU
加速结果。

## QNN 分支边界

2026-07-09 会话结论：可以新开 QNN 分支，但只能作为隔离 spike，不能影响当前 `Qwen3-4B-MNN` CPU 默认主链路。

开分支的理由：

- 已有 4B CPU 基线：前台 LLM-only 路径在 Vivo 设备上完成，`total_ms=71392`，RSS 约 3.19 GiB。
- 已有 `meetnote.debug.input_transcript_json` 快环，可用真实 transcript 文本直接跑 LLM，不再被 ASR 迭代成本干扰。
- 当前 summary quality gate 已能移除无证据的 decisions/action_items，适合作为 QNN A/B 输出质量基准。

QNN 分支必须保持这些约束：

- 分支建议命名 `feat/qnn-backend-spike`。
- CPU backend 仍是默认和回退边界；QNN 构建、模型产物、runtime 库缺失时，不得影响 CPU 版 APK 安装和运行。
- 不要把 QNN 当作已验证的高通 NPU 加速路径。只有跑通同一 transcript、输出合法 summary JSON、质量门禁通过、
  且 logcat 无 native crash / HTP 错误后，才能声明“QNN 路径可用”。
- QNN spike 验收先看可用性和失败域，不先承诺更快：
  1. MNN 以 `-DMNN_QNN=ON` 构建成功。
  2. APK 或测试环境带齐 `libQnnHtp.so`、`libQnnHtpPrepare.so`、目标 HEXAGON ARCH 对应 stub/skel 等依赖。
  3. `ADSP_LIBRARY_PATH` / `LD_LIBRARY_PATH` 能让设备实际加载 QNN HTP 库。
  4. Qwen3-4B 生成 QNN 所需在线或离线图产物，并记录产物生成命令。
  5. 同一 `build/llm-e2e/run1/input.json` 前台 LLM-only 可完成，summary JSON 合法，quality gate 通过。
  6. 输出 CPU/QNN 对比：load/prefill/decode/total、RSS、温度或功耗观察、失败模式。

已知高风险：

- MNN 文档说明 NPU 后端不支持可变形状、控制流等动态模型，算子覆盖少；Qwen LLM 的动态上下文和 cache
  形状可能需要额外静态化或离线图拆分。
- `MNN_QNN`、`MNN_COREML`、`MNN_NNAPI` 在 MNN 中共用同一个 backend type，编译宏最多只能打开一个；
  QNN 实验不要污染现有 CPU 构建产物。
- 公开 MNN issue 中已有 Qwen3-4B + QNN HTP crash 案例。若遇到 HTP crash 或设备 NPU 卡死，需要记录
  QNN SDK 版本、HEXAGON ARCH、SoC、MNN commit、转换命令和 logcat，不能只在 app 层吞掉。

### QNN spike 实现约定

`feat/qnn-backend-spike` 上新增了显式实验开关，普通 CPU/MNN 构建路径不变：

```bash
./gradlew --no-daemon --no-watch-fs :samples:meeting-demo:assembleDebug \
  -PmeetnoteLlmQnn=true \
  -PmeetnoteMnnRoot=/Users/mingdongyu/Documents/workspace/MeetNote/third_party/MNN \
  -PmeetnoteMnnQnnAndroidBuildDir=/path/to/mnn-qnn-android-build \
  -PmeetnoteQnnJniLibsDir=/path/to/qnn-arm64-libs
```

- `meetnoteLlmQnn=true` 会隐式打开 demo 的 MNN native wrapper，但要求使用独立的 QNN 版 `libMNN.so` 构建目录。
- `meetnoteMnnAndroidBuildDir` 仍保留给 CPU 版 APK；不要把 QNN 版 `libMNN.so` 覆盖到现有
  `local-assets/mnn-android-build/build_llm_stdout_64`。
- `meetnoteQnnJniLibsDir` 是可选项，指向一个包含 `*.so` 的 arm64 QNN runtime 目录；Gradle 会把这些 so
  复制进 APK 的 `lib/arm64-v8a/`。若使用 `/data/local/tmp/meetnote_mnn_eval/qnn-libs` 调试，也可以不打包。
- QNN profile 使用独立实验模型目录 `Qwen3-4B-MNN-NPU`，要求存在 `config_qnn.json` 和
  `qnn/llm.mnn`；CPU profile 继续使用 `Qwen3-4B-MNN`，要求 `config.json`、`llm.mnn` 和
  `llm.mnn.weight`。不要把 QNN graph 写回 CPU 目录。
- QNN profile 当前按 MNN LLM 离线 QNN 产物的约定执行：`config_qnn.json` 中的 plugin/替代模型负责进入
  QNN，JNI 传给 MNN 的 `backend_type` 仍是 `cpu`，避免把离线 plugin 路径误切成在线 `MNN_FORWARD_NN`。
- JNI 会在加载模型前把 QNN profile 的 `qnnLibraryPath` 写入 `ADSP_LIBRARY_PATH` 和 `LD_LIBRARY_PATH`，
  用于 USB spike。QAIRT/QNN 2.48 后按 host/DSP 分目录：
  `/data/local/tmp/meetnote_mnn_eval/qnn-libs/arm64-v8a` 和
  `/data/local/tmp/meetnote_mnn_eval/qnn-libs/hexagon-v81/unsigned`；profile 使用冒号连接这两个目录，
  避免把 Android arm64 host lib 和 Hexagon DSP skel/extension 混在同一个 `arm64-v8a` APK 目录里。
  如果设备仍无法加载 HTP stub/skel，需要保留 logcat 和 QNN SDK 版本。

2026-07-09 当前 USB 设备探测：

- 设备：Vivo `V2505A/PD2505`，Android 16/API 36，`ro.board.platform=canoe`，
  `ro.soc.manufacturer=QTI`，`ro.soc.model=SM8850`。
- 设备 QNN V81 runtime 在 `/odm/lib64/npuhw/qnnv3/` 和 `/vendor/lib64/hw/`：
  `libQnnHtp.so`、`libQnnSystem.so`、`libQnnHtpV81Stub.so`、`libQnnHtpV81Skel.so`。
- MNN 公开 QNN 2.37 依赖包只有 V68/V69/V73/V75/V79 stub/skel，没有 V81。第一轮可用该包验证
  app 打包和 CPU fallback；V81 设备 runtime 已拉到 ignored `local-assets/qnn-device-v81/arm64-v8a/`，
  仅用于本机 USB spike，不提交入仓库。
- SoC ID 不能从 `libQnnHtpV81.so` 文件名反推，生成离线产物前必须用高通/QNN 官方来源确认。
- Qualcomm 当前公开资料指向 Qualcomm AI Runtime / AI Engine Direct SDK 和 QPM 下载入口；MNN 的
  `npu_convert.py` 明确需要完整 Linux x86_64 host SDK 中的 `qnn-model-lib-generator`、
  `qnn-context-binary-generator`、`libQnnHtp.so` 和 `libQnnHtpNetRunExtensions.so`。MNN 公开 2.37
  依赖包不包含这些 host 工具。
- 本机已用 MNN `prepare_qnn_deps.sh` 下载并准备 QNN 2.37 依赖：
  `third_party/MNN/source/backend/qnn/3rdParty`。
- QNN 版 MNN Android runtime 已构建成功，产物隔离在：
  `local-assets/mnn-android-build/build_llm_qnn_64/`。
- QNN runtime libs 已复制到 `local-assets/qnn-sdk-2.37/arm64-v8a/` 和
  `local-assets/qnn-device-v81/arm64-v8a/`；设备调试目录
  `/data/local/tmp/meetnote_mnn_eval/qnn-libs/arm64-v8a/` 已补入 V81 runtime。
- QNN 实验 APK 已用 `-PmeetnoteLlmQnn=true` 构建并安装成功，APK 内确认包含 `libMNN.so`、
  `libmeetnote_mnn_llm.so`、`libQnnHtp.so`、`libQnnSystem.so`、V81 stub/skel。Gradle 的 QNN so
  准备任务会先清空 generated jniLibs 目录，避免从 V79/V81 两套 runtime 混包。
- MNN standalone online QNN 探测：将 `Qwen3-0.6B-MNN` 临时设为 `backend_type=npu` 后，QNN backend
  确实被触发，但设备 logcat 报 `libQnnHtpPrepare.so` 缺失、HTP prepare backend 加载失败，随后
  `llm_demo` SIGSEGV。该路径只作为失败边界记录，不接入 app 默认行为。
- 由于模型目录仍缺 `config_qnn.json` 和 `qnn/llm.mnn`，当前运行按设计 fallback 到 CPU profile；
  同一 LLM-only 输入完成，`total_ms=87854`、RSS `3245264KB`，quality gate 通过，无 native crash。
- 已新增 `tools/build_mnn_qnn_host_tools.sh` 和 `tools/generate_mnn_qnn_artifacts.sh`，用于在具备完整官方
  QAIRT/QNN SDK 的 Linux x86_64 host 上先构建 MNN `generateIO`/`compilefornpu`，再生成
  `config_qnn.json` 和 `qnn/llm.mnn`。生成脚本会显式检查：
  `QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator`、
  `qnn-context-binary-generator`、`lib/x86_64-linux-clang/libQnnHtp.so`、
  `libQnnHtpNetRunExtensions.so`，以及 Linux x86_64 版 MNN `generateIO`/`compilefornpu`。
- 2026-07-09 server spike：用户上传的 QAIRT/QNN `v2.48.40.260702.zip` 已在 `amphion-119`
  解包到 `/home/ubuntu/meetnote-qnn-spike/sdk/qairt/2.48.40.260702`。该 SDK 具备完整 host tools 和
  Android V81 runtime。SDK 内 `QnnTypes.h` / backend-aware config / QAIRT docs 一致指向
  SM8850 使用 `SOC_ID=87`、`DSP_ARCH=v81`；不要使用 Android `/sys/devices/soc0/soc_id` 风格的 `660`
  作为 MNN `--soc_id`。
- MNN host tools 在 Linux x86_64 上必须同时打开 `MNN_QNN_CONVERT_MODE`：
  `-DMNN_QNN=ON -DMNN_QNN_CONVERT_MODE=ON`。只开 `MNN_QNN=ON` 时 `compilefornpu` 会报
  `Can't Find type=32 backend`，后续 `npu_convert.py` 找不到 graph 目录。服务器构建产物位于
  `/home/ubuntu/meetnote-qnn-spike/work/MNN/build_qnn_host/`。
- QNN SDK 的 `qnn-model-lib-generator` 调用 `clang++`，在该服务器上需要额外暴露 libstdc++ 头文件和库：
  `CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:...`，
  `LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:...`。脚本已处理这个环境注入。
- Qwen3-4B QNN 离线产物已完成生成。最终成功配置：
  `QNN_SDK_ROOT=/home/ubuntu/meetnote-qnn-spike/sdk/qairt/2.48.40.260702`、
  `SOC_ID=87`、`DSP_ARCH=v81`、`MAX_HISTORY_TOKEN=0`、
  `MNN_QNN_HOST_BUILD_DIR=/home/ubuntu/meetnote-qnn-spike/work/MNN/build_qnn_host_llm`。服务器根分区空间不足时，
  `npu_convert.py` 在 graph tar 合并阶段会报 `No space left on device`；实际成功转换是在
  `/dev/shm/meetnote-qnn-work` 下完成，再持久化回
  `/home/ubuntu/meetnote-qnn-spike/work/local-assets/models/Qwen3-4B-MNN/`。
- 成功产物已经同步到本机 ignored `local-assets/models/Qwen3-4B-MNN/`：
  `config_qnn.json`、`qnn/llm.mnn`、38 个 `qnn/graph*.bin`。本地完整性检查结果：
  `config_qnn.json` 可解析，`qnn/llm.mnn` 引用 38 个 graph，缺失引用数为 0，`qnn/` 约 7.3 GiB。
  成功转换日志保留在 `/home/ubuntu/meetnote-qnn-spike/generate-mnn-qnn-artifacts-shm-success.log`。
- 无线 ADB 安装验证：QNN APK 构建成功并安装到 Vivo `V2505A`，APK 内含 QNN 版 `libMNN.so`、
  `libmeetnote_mnn_llm.so`、`libQnnHtp.so`、`libQnnHtpPrepare.so`、`libQnnHtpV81Stub.so` 和
  `libQnnSystem.so`。设备 `/data/local/tmp/meetnote_mnn_eval/models/Qwen3-4B-MNN/qnn` 已有 38 个
  `graph*.bin`，runtime 目录已清理为 QAIRT/QNN 2.48 V81 host/DSP 两段布局。
- LLM-only QNN run 结果：未完成生成，状态为 `remote LLM service disconnected`，不能比较速度。排查进展：
  先后修正 `ADSP_LIBRARY_PATH` 分隔符、改用设备 V81 skel 目录、当时误用
  `uses-library libcdsprpc.so`、打开
  `jniLibs.useLegacyPackaging=true`，并临时打包 `libcdsprpc.so`/`libadsprpc.so` 及其系统依赖后，
  `libcdsprpc.so`/`loadRemoteSymbols` 层错误消失；最终仍在 MNN `Llm::tuning` / `Llm::forwardRaw` 预热阶段
  SIGSEGV。日志留档：`build/qnn-spike/qnn-run9-hardware-dep/logcat.txt`。
- 2026-07-10 QNN C64 结果：当前可声明“实验 QNN/HTP 路径在该输入上可用”，但仍不能作为产品默认。
  关键变化不是采样参数，也不是 `vtcm_mb`，而是把离线 QNN graph 的实际 `chunk_size` 从 128 降到 64。
  MNN 当前 `generate_llm_qnn.py` 虽然有 `--chunk_size` 参数，但脚本内 `makeIOJson(args, 128, ...)`
  仍硬编码 128；本轮先在服务器临时改成 `makeIOJson(args, args.chunk_size, ...)` 后重新生成
  `/dev/shm/meetnote-qnn-work/models/Qwen3-4B-MNN-NPU-C64`，`config_qnn.json` 中确认
  `chunk_limits=[64,1]`，`qnn/` 38 个 graph、约 2.0 GiB。仓库
  `tools/generate_mnn_qnn_artifacts.sh` 现已把 `CHUNK_SIZE=64` 设为已验证默认值，并只在临时 generator
  副本中替换该硬编码，不再要求污染 MNN source；生成后还会校验 `chunk_limits` 和 graph 数量。
- C64 前的失败边界：
  - 原始 NPU export 和 VTCM4 都能生成文本，但 app 输出 JSON 不合法；logcat 有
    `HAP_compute_res_acquire`、`contextFromBin Failed code: 5005`、`map result: 8003`，主要发生在后部 context。
  - `vtcm_mb=4` 没有消除上述 HTP context 错误，只把首轮 total 降到约 22.9s，仍因 JSON/repair 失败不可用。
  - `weight_sharing_enabled=false` 让 `qnn/` 增到约 4.0 GiB，并出现大量
    `Could not allocate persistent weights buffer` / `Failed to initialize graph memory`，比默认 weight sharing 更差。
  - `--lm_quant_bit 8` 的 LM8 变体输出乱码并触发 768 token 上限，不是修复方向。
- C64 app A/B 结果使用同一 `build/llm-e2e/run1/input.json`：
  - CPU 基线：`status=done`，quality gate 通过，`total_ms=70848`、`prefill_ms=47872`、
    `decode_ms=18657`、`decode_tokens=219`、RSS `3220960KB`。留档在 `build/qnn-spike/cpu-baseline/`。
  - QNN C64 run25：首轮 raw 仍是不合法 JSON，但 repair 成功；quality gate 通过，`total_ms=45823`、
    `prefill_ms=15608`、`decode_ms=30189`、`decode_tokens=474`、RSS `504100KB`。留档在
    `build/qnn-spike/qnn-run25-c64/`。
  - QNN C64 run26 repeat：复现 `status=done` 和 quality gate 通过，`total_ms=45915`、
    `prefill_ms=15672`、`decode_ms=30229`、`decode_tokens=474`、RSS `505700KB`。留档在
    `build/qnn-spike/qnn-run26-c64-repeat/`。
  - C32 触发 HTP SSR 后恢复 C64 的 run27：再次 `status=done` 和 quality gate 通过，
    `total_ms=49068`、`prefill_ms=17434`、`decode_ms=31608`、`decode_tokens=474`、RSS `505220KB`。
    留档在 `build/qnn-spike/qnn-run27-c64-recovery/`，证明设备恢复后 C64 基线仍可用。
  - 与 CPU 相比，C64 的总时延约 `1.54x` 更快，prefill 约 `3.05x` 更快，RSS 约 `6.37x` 更低；
  - 2026-07-18 长输入复查发现 C64 配置虽然写了 `repetition_penalty=1.1`，但没有显式
    `mixed_samplers`。当前 MNN 默认 mixed pipeline 不包含 `penalty`，因此该值会被读取却不会执行；
    120s 会议的首轮和 repair 都退化为重复 `0` 并打满 768 tokens。修复必须先把 `penalty` 显式加入
    mixed pipeline，再评估 penalty 数值；单纯调大 `repetition_penalty` 或 token 上限不会生效。
    但因为首轮输出仍需 repair，decode 总量和 decode 耗时比 CPU 更高，产品化前必须继续压输出格式稳定性。
- C64 HTP 侧证据：run26 用 `logcat -b all` 清空后重跑，app 远端进程成功加载
  `libQnnHtpV81Skel.so`，创建 CDSP user PD，并通过 `remote_handle64_open` 打开 skel；未再出现
  `HAP_compute_res_acquire`、`contextFromBin`、`Failed to initialize graph memory`、SIGSEGV 或
  `Fatal signal`。但 run26/run27 都仍出现 2 次 shared-weight map `8003`，对应
  `Context 38 failed` / `Context create from binary failed ... err 1002`，随后运行时继续执行并成功生成；另有
  大量 FastRPC `Error 0x80000448 ... unable to unmap` / `remote_handle_control` warning。这些不是可忽略的
  干净日志，需作为 HTP runtime 边界继续观察。standalone `llm_demo` C64 也能生成文本并复现 skel/user-PD
  加载，恢复验证在 `build/qnn-spike/standalone-qnn-c64-recovery/`。standalone 运行时把 host so 目录放入
  `LD_LIBRARY_PATH`，`ADSP_LIBRARY_PATH` 只指向 `hexagon-v81/unsigned`；用冒号把 host 目录拼进
  `ADSP_LIBRARY_PATH` 会被 FastRPC 当成一个不存在的路径并导致 `Failed to load skel`。
- C32 失败边界：在同一 MNN/QNN SDK、模型和设备上用实际 `chunk_size=32` 生成的 38 个 graph 能完成 host
  conversion，但 standalone 加载约 13.7s 后报 `contextFromBin Failed code: 1003`、
  `Context create from binary ... err 5005`、`This context binary is invalid ... 1007`，随后 HTP SSR 和
  `SIGSEGV`。因此不进入 app 测试；日志在 `build/qnn-spike/standalone-qnn-c32/`，本机 ignored 产物在
  `local-assets/models/Qwen3-4B-MNN-NPU-C32/`。生成脚本默认拒绝 `<64`，只有显式
  `ALLOW_UNVERIFIED_CHUNK_SIZE=true` 才允许继续隔离实验。
- 历史失败边界仍需保留：`MAX_HISTORY_TOKEN=2048` 会让 QNN attention path 早停在
  `MNN_QNN: Not registered type 299 ... Attention`；未打开 `MNN_QNN_CONVERT_MODE` 会让 `compilefornpu`
  报 `Can't Find type=32 backend`；磁盘不足会产生不完整 graph tar。只有完整性检查通过的 `qnn/`
  目录可以进入 APK/设备验证。
- 本机 macOS arm64 预检失败是预期结果：`QNN offline conversion must run on Linux x86_64`。Qualcomm
  Software Center / QPM 公开页面和社区文档都指向 QAIRT/QNN SDK 下载入口，但直接下载
  `Qualcomm_AI_Runtime_Community` ZIP 会被重定向到 `apigwx-aws.qualcomm.com` 并返回 `403`；QPM
  catalog API 也需要 Qualcomm OAuth token。不要把 MNN 公开 2.37 runtime 包当成完整 host SDK。
- 结论：官方 QAIRT/QNN SDK、`SM8850` 的 `SOC_ID=87`、Linux x86_64 host conversion 和 Vivo USB
  上的 C64 QNN/HTP app 路径均已打通；同一输入重复验证的总时延约比 CPU 快 `1.44x-1.54x`，prefill
  快约 `2.75x-3.05x`，RSS 低约 `6.37x`。但首轮 JSON 仍依赖 repair，且 HTP 仍有 shared-weight map 和
  FastRPC unmap 错误，所以只保留为显式 spike，CPU 继续作为产品默认与回退边界。

## 2026-07-19 QNN 4B 文本单测与 Vivo 前台约束

- Android manifest 中 vendor native 依赖应声明为
  `<uses-native-library android:name="libcdsprpc.so" android:required="false" />`，不是 Java
  `<uses-library>`。
- `libQnnHtpV81Skel.so` 不能经过 Android Gradle Plugin strip。原始 SDK SHA-256 为
  `a0b9750d...a665a32`，默认 strip 后变为 `90a2dce8...19e52`；加入
  `jniLibs.keepDebugSymbols += "**/libQnnHtpV81Skel.so"` 后 APK 内哈希与 SDK 原文件一致。
- Vivo `V2505A` 会冻结不在前台的 QNN app。实测文本单测加载到 `32/38` contexts 时前台被切到其他 app，
  QNN 不再推进；重新 `am start` 把 MeetNote 拉回前台后立即继续到 `38/38`。相同设备状态下 shell
  `llm_demo` 可正常生成，因此这不是模型损坏或必须重启设备。设备验证期间应保持 MeetNote 可见和亮屏；
  不要用反复重启手机掩盖后台冻结。
- compact prompt 原先给出空 JSON 模板，4B 会直接抄写空字段，并把
  `confirmed|proposed|unclear`、`high|medium|low|unknown` 当成字段值。现改为字段合同，明确枚举只能选择
  单值且禁止空字符串；解析边界还会把 `_summary` 归一为 `overview`，并恢复输入中的原始会议标题。
- 实际包 `ai.meetnote.demo` 的 512-token 文本单测通过：instrumentation `OK (1 test)`，最终
  `status=done`，`prompt_tokens=866`、`decode_tokens=286`、`prefill_ms=3632`、
  `decode_ms=18267`、`total_ms=21911`、RSS `345184KB`。输出通过完整 schema、标题、overview、
  数组类型和未知顶层字段检查，无模板枚举复制和重复退化。产物在
  `build/qnn-spike/qnn-text-probe-20260719/qnn-schema-normalized-final-pass/`。

## 2026-07-21 长会议纪要压测

- 压测语料由公开会议材料的议题结构重新合成，不复制原文；覆盖明确决议、未决提议、责任人、截止日期、
  多类风险，以及会议末尾新增的风险和决议。设备输入位于
  `/data/local/tmp/meetnote-summary-stress/`，生成脚本为
  `tools/generate_meeting_summary_stress_inputs.mjs`。
- QNN mixed sampler 已显式启用 `penalty -> topK -> topP -> temperature`，参数为
  `repetition_penalty=1.1`、`top_k=20`、`top_p=0.8`、`temperature=0.2`。这能消除原先
  `utt-1-1-1` 一类 ID 循环，但不能保证 4B 模型稳定生成完整 JSON。
- 根因边界：QNN 局部抽取会把“百分之九十七”改成 `17%/7%`，并偶发编造缓解措施；因此不能继续靠
  parser alias 或 repair 修补。QNN quality-agent 改为确定性分块取证，只让模型用最多 128 tokens 做一次
  全局标题/摘要归纳，结构化条目和最终渲染均从原始 evidence 恢复。模型失败、截断或 JSON 不合法时直接
  使用确定性结果，不再追加一次昂贵 repair。质量门禁会拒绝原文不存在的数字。
- 96 句话、5497 字、约 45 分钟档真机通过，instrumentation `OK (1 test)`，端到端 `40.45s`；QNN
  `prompt_tokens=2296`、`decode_tokens=34`、`total_ms=20343`、RSS `487256KB`。前三项明确决议、
  `utt-94` 末尾风险和 `utt-95` 末尾决议均保留。
- 320 句话、17357 字、约 149 分钟极限档真机通过，instrumentation `OK (1 test)`，端到端
  `43.25s`；QNN `prompt_tokens=2205`、`decode_tokens=128`、`total_ms=25566`、RSS `497008KB`。
  最后主题和决议均引用 `utt-319`，末尾风险引用 `utt-318`。全过程 MeetNote 保持前台，未发生手机重启。
- 验收产物位于 `build/qnn-spike/meeting-summary-stress-20260721/long-96-grounded-v3/` 和
  `build/qnn-spike/meeting-summary-stress-20260721/extreme-320-grounded-v1/`。

## QNN 模型网络分发

2026-07-12 验证了“模型塞进单 APK”不可作为发布方式。C64 QNN 的必要资产包括
`llm.mnn.weight`、`embeddings_bf16.bin`、38 个 graph、配置/tokenizer 和 V81 DSP runtime，合计约
4.74 GiB；缺 weight 会在加载阶段失败，缺 embedding 虽能进入 QNN，但只会生成重复的 `<`。完整资产用
Android APK 打包会超过 ZIP/Bundle 边界并在 `packageDebug` 报 `integer overflow`，高强度 XZ 压缩又会让
本机打包耗时不可接受。

发布包改为“小 APK + HTTPS 清单”：

```bash
./gradlew --no-daemon --no-watch-fs :samples:meeting-demo:assembleDebug \
  -PmeetnoteLlmQnn=true \
  -PmeetnoteQnnModelManifestUrl=https://host/prefix/manifest.tsv \
  -PmeetnoteQnnModelManifestSha256=<manifest-sha256>
```

- 清单格式为 `relative-path<TAB>byte-size<TAB>sha256`，对象路径相对清单 URL 解析；APK 还会固定清单自身的
  SHA-256，避免清单和对象校验值被一起替换。
- APK 只包含 QNN host native 库，不包含对象存储 AK/SK；模型对象必须提供匿名 HTTPS GET 和 Range。
- 首次启动下载到 app 私有 staging 目录，支持 `.part` 断点续传；每个文件按大小和 SHA-256 校验，整套完成后
  才替换正式模型和 DSP 目录。
- QNN 模型校验必须检查 `chunk_limits[0]=64` 和完整的 `graph0..37.bin`，不能只检查文件名存在。C128 目录
  虽然资产齐全，但在当前 Vivo/SM8850 上会报 `context binary invalid (1007)` 并使远端 LLM 进程崩溃。
- 清单必须最后上传。否则客户端可能在大对象尚未上传完成时拿到一份看似有效的版本。
- 模型下载不能阻塞 ASR：ASR preload 完成后录音按钮必须可用，LLM 模型继续在后台下载。
- 本轮正式 C64 清单位于
  `https://meetnote.tos-cn-guangzhou.volces.com/qnn/v81/c64-2026-07-12/manifest.tsv`，清单 SHA-256 为
  `11706dfce267f4bede1193206e1eae4d98b7960188c7afdb81e94c79e372a067`。云端长期 AK/SK 不得写入
  `local.properties`、源码、APK 或上传清单；上传完成后应轮换聊天中暴露过的凭据。
- 火山 TOS 默认 endpoint 会对 APK 内容返回 `ApkDownloadForbidden`，即使对象名不带 `.apk` 也会按内容识别；
  直链 APK 必须给桶绑定 CNAME。当前公开下载先使用无压缩 ZIP，解压后安装其中 APK：
  `https://meetnote.tos-cn-guangzhou.volces.com/releases/MeetNote-QNN-v81-online-debug.zip`。
- USB 实测先从旧 C128 版本复用 2.95GB 相同权重，再下载 C64 graph；最终 app 私有模型校验为
  `chunk_limits=[64,1]`。同一 4 条转写连续两次 QNN 任务均为 `done`，最终一次
  `prefill_ms=7595`、`decode_ms=10075`、RSS `423684KB`，logcat 无 crash、SIGSEGV、1007 或 remote
  service disconnect。验收产物在 `build/exports/qnn-c64-url-e2e/` 和
  `build/exports/qnn-c64-url-final-logcat-all.txt`。

### 单 APK 的 CPU 保底

当前 C64 离线 graph 不是“所有高通芯片通用”的资产：它只在 `SM8850`、Hexagon V81 和当前 QAIRT/QNN
runtime 组合上完成过实机验证。QNN APK 因此使用保守 allowlist；只有 `SM8850 + arm64-v8a` 会选择 QNN，
其余 SoC 在模型加载前直接选择 CPU，避免用 native crash 探测兼容性。

同一个 QNN APK 同时包含 MNN CPU backend，但不内置 CPU 权重。CPU 保底模型保持同规格 4B，固定为官方公开的
`taobao-mnn/Qwen3-4B-MNN` revision `462ee25d099fb7bcc690c54b38481f99afccc555`，在 NPU 不可用或 QNN
资产/运行失败时从 Hugging Face `resolve` HTTPS URL 下载约 2.53 GiB。客户端支持 `.part` 断点续传，并逐文件
校验固定 size 和 SHA-256；校验完成后原子替换到 app 私有目录。
CPU 模型不上传 MeetNote TOS，也不把任何云存储凭据放进 APK。

QNN 远端任务若返回失败或超时，应用会把当前 QNN manifest SHA 标记为不可用，按需下载 4B CPU 模型后自动
重试该条纪要；更换 manifest SHA 后会重新尝试 QNN。QNN 资产下载失败只在当前进程回退 CPU，下一次启动仍会
重试 QNN，避免一次网络抖动永久关闭 NPU。NPU 正常设备启动时不会预先多下载一份 2.53 GiB CPU 模型。界面会
明确提示“NPU 不可用，当前使用 CPU 4B 模型”。主要权衡是 Hugging Face 在部分网络环境可能不可达；下载完成后
不再依赖网络。

USB 验收结果：同一 APK 在恢复 QNN health 后，job
`llm-fa07cc48-0a09-4bfe-bf93-9427169423ba` 为 `done`，`total_ms=18589`、RSS `424240KB`；把当前
manifest SHA 标记为不可用后，同一 APK 显示 CPU fallback 提示并使用公开 4B CPU 模型，job
`llm-fa8a00d9-063a-4845-b04d-56e33eef0a4b` 为 `done`，`total_ms=56253`、RSS `3120680KB`。测试后已清除
设备上的模拟熔断状态，恢复默认 QNN 选择。截图、result 和 metrics 位于
`build/exports/cpu-4b-fallback/`。固定 revision 的 4B weight URL 已验证支持 HTTP `Range`，返回 `206`。
