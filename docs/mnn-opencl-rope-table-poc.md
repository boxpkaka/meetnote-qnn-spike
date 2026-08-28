# MOSS OpenCL Attention + INT32 RoPE Table PoC

日期：2026-08-26

## 结论

`INT32 position -> Gather(precomputed FP32 cos/sin table) -> OpenCL normal FusedAttention` 已经通过长位置正确性和 case 2 完整语义门禁。在该精确路径上把 OpenCL host staging upload 改为 nonblocking 后，case 2 decode 进一步从 `317.406 s` 降到 `273.113 s`，total 为 `329.896 s / RTF 1.09965`，peak PSS 为 `1,041,199,104 bytes`。输出与修复版正确性基线逐字一致，但 RTF 仍未达到 1.0，暂不能作为发布候选。

这条路线应保留为新的优化基线。RTF 1.0 只剩 `29.896 s` 缺口；下一轮不再讨论 RoPE 正确性，而应优先定位 QNN/OpenCL 边界同步、host 调度和 prefill，再决定是否承担 Qualcomm-native decoder 的结构迁移成本。

## 实现与结构门禁

- exporter 通过 `MNN_ROPE_TABLE_SIZE=10240` 预计算 RoPE cos/sin table，position 保持 INT32 并直接用于 `index_select`。
- converter 只对 `/rotary/Gather_output_0` 跳过负索引保护的 float `ADD/MOD`，未启用全局 optimize level 2。
- 28 层 `FusedAttention`、RoPE reshape/gather 和输入 slice 留在 MNN；projection、FFN、lm_head 仍走 QNN context graph。
- decoder wrapper SHA256：`305965f5cfa93ca2007fda58368a68e71d157ea9eb96349de9d6cea6272e6c67`。
- graph0 SHA256：`8ade6fce565797d9db2f2dadd73384f99e570f423d91e2984c27642cc71ab628`。
- graph30 SHA256：`4fdc7630aa14f9d6cb7f35a9c10338bb92d39acf11a479775c748d6b17904e0f`。

模型结构确认 RoPE Gather 直接消费 INT32 position，不再出现 position Cast/Mul/Cos/Sin 或 float ADD/MOD。

## 服务器正确性

- 48-step teacher：对 baseline MNN top-1 `48/48`，cosine mean/min `0.9977543/0.9780459`；对 HF BF16 top-1 `48/48`，cosine mean/min `0.9998009/0.9990083`。
- case 2 3240-step sparse100：34 个采样点 top-1 `34/34`；对正确 MNN CPU cosine mean/min `0.9999988/0.9999960`；无 2048/4096 位置断崖。

## 真机结果

设备：Vivo V2547A / SM8850。输入 case 2 SHA256：`ce6f3b65deb8fe5545e7ba20ff02a18bafe6e72354d2aad5e4a25420e2535e0d`。

| Case | 结果 | Decode | Total / RTF | Peak PSS |
| --- | --- | ---: | ---: | ---: |
| 32 token | 正文和标点恢复为 CPU baseline；仅首时间戳差 `0.03 s` | `3.459 s` | `67.595 s / 0.225` | `757,763,072` |
| 256 token | 完整跨过旧 normal 路径约 121-token 失真点 | `19.062 s` | `73.758 s / 0.246` | `698,780,672` |
| R8007 300s 附加回归 | 自然结束、无重复、内容完整 | `441.017 s` | `500.728 s / 1.669` | `1,183,895,552` |
| case 2 300s | 120 段、3704 tokens、完整到 300.00s | `317.406 s` | `385.581 s / 1.285` | `1,042,791,424` |
| case 2 + nonblocking upload | 与上一正确结果逐字一致 | `273.113 s` | `329.896 s / 1.09965` | `1,041,199,104` |

case 2 与正确 MNN CPU reference 的 120 段 speaker/text 全部一致；去掉时间戳后 raw text byte-identical。时间戳 start/end MAE 为 `0.0045/0.00183 s`，最大绝对差 `0.04 s`。充电宝、手机壳、贴膜和品牌限量赠品讨论全部保留，无重复尾部。

相对当前 v23 真机正确性基线：

- decode `474.140 -> 317.406 s`，下降 `33.1%`；
- total `562.590 -> 385.581 s`，下降 `31.5%`；
- peak PSS `3.488 -> 1.043 GB`，下降约 `70%`。

## 剩余性能缺口

RTF 1.0 要求 total 不超过 `300 s`。nonblocking upload 版本还需下降 `29.896 s`；若 encoder、prefill 和其他开销不变，decode 需要从 `273.113 s` 降到 `243.217 s`，即再降低约 `10.95%`。

历史 W8A16 单独把 decode 从 `474.140 s` 降到 `439.453 s`，只节省 `34.687 s`。即使与本次收益完全相加，估算 total 仍约 `350.9 s / RTF 1.17`，因此 W8 只能作为组合项，不能单独闭合目标。

## 精确 decode profile

为避免 Android logd 限流丢失逐 kernel 日志，profile build 直接读取 OpenCL event time，并把 28 层 decode Attention 的 `rearrange / QK / softmax / QKV` 汇总到同一份 JSON。该 build 同时保留 QNN input、execute/sync、output 和 state update 计时。

| 轨迹 | Decode | QNN | OpenCL Attention | 未记账 |
| --- | ---: | ---: | ---: | ---: |
| 32 token | `2508.47 ms` | `1095.76 ms` | `423.59 ms` | `989.11 ms` |
| 256 token | `20851.0 ms` | `6512.04 ms` | `3488.79 ms` | `10850.2 ms` |

256-token 轨迹共有 `7168 = 256 x 28` 次 Attention，计数完整。Attention 内部为：

- `QKV`: `2263.43 ms`，占 Attention `64.9%`；
- `QK`: `1060.82 ms`，占 `30.4%`；
- `softmax`: `125.77 ms`，占 `3.6%`；
- rearrange: `38.77 ms`，占 `1.1%`。

GPU event 查询会强制等待，profile 版 256-token decode 比无 GPU profile 的同路径 `18871 ms` 慢约 `1980 ms`。因此 profile wall time 不能代替 release 性能；把这部分测量扰动扣除后，当前约 4K history 的结构占比约为 QNN `34.5%`、OpenCL Attention `18.5%`、其余 OpenCL/host/backend 边界 `47.0%`。即使按 history 近似线性外推到 case 2 的平均长度，Attention 约占完整 decode `23%`，仍低于“单独迁移或重写 Attention”的 `35%` 门槛。

由此调整优先级：先完成 Exact RoPE + W8A16，并继续拆解未记账的 backend 切换、host wait 和其他 OpenCL 小算子；不再把 Attention fusion 当作第一优化项。若后续仍需改 Attention，应先优化 QKV 聚合而不是 softmax。

Profile artifacts：`phone-case2-gpuprofile32-v5/`、`phone-case2-gpuprofile256-v1/`。首个完整 profile build 漏开 `MNN_WITH_PLUGIN`，会在加载 `qnn/plugin/op` 时失败；修正后的完整匹配 runtime 已单独归档，未污染 release baseline。

### 全 OpenCL kernel 与 backend copy profile

后续 profiler 不再在每个 Attention 内直接等待 event，而是在 OpenCL runtime 的统一 event 回收点汇总全部 kernel。256-token 结果为 decode `19343.8 ms`、QNN `6534.92 ms`、全部 OpenCL kernel `3494.75 ms`、未记账 `9314.18 ms`，占比分别为 `33.8% / 18.1% / 48.2%`。其中非 Attention OpenCL kernel 只有 `2.06 ms`，可以排除“其他 OpenCL 小算子”作为主要缺口；新 profiler 相对无 profile 的同路径只增加约 `473 ms`，也避免了旧版约 `1980 ms` 的逐 event 等待扰动。

再对 OpenCL backend copy 做 wall-time 统计，256 token 共发生 `22267` 次 CPU→OpenCL 和 `7679` 次 OpenCL→CPU，即每 token `86.98 / 30.00` 次。对应 wall time 为 `4939.94 / 5668.39 ms`。copy-from 会等待此前入队的 Attention kernel，因此这些 wall time 与 `3494.75 ms` kernel time有重叠，不能直接相加；但调用数稳定对应每层约 `3` 次 Q/K/V 上传和 `1` 次 Attention 输出下载，外加每 token 约 `3/2` 次固定往返。剩余性能主矛盾由此从 Attention 算力进一步收敛为 QNN→host→OpenCL 的逐层 materialization 和同步。

下一实现优先验证 Q/K/V 聚合传输：让每层三个 QNN 输出先形成单个 packed QKV，再一次上传并由 OpenCL Attention 按 offset 读取。该方向保持 FP16 KV 和现有精确 Attention 数学不变，先用 32/256 token 测 copy 调用数和端到端收益；若只能增加 memcpy 而不能减少同步，立即止损。Artifacts：`phone-case2-gpuprofile-total32-v1/`、`phone-case2-gpuprofile-total256-v1/`、`phone-case2-gpuprofile-copy32-v1/`、`phone-case2-gpuprofile-copy256-v1/`。

### Exact RoPE + W8A16 门禁

远端已完成新的 W8A16 export、移除 activation quant 描述，并生成 31 个 SM8850/V81 context；decoder weight 从 W16 的约 `1.1 GiB` 降至 `578 MiB`。首次 QNN phase3 因 host PATH 缺 `clang++`，续跑又缺 `libc++.so.1`，补齐 host toolchain 后复用已编译 graph 成功结束。两者均为服务器构建环境问题，不是 HTP graph 编译失败。

真机 warm 32-token 输出与 W16 前缀一致，decode `2235.06 ms`。256-token decode 为 `19154.7 ms`，反而比同一路径 W16 无 profile 的 `18871 ms` 慢 `283.7 ms`（`+1.5%`），并在早期自由生成出现 `找得了吧` 对 W16 `找得了的` 的 token 分叉。该组合既没有性能收益，也没有保持当前 exact operating point，因此按小门禁止损，不运行完整 case 2。Artifacts：`phone-case2-w8-gate32-warm-v1/`、`phone-case2-w8-gate256-v1/`。

设备组包阶段曾因漏放 `afq/abq` 和 wrapper 外部路径 `qnn/graph*.bin` 触发两个可复现的 missing-file SIGSEGV；补齐 symlink 后同一 graph 正常完成 32/256 token，未见 DSP SSR。后续实验包必须同时校验 wrapper 内 external path、audio QNN context 和 decoder contexts，不能只核对 decoder 文件 SHA。

### Packed QKV 止损

layer 0 packed QKV 将一次 decode 的 CPU→OpenCL upload 调用减少了预期的 `2` 次，256-token 总调用从 `22267` 降到 `21757`；但 concat/slice view 让 OpenCL Attention 时间从约 `3492 ms` 增到约 `3775 ms`，两次相邻运行的 decode 为 `19672/19626 ms`，均慢于 `19116 ms` 基线。输出保持逐字一致，但传输收益被新的 view/slicing 成本覆盖，因此不扩 4 层或 28 层。

### Nonblocking host upload

OpenCL in-order queue 中，Q/K/V staging upload 原来使用 blocking `enqueueWriteBuffer(CL_TRUE)`；同层稍后的 Attention 输出 readback 本来就是依赖屏障，因此上传不需要提前阻塞 host。将普通 float staging upload 改为 `CL_FALSE` 后：

- 32-token decode `2261.58 -> 1804.0 ms`，CPU→OpenCL copy wall time `624.87 -> 122.72 ms`；
- 256-token decode `19116.2 -> 15718.5 ms`，copy wall time `4939.94 -> 1010.65 ms`；
- 完整 case 2 decode `317.406 -> 273.113 s`，total `385.581 -> 329.896 s`；
- 3704 tokens、120 段、prompt IDs、raw text 和 segments 均与精确基线一致，无 crash、SSR、OOM 或 non-finite。

完整 profile 中 QNN execute 为 `97.287 s`，OpenCL kernels 为 `72.560 s`，其余未直接归因时间为 `102.195 s`。OpenCL→CPU wall time包含已入队 kernel 的等待，不能重复相加。Artifacts：`phone-case2-async-upload32-v1/`、`phone-case2-async-upload256-v1/`、`phone-case2-async-upload-full-v1/`。

### 次级调度与 OpenCL 配置门禁

同一 256-token 文本前缀在所有实验中 SHA256 一致。`taskset f0` 只偶发缩短 prefill，decode 无稳定收益；OpenCL record-op 基本持平，record-batch 变慢；显式 buffer mode 基本持平；`precision=low` 变慢。profiling on/off 与快慢不再稳定相关，约 `32 s` 和 `42 s` 的 audio encoder 两档更符合连续运行升温造成的设备状态差异，后续完整性能验收必须从冷机状态开始并保存 thermal/battery。

QNN graph 采样显示 `graph1_30` 对应约 `313 MB` 的 lm_head context，单次 execute mean `5.54 ms`，明显高于其余单 graph；但全量 W8A16 已证明当前 W8 路径既不更快也不保持 exact 输出，因此只把该结果作为后续定向量化证据，不直接重开全量 W8。

### C128 prefill 止损

远端生成了独立的 `[128, 1]` decoder 和 31 个 SM8850/V81 context，未覆盖 C64 基线。首次设备启动因
wrapper external context 路径仍按进程 cwd 解析而找不到 `qnn/graph*.bin`；改为从独立 decoder 目录运行并
补齐共享资产 symlink 后正常完成。warm 32-token 的 prompt IDs 和 raw text 与 C64 完全一致，但 prefill
`21.464 -> 22.768 s`、decode `1.804 -> 1.956 s`、total `55.037 -> 56.527 s`，peak PSS 也从
`701.6 -> 843.9 MB`。正确性通过但性能和内存均回退，因此不运行 256/full，也不继续生成 C256。

### OpenCL decode kernel autotune 与 GQA 止损

设备 autotune 已在现有 QKV 四个候选中选择 `matmul_qkv_decode_b8 + unroll8 + LWS(8,1)`；约 4K history
下候选 cost 为 `299`，其余 b8/unroll4、b4/unroll4、b4/unroll8 分别为 `406/879/938`，没有遗漏一个
显然更快的现成组合。

MOSS 实际 decode shape 为 `num_head=16 / kv_num_head=8 / head_dim=128`，即 GQA group size 2。新增实验
kernel 让同一 KV head 的两个 query heads 共享 K/V 读取：

- 第一版串行 QKV 聚合 cost `1154`，明显慢于原 kernel，立即淘汰；
- unroll8 QKV 将 cost 降到 `305`，32/256-token 输出均逐字一致，但 256-token decode
  `15.719 -> 16.396 s`；
- 只启用共享 QK 时，256-token 输出仍逐字一致，但 decode `15.719 -> 16.253 s`。

带宽减少没有抵消寄存器压力、并行度下降和调度成本，因此这组 GQA kernel 不扩 full case。Artifacts：
`phone-case2-c128-smoke32-v4/`、`phone-case2-c128-smoke32-v5-warm/`、
`phone-case2-kernel-tuner-default32-v1/`、`phone-case2-gqa2-qk-gate256-v1/`、
`phone-case2-gqa2-qkv-unroll-gate256-v1/`。

### QNN/OpenCL DMA-BUF 共享内存可行性

设备扩展探针确认 Adreno 840 同时暴露 `cl_qcom_ext_host_ptr`、`cl_qcom_dmabuf_host_ptr` 和
`cl_khr_external_memory_dma_buf`；不暴露 `cl_qcom_ion_host_ptr` 和 `cl_arm_import_memory`。因此这里应走
DMA-BUF，而不是按旧 ION extension 名称判断为“不支持”。

进一步在真机用 `libcdsprpc.so` 的 `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED)` 分配
`4096 bytes`，获得 fd `8`，再以 `CL_MEM_DMABUF_HOST_PTR_QCOM + CL_MEM_EXT_HOST_PTR_QCOM` 直接导入
OpenCL。`clCreateBuffer` 返回 `CL_SUCCESS`；CPU 写入后 GPU 原地计算、`clFinish` 后 CPU 直接读取连续两轮均
逐项正确，没有 staging copy、crash 或 SSR。这证明同一块 rpcmem 可以被 QNN 注册并由 OpenCL kernel 原地访问，
shared-buffer 路线在当前手机上具备底层能力，不再只是接口猜测。

该探针还不能证明现有 MNN hybrid decoder 已经零拷贝。当前 QNN plugin 仍把 internal RPC output memcpy 到
MNN tensor，OpenCL backend 又将其上传；反向则 blocking readback 后再 memcpy 进下一个 QNN RPC input。下一步
必须让两端绑定同一个 rpcmem allocation，并显式保留 QNN execute、OpenCL event/`clFinish` 的所有权交接。
仅把已经 memcpy 过的 host tensor 再导入 OpenCL，不会消除逐层 materialization。

首个集成 PoC 只改 layer 0 的 Q/K/V 输出边界：由共享 allocator 生成 rpcmem，分别注册到对应 QNN context，
再由 OpenCL 通过 DMA-BUF fd 导入并直接作为 Attention 输入。先验证 32/256-token exact 输出、copy 调用数和
wall time；若 QNN context 无法稳定绑定外部 buffer，或仍需额外 device copy 才能执行，则停止扩 28 层，转向
同 backend 的 4-layer native shard。反向 Attention output 到下一 QNN input 等首边界确有收益后再做。

Artifacts：`shared-buffer-probe/device-output.txt`、
`shared-buffer-probe/dmabuf-import-v1/device-output.txt`、
`shared-buffer-probe/dmabuf-import-v1/logcat.txt`。

### QNN/OpenCL shared-buffer 集成止损

layer 0 集成进一步区分了 allocator、QNN output binding 和 OpenCL import 三个变量：

- 设备上的 `libdmabufheap.so` 只接受 `system` heap；`qcom,system` 返回 `EINVAL`。直接打开
  `/dev/dma_heap/qcom,system` 又被 shell SELinux 拒绝。`system` heap 虽可分配和 mmap，但以
  `QNN_MEM_TYPE_DMA_BUF` 注册时在 `libQnnHtp.so` 内崩溃，因此不能作为当前 app/shell 路径。
- 使用现有 QNN runtime 同源的 `rpcmem` 和 `QNN_MEM_TYPE_ION` 时，layer 0 output 2 可以作为外部
  memhandle 正常执行。OpenCL 随后用同一 fd 直接导入，1/32/256-token 输出均与 async baseline 精确一致，
  无 QNN execute 错误。
- layer 0 output 3 或 4 单独注册都会让 `graph1_1` 返回 `1100`；不能把三个 Q/K/V 输出全部改为外部
  memhandle。后续层也只有固定子集允许最大 Q 输出使用该方式。
- 兼容层集合为 `1,2,3,4,5,8,11,12,16,19,20,23,24,28`。14 层共享 Q 路径在 32/256 token 上均
  exact，OpenCL 确认导入 14 个 fd，且无 graph execute 错误；但 256-token decode 为 `16164.3 ms`，
  比 async baseline `15718.5 ms` 慢 `445.8 ms`（`+2.8%`），peak PSS 从 `728,895,488` 增至
  `747,848,704 bytes`。

因此 shared-buffer 底层能力成立，但现有多 context hybrid decoder 的外部 output 约束和 import/memhandle
开销抵消了省下的 staging copy。该路线按性能门槛停止，不做 Attention output 反向共享，也不进入 full case 2。
实验补丁和所有失败/成功 artifacts 已归档；远端 MNN 源码随后恢复到实验前版本并完成 `MNN/MNN_CL` 重建。

关键 artifacts：`shared-buffer-probe/dmabufheap-allocator-v1/`、`shared-buffer-probe/dmaheap-ioctl-v1/`、
`phone-case2-rpcmem-output2-direct-256-v1/`、`phone-case2-rpcmem-all-q-direct-32-v1/`、
`phone-case2-rpcmem-safe-q-direct-32-v1/`、`phone-case2-rpcmem-safe-q-direct-256-v1/`。

### Host phase 与真实 history profile

为拆解完整 decode 中约 `102 s` 的未记账时间，独立 profiler 在生成循环、embedding、`forwardVec`、
`forwardRaw`、`Module::onForward`、output map、显式 wait 和 `KVMeta::sync` 周围加入低频聚合计时；同时修正
OpenCL 主 backend 下 QNN plugin 取不到 `KVMeta`、导致所有 QNN 调用错误落入 `0-2048` 桶的问题。实验只使用
独立 runtime，结束后远端源码和构建产物已恢复。

完整 case 2 仍生成 `3704` token、`120` 段，raw text SHA256
`e449995e0b332971d5d44273ebef3d753177c9b98964bfa9310359a7d73fc012`，与 async release baseline 精确一致。
该次冷态附近运行 total `326.576 s`、decode `269.497 s`、RTF `1.08859`；运行波动不替代正式 baseline
`329.896 s / RTF 1.09965`。

Host phase 结果：

- `Module::onForward`: `268.591 s`，占 decode `99.66%`；
- embedding: `0.770 s`；sample: `3.346 s`，位于现有 `decode_us` 计时之外；
- tokenizer + 文本拼接: `17.7 ms`，stream write: `5.5 ms`；
- output map: `3.8 ms`，显式 `wait`: `0.85 ms`，`KVMeta::sync`: `0.76 ms`；
- 生成循环计时残差仅 `5.3 ms`。

因此未记账时间并不是 tokenizer、输出、采样框架、embedding、显式 tensor wait 或 KV bookkeeping；它几乎
全部位于 heterogeneous `Module::onForward` 内，即 QNN/OpenCL 调度、逐层 materialization、blocking readback
和命令提交边界。

修正后的完整 device profile 为 QNN execute `94.038 s`、OpenCL kernel `72.608 s`、其余
`101.801 s`。OpenCL CPU→device wall `15.343 s`，device→CPU wall `110.181 s`；后者包含已入队 kernel 的
等待，不能与 kernel time相加。真实 history 分桶如下：

| History | QNN execute | OpenCL kernel | QNN calls | Attention calls |
| --- | ---: | ---: | ---: | ---: |
| `2048-4096` | `2.353 s` | `1.325 s` | `3069` | `2772` |
| `4096-6144` | `49.831 s` | `35.350 s` | `63488` | `57344` |
| `6144-8192` | `41.515 s` | `35.933 s` | `48236` | `43568` |

实际轨迹只到约 `7700`，没有进入 `8192-10240`。晚段 Attention 单层单 token kernel time 从约
`0.616 ms` 增到 `0.825 ms`，是目前最清楚的随 history 增长项。Artifacts：
`host-phase-profile/`、`phone-case2-host-profile32-v2/`、`phone-case2-host-profile256-v1/`、
`phone-case2-host-profile-full-v1/`。

### FlashDecode、HMX lm_head 与 static decoder 止损

单 pass online-softmax FlashDecode 已按 layer 0 门禁验证。exact-math 版本保持输出一致，但累计约
`669-712 ms`，慢于原 kernel 约 `424 ms`；subgroup 版本还触发设备侧崩溃，因此没有扩到 28 层。

定向 HMX W8A16 `lm_head` context 经 QNN profile 确认真正使用 HMX，standalone latency 约从
`38.93 ms` 降至 `20.33 ms`。集成 256-token 只减少约 `0.533 s` decode，并在早期 token 出现分叉，既不能
闭合约 `29.896 s` 缺口，也没有保持 exact operating point，按门槛停止。

Qualcomm-native static decoder 的 FP16 c8192 单层约 `3.353 ms/token`。按 28 层和 case 2 轨迹外推，decode
约 `277.9 s`，已不优于当前 hybrid 的 `273.113 s`。native 静态 KV、INT32 position、RoPE table、单层和
4-layer 扩展的可行性已经验证，但在缺少官方 HMX/Genie LLM export 资产时没有全量替换的性能依据。

### HMX W8A16 audio encoder 止损

远端使用 QAIRT `2.48.40` 生成 per-channel W8/A16、FP32 IO 的 HMX front/back context。sample 0 数值对齐为：
front cosine `0.9976072`、isolated back `0.9955181`、完整链路 `0.9937041`。真机 standalone 10-run 中：

- front `31.868 -> 5.937 s`；
- back `31.043 -> 4.880 s`。

但集成结果没有把 standalone 收益完整兑现：

| Case | Encoder | Prefill | Decode | Total / RTF | 质量 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline 256 | `32.458 s` | `20.515 s` | `16.081 s` | `69.438 s` | 基线 |
| HMX audio 256 | `6.414 s` | `20.314 s` | `17.428 s` | `44.542 s` | 正文一致，时间戳轻微漂移 |
| baseline full | `32.372 s` | `20.958 s` | `273.113 s` | `329.896 s / 1.09965` | 3704 token，语义通过 |
| HMX audio full | `6.972 s` | `20.197 s` | `293.034 s` | `323.778 s / 1.07926` | 3735 token，语义失败 |

完整运行虽然把 encoder 缩短约 `25.4 s`，但 decoder 增加约 `19.9 s`，净收益只剩 `6.118 s`。逐段审核中仅
`78/120` 段 speaker/text 完全一致，出现“送充电宝”变成“送赠品”、关键销售讨论被其他内容替换等实质错误。
释放 audio module、进一步释放 processor runtime、以及 `power=high` 都没有改善 256-token decode；front-only
和 back-only 量化也没有形成可闭合目标的候选。因此不继续靠扩大 calibration 集合重试。Artifacts 和复现说明见
`hmx-audio-poc/README.md`。

### HTP lm_head ArgMax 止损

新的 FP16 HTP context 在 `lm_head` 后直接执行 `ArgMax + Cast(INT32)`，避免把完整 logits 拉回 CPU 再做
greedy sample。32 个 teacher hidden 与 production `graph1_30` 的 top-1 为 `32/32` 一致。runtime 集成中还
定位并修复了 prefill 阶段提前发布 token、导致首个 decode 消费 stale token 的问题；最终 32/256-token 输出
均与 baseline 精确一致。

256-token A/B 中，baseline decode/total 为 `16.081/69.438 s`，ArgMax 为 `16.335/69.563 s`。由阶段计时反推，
sampler/其他时间约减少 `248 ms`，但 decode 增加约 `254 ms`；HTP reduction、额外 decode-only context 和首次
lazy load 抵消了 readback 收益。因此不跑完整 case 2。双跑 verify 模式只用于诊断，曾在结束阶段 exit 139，
不能作为发布配置。32/256-token 是显式截断小门禁，`token_budget_exhausted` 导致的 `status=error/exit=1` 不代表
runtime crash；ArgMax 256 与 baseline 的 raw text SHA256 都是
`1b03f616ae7b0badc919ddfbc7db35998650312827abd94e1c071a8ba5e5c569`。Artifacts 和复现说明见
`hmx-lmhead-poc/README.md`。

## 下一步优先级

1. 保持 async OpenCL release 为当前唯一正确基线：case 2 为 3704 tokens、120 段、total `329.896 s`、
   RTF `1.09965`、peak PSS `1.041 GB`。HMX audio 虽有 `RTF 1.07926`，但语义审核失败，不能晋级。
2. 下一项只有结构性方案仍可信：把多层 projection、Attention、KV update 和 MLP 放进同一 backend/shard，实际
   消除逐层 QNN/OpenCL materialization 与 blocking readback。4-layer 小门禁必须同时保持 v23 数值 operating
   point，并实测低于约 `2.3 ms/layer/token`；否则不扩 28 层。
3. 只有获得可验证的 Qualcomm Genie/GAIT/QPM HMX LLM export 资产，才重开 Qualcomm-native 全 decoder。
   当前通用 ExecuTorch FP16 static decoder 已有反向性能证据，不能仅凭架构更“native”继续投入。
4. HMX audio 只在新量化方案同时通过完整语义门禁、且另有 decoder/shard 证据能把 total 压到 `<=300 s` 时重开；
   单纯增加 calibration 数据无法解释或闭合现有性能缺口。
5. 不再重试 FlashDecode、HMX lm_head、HTP ArgMax、shared-buffer、packed QKV、C128、affinity、record queue、
   现有 GQA kernel、CPU Attention 微调、KV 扩容步长或 parser/repetition 补救。这些方向已有真机反证，或理论
   收益上限不足以达到 RTF 1.0。
