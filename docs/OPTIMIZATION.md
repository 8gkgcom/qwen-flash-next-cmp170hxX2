# 优化记录

目标是保留主模型精度，支持两路完整 262144 总窗口，再降低 decode 开销。

## 当前保留的优化

| 优化 | 实现 |
|---|---|
| RAM PLE | 原始 FP8 文件 mmap、预热、mlock；约 47.68 GiB 常驻 RAM |
| 字节 gather | `ple_ram_gather.cpp` 直接复制原始行字节，保留 pinned H2D |
| 同进程 PLE | `sm80_ple_local.py` 减少独立 CPU 进程与 ZMQ 交接 |
| PP2 / 24+24 | 减少宽度并行通信，MTP 所在末段放物理 GPU0 |
| CUDA Graph | `FULL_AND_PIECEWISE`，保留特殊算子分割项 |
| PP MTP4 | `sm80_pp_draft_int8.py` 使用 INT8 候选筛选、BF16 top-32 重排 |
| CPU performance | 减少 CPU 准备阶段的调频延迟，未固定绑核 |

PLE 路径：行号规划 → RAM 原始 FP8 gather → pinned buffer → 异步 H2D → 原模型 PLE 算子。保留必要的事件/信号量同步；行号对照通过 108 组模拟和 108 组真实 checkpoint 检查。

## 同条件对照

普通代码和 xhigh 思考场景各两次，单请求输出 2048 tokens。temperature=1、seed=0；完整数据见 [results.json](../benchmarks/ple-cpu/results.json)。

| 设置 | 平均轮次 ms | 结果 |
|---|---:|---|
| 跨进程 PLE 基线 | 28.112 | 对照 |
| PP 小矩阵内核 | 28.508 | 无整体收益，关闭 |
| 同进程 PLE | 26.193 | 保留 |
| 同进程 + GPU1 4GiB PLE cache | 26.890 | 比同进程本身慢约 2.66%，关闭 |
| **同进程 + CPU performance** | **25.486** | **采用；相对基线下降 9.34%** |
| 再加固定绑核 | 25.434 | 单请求改善约 0.20%，未采用 |
| 同进程恢复 ondemand | 26.268 | 慢于 performance |

轮次耗时按 `(elapsed − TTFT) / speculative draft count` 计算，仅用于单请求开销对照。SSE 速率按 `completion_tokens / (末个有效输出事件 − 首个有效输出事件)`，包含思考 token。

样本较少，输出和 MTP 接受率会变化；轮次变快不等于所有任务 tok/s 等比例增加。

## 未采用的方案

- **GPU PLE 缓存**：4GiB 缓存命中率约 40.86%，命中检查、缺失处理和同步成本超过查询节省。相关模块作为 PLE 可选依赖保留，默认关闭。
- **PP 小矩阵内核**：部分 M=1 微基准更快，但 PP 与作者 TP/TEP 矩阵形状不同，整模型没有收益。
- **固定绑核**：单请求改善不足 0.3%，恢复原亲和性。
- **GDS**：严格读盘仍失败。当前 ext4 路径中 `dma_direct_map_sg` 返回 `-121`（EREMOTEIO），不能只归因于磁盘格式。RAM 盘也不能修复 PCI P2PDMA；已有 mmap+预热+锁页在从 RAM 读取。

## MTP 与精度

当前采用 MTP4。历史温度 1 对照：

| 草稿路径 | 代码 tok/s | 思考 tok/s | 双请求合计 tok/s |
|---|---:|---:|---:|
| BF16 / MTP6 | 123.01 | 63.29 | 92.31 |
| INT8 筛选 / MTP4 | 133.09 | 73.17 | 118.66 |

见 [历史 MTP 数据](../benchmarks/mtp-int8-results.json)。该阶段与最新 PLE 测试条件不同，不叠加收益。

主模型权重及验证 logits 保留原精度；INT8 只用于草稿 lm_head 的候选筛选，随后 BF16 重排。256 组随机隐状态检查通过，但不保证所有隐状态都命中全词表 BF16 的最优候选，不能宣称所有输出逐字一致。

## 分层与容量

物理 GPU1 处理前 24 层，GPU0 处理后 24 层及 MTP。平均层数不保证平均显存或功耗，末段还承担 MTP/lm_head 等固定开销。26/22 历史方案未成为当前速度最优配置。

8 槽位共享 GPU KV 池，已验证两路完整 262144 和八路短请求；未保证八路完整窗口。当前 QSA/Mamba 路径未启用 CPU KV offload。改变分层、MTP、上下文或显存预算后需重新测容量。

## 继续优化的测量项

用相同任务同时记录 TTFT、decode tok/s、每轮接受 token 数、并发总吞吐、抢占、两卡频率/功耗/温度和 CPU governor。优先对实际 agent 任务扩大样本，再评估内核或缓存修改。

作者公开结果使用不同的 GDS/TEP2/MTP6 条件，不能直接与本机 PP2/RAM 结果相除比较。参考 [上游项目](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX)；GDS 条件见 [NVIDIA 文档](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)。
