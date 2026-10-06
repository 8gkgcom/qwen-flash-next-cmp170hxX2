# 双 CMP 170HX：Qwen3.8 Flash Next 部署与优化

## 2026-10-06：当前 64GB / P2P / 74 SM 驱动源码

[源码、编译、安装与回退](drivers/610.57.04-sm74-p2p/README.md) · [完整源码附件](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.06-driver-sm74)

NVIDIA **610.57.04**，验证内核 **6.8.0-138-generic**。保留原有 64GB/BAR1/P2P 与 Linux 6.8 兼容修改，仅加入 SM 解锁增量；两卡实测 **74 SM**。3871 个源码文件与当前构建目录一致，5 个安装模块与构建产物哈希一致。未启用 v0.5 ECC 改动或 Gen3/Gen4 实验。附件含完整已打补丁源码、原版到当前版的补丁及校验清单，不含编译模块和本机配置。

## 2026-10-05：Orca NVFP4 单路优化

**[本轮源码、安装与回退、实测结果](updates/20261005-pp-mtp2/README.md)** · [当前推荐参数](updates/20261005-pp-mtp2/recommended.json)

基于作者 v0.2.0 和已有 Orca PP2/RAM 适配，保留固定形状 draft scatter、三行 HC 融合，修复长输入内核加载及上下文末端停滞。当前组合为 **PP2 24/24、MTP2、262144 上下文、4 槽位、8192 预填充、95% 显存、RAM PLE、xhigh、温度 1.0**。本轮 GPU0 为 PCIe Gen2 ×16、GPU1 为 Gen2 ×4。

同条件平均每轮耗时 **19.066 → 18.323 ms，约下降 3.9%**；代码样本稍快、最终推理样本稍慢，不代表所有任务都提速。两路完整 262K、四路短请求及接口检查通过；4 槽位不等于 4 路完整 262K。

这是对已有兼容环境的**增量补丁**，安装器校验基础源码版本；不能直接覆盖下方旧版本部署包。包含去敏后的源码与结果，不包含控制台、凭据、模型权重或本机编译缓存。

## 2026-10-01 历史方案：dealignai NVFP4 / v0.1.7

以下部署、硬件状态、参数和实测均属于 10 月 1 日的历史方案，与上方 Orca 更新分开使用。

**[下载完整部署源码包（含 64GB/P2P 驱动源码）](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.01)** · 约 36 MB。请下载发布页的 `qwen-flash-cmp170hx-share-20261001.zip`。仓库网页提供文档、示例和测试数据；部署时使用完整附件解压目录，GitHub 自动生成的 Source code 包不包含完整驱动及推理覆盖层。

基于 [Qwen-Flash-SM80-170HX v0.1.7](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX)，原生运行 [dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)。本仓库提供推理补丁、独立启动脚本、64GB/P2P 驱动源码和测试数据。

**推荐组合：PP2 · 24/24 分层 · MTP4 · RAM PLE · CUDA Graph。** 实测日期：2026-10-01。

[部署步骤](docs/DEPLOYMENT.md) · [优化记录](docs/OPTIMIZATION.md) · [驱动编译](docs/DRIVER.md) · [许可证](THIRD_PARTY_NOTICES.md)

## 硬件

| 项目 | 配置 |
|---|---|
| CPU / RAM | i3-14100，4 核 8 线程；96GB DDR4，2666 MT/s |
| 主板 | ASUS ROG STRIX Z790-A GAMING WIFI D4，BIOS 3202 |
| GPU | 2 × CMP 170HX 64GiB；各 PCIe Gen2 ×4、200W |
| GPU1 接线 | **M.2 NVMe 接口 → 转接显卡坞 → CMP 170HX** |
| 系统 / 内核 | HiveOS / Ubuntu 22.04.5；`6.8.0-138-generic` |
| 驱动 | `610.57.04`，64GB/BAR1/P2P 修改版 |

两卡均接 CPU PCIe 根端口，拓扑为 PHB，无 NVLink：

```text
CPU
├── 00:01.0 → 01:00.0 → GPU0
└── 00:06.0 → 03:00.0 → M.2 转接显卡坞 → GPU1
```

主模型及 MoE experts 放入双卡 HBM；约 **47.68 GiB 原始 FP8 PLE** 预热并锁定在 RAM，活跃 KV 使用 BF16。

## 推荐参数

| 参数 | 值 |
|---|---|
| 并行 / 分层 | TP=1、PP=2；24/24 |
| GPU 顺序 | 物理 GPU1 → GPU0；后半层及 MTP 位于 GPU0 |
| 主模型 / 后端 | `modelopt_fp4` / `marlin` |
| 计算 dtype / KV | BF16 / BF16 |
| 目标上下文 / 引擎窗口 | 262144 总 tokens / 262400，留 256 边界余量 |
| 显存预算 / 调度槽位 | 95% / 8 |
| 预填充总预算 / 长请求分块阈值 | 8192 / 1280 |
| block size / Mamba cache | 1632 / `align` |
| MTP | 4；INT8 草稿候选筛选 + BF16 top-32 复核 |
| CUDA Graph | `FULL_AND_PIECEWISE` |
| 前缀缓存 / 分块预填充 / 异步调度 | 开启 |
| PLE | RAM mmap + prewarm + mlock，同进程字节 gather |
| NCCL P2P | 开启；自定义 all-reduce 关闭 |
| GDS / GPU PLE cache / CPU KV offload | 关闭 |
| CPU governor | `performance` |
| 默认思考 / 采样 | `xhigh`；temperature=1、top_p=0.95、top_k=20 |
| API | `qwen`，端口 `8000`；文本输入 |

完整命令见 [启动脚本](examples/start-nvfp4.example.sh)。进程内 CUDA:0 是物理 GPU1，CUDA:1 是物理 GPU0。

8 槽位共享 KV 池，**不代表 8 路完整 262K**。上下文可调整，当前验收到 262144；95% 为引擎预算，其他 CUDA 开销仍会占显存。

## 实测结果

普通代码关闭思考，agent 使用 xhigh；temperature=1、seed=0。单请求各生成 2048 tokens，每类两次；双请求各生成 1024 tokens。

| 设置 | 平均轮次 ms | 代码 tok/s | 思考 tok/s | 双请求合计 tok/s |
|---|---:|---:|---:|---:|
| 跨进程 PLE 基线 | 28.112 | 135.86 | 73.51 | 118.35 |
| 同进程 PLE | 26.193 | 123.94 | 82.78 | 125.36 |
| **同进程 PLE + CPU performance** | **25.486** | **136.76** | **83.27** | **118.32** |

平均轮次耗时下降 **9.34%**，实际吞吐还受 MTP 接受率和输出内容影响。完整对照见 [优化记录](docs/OPTIMIZATION.md) 与 [测试数据](benchmarks/ple-cpu/results.json)。

| 验证 | 结果 |
|---|---|
| 两路完整窗口 | 每路 262077 输入 + 67 输出 = **262144**；首部标记回取正确，零抢占 |
| 8 路短请求 | 每路 1005 输入 + 275 输出；同时运行 8 路，零抢占 |
| PLE 行号一致性 | 108 组模拟 + 108 组真实 checkpoint 对照通过 |

长窗口测试验证容量与标记回取，不代表完整能力评测。MTP 候选筛选的精度边界见 [优化记录](docs/OPTIMIZATION.md)。

## 目录

```text
runtime/overlay/  固定 vLLM 基础上的增量源码及 preload
scripts/         安装与复测脚本
examples/        独立启动命令、环境变量、systemd 服务
drivers/         cmpunlocker 补丁及完整已打补丁 NVIDIA 源码
benchmarks/      性能与并发测试数据
provenance/      模型版本、源码差异和驱动身份
```

按 [部署文档](docs/DEPLOYMENT.md) 准备固定环境、下载模型、安装覆盖层并启动。运行时无需 Docker；模型权重和基础编译扩展另行准备。文件来源和校验见 [源码说明](docs/SOURCE_LAYOUT.md)。
