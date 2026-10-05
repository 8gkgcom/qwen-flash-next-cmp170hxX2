# Orca NVFP4 · PP2 单路优化 · 2026-10-05

本轮保留固定形状 draft scatter、三行 HC 融合，并修复长输入内核加载与上下文末端停滞。**最终单轮耗时下降约 3.9%，没有实现单路 220 tok/s，也没有证明所有任务都提速。**

## 适用范围

- 模型是 OrcaRouter BF16 母版在本地导出的 ModelOpt NVFP4 / Marlin W4A16；本地导出目录名不是 Hugging Face 仓库名，权重不在本仓库。
- 运行基础为作者 `v0.2.0`、提交 `11421471c050cccc036b004e91345583e33993b2`，加已有的 Orca PP2/RAM 兼容适配；vLLM 为 `0.1.dev20073+g8e685d198`。
- 本目录是**增量补丁和复测工具**，需要已经能运行该模型的 PP2/RAM 环境。不是从零安装包，也不能直接覆盖本仓库 10 月 1 日的 v0.1.7 / dealignai 发布包。安装器会拒绝不匹配的基础文件。
- 两张 CMP 170HX 64 GiB，i3-14100、96 GB RAM。此次 GPU0 为 Gen2 ×16，GPU1 为 Gen2 ×4；GPU1 通过 CPU M.2 接口转接。当前 GPU 顺序为物理 GPU1 → GPU0。

## 当前验证并保留的参数

| 参数 | 值 |
|---|---|
| 并行与分层 | TP1 / PP2；24＋24 层，MTP 在物理 GPU0 |
| 上下文 | 262144 输入＋输出总 tokens；可手动调整 |
| 调度槽位 / 显存预算 | 4 / 95% |
| 预填充预算 / 长输入分块 | 8192 / 1280 |
| MTP | 2；既有 INT8 草稿筛选与 BF16 候选复核 |
| 计算 / 活跃 KV | BF16 / BF16（启动参数 `kv_cache_dtype=auto`） |
| PLE | FP8 表常驻 RAM，预热锁页、同进程原生查表、pinned 异步 H2D |
| 调度与图 | 异步；FULL_AND_PIECEWISE；前缀缓存、分块预填充 |
| 采样 | xhigh；temperature 1.0、top_p 0.95、top_k 20、presence_penalty 0.5 |
| 接口 | 模型名 qwen、端口 8000；凭据由使用者自行设置 |

完整机器无关参数见 [recommended.json](recommended.json)。4 槽位共享 KV 池，不代表 4 路完整 262K。GDS、GPU PLE 热行缓存、CPU KV offload、历史草稿及 TEP2/IPC 未在本轮最终方案启用。

## 改了什么

1. `pp_draft_scatter.py`：固定形状、带掩码的 Triton scatter，替代 GPU 布尔压缩索引，减少一次主机同步。该项来自当天前一轮，是后续测速基线的一部分。
2. `m3_hc_runtime.py` / `m3_hc_kernels.py`：按 MTP2 的三行形状改写 HC down＋SiLU、up＋gate。继承作者七行实现的计算结构，FP32 累加并保留 BF16 阶段边界；非三行调用完整原始编译函数。不会把任意 GEMM 都替换成该内核。
3. QSA 预加载：把本机已编译、原清单遗漏的 31 个 SM80 QSA 内核加入启动预加载，避免在 PLE 流等待期间首次加载 CUDA 模块而卡住。不同机器需依据自己的缓存重新生成清单。
4. `03-terminal.patch`：异步 MTP 到上下文末端时，只在输出占位和在途计算都清空后缩短草稿，避免剩余空间容纳不下完整验证批次而永久等待；不修改输出预算和停止条件。

未保留 PP 元数据缓存（约 0.1% 波动）、GDN 投影融合（微测更慢）及部分 TEP2 试验。它们没有被描述成已生效优化。

## 实测与限制

同一代码、推理提示，xhigh、温度 1、seed 123；单路每次最多输出 1536 tokens，每类两次，包含思考 tokens。测速长度是固定截断，另做完整答案检查。

| 第二轮对照 | 平均每轮 ms | 代码 tok/s | 推理 tok/s |
|---|---:|---:|---:|
| 基线：已包含 scatter 修复 | 19.066 | 87.914 | 113.810 |
| 最终保留版本 | 18.323 | 90.553 | 110.879 |

“每轮耗时”是流式 decode 时间除以引擎 draft 轮数，**不是 GPU 纯 kernel 时间**。最终推理样本的 tok/s 反而略低；输出变化与草稿接受率会抵消单步耗时收益。保留了之前一次复测及无收益试验的摘要，没有只挑最快结果。完整汇总见 [results.json](results.json)。旧模型、旧 BIOS、不同提示长度的结果不能直接拼成同条件加速比。

| 验证 | 结果 |
|---|---|
| 两路完整 262K（视觉请求之后） | 每路 262128 输入＋16 输出＝262144；约 80.3 秒；正常 length 结束 |
| 冷启动长输入 | 两路 262128 输入、16 输出预算，均自然停止；用于检查首次内核加载 |
| 四路持续短请求 | 每路 512 输出；两组合计约 253 / 302 tok/s，零抢占；这是总吞吐 |
| 数值 | 合成形状、15 次变化输入 CUDA Graph 重放；18 组真实 HC 权重检查通过 |
| 非三行回退 | 27 个编译函数的原始函数体 AST 一致 |
| 调度边界 | 42 项离线门控检查；覆盖已排空与在途状态 |
| 完整答案 | 一道数学题正确；一道代码题通过 9331 个输入用例，不等于 9331 道题 |
| 接口与视觉 | Chat Completions、Responses、Messages、图片识别检查通过 |

真实权重 HC 链的最大相对 L2 差约 0.000384；浮点归约顺序有差异，不宣称逐 bit 一致。长上下文测试使用重复填充和简单回取指令，验证容量与结束行为，不代表全面长文本理解能力。上述结果也不是模型能力排行榜。

## 集成与回退

先停止模型，指定**现有私有运行环境**，用对应的 Python 3.12 环境执行。`SITE` 和 `PATCH` 需要自行填入真实路径。

```bash
SITE=/path/to/existing-runtime/site
PATCH=/path/to/this-directory
python "$PATCH/apply_changes.py" --target "$SITE"
python "$PATCH/apply_changes.py" --target "$SITE" --apply
```

首条仅校验并预览；第二条验证基础文件 SHA256、备份后安装。需系统已安装 Git。基础不匹配时停止处理，不能绕过校验硬套到其他 vLLM。`m7_joint_*` 是保留的上游依赖，M3 适配使用其结构匹配器；并非启用了 MTP7。

在原先可工作的启动器中加入 [optimization.env.example](optimization.env.example)，保留既有模型、RAM PLE、视觉和 API 配置。使用新编译缓存目录，或确认旧缓存未混入其他实验补丁；M3 会为改写的编译文件保存 `.before-round2-m3` 副本。

预加载缓存用本机文件生成，不从公开包下载含本机调试路径的 cubin：

```bash
python "$PATCH/collect_qsa_preload.py" \
  --base /path/to/existing-model-vision-preload \
  --cache /path/to/local-triton-cache \
  --output /path/to/new-merged-preload
export Q38_TRITON_PRELOAD_DIR=/path/to/new-merged-preload
```

该工具只合并**已编译**的 CUDA SM80 QSA 内核，验证原清单哈希，不编译缺失内核。全新环境没有本机缓存时，需要先在隔离预热流程生成相应形状，再收集；不能把这份清单视为任意模型/版本的通用预热。保留原基础预加载目录，覆盖到的新形状仍需实测。

安装器备份可回退源码：

```bash
python "$PATCH/apply_changes.py" --target "$SITE" --restore /path/to/backup
python "$PATCH/apply_changes.py" --target "$SITE" --restore /path/to/backup --apply
```

回退前停模，移除新增环境变量并恢复原预加载目录；改用干净编译缓存或恢复 `.before-round2-m3` 文件后再启动。完整回退也会移除两个稳定性修复。脚本不修改 systemd，也不自动重启服务。

## 复测与发布内容

发布工具通过安装预览、精确哈希安装、重复安装、版本拒绝、回退及预加载清单校验，共 12 项离线检查，见 [PUBLISH_CHECKS.json](PUBLISH_CHECKS.json)。这是打包验证，没有在发布过程中重新测速或重启模型；文件完整性清单见 [SHA256SUMS](SHA256SUMS)。

```bash
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
export Q38_PP_M3_HC=1
export MODEL_DIR=/path/to/local-model
python "$PATCH/test_m3_hc.py"
python "$PATCH/test_m3_checkpoint.py"
export BENCH_URL=http://127.0.0.1:8000
# BENCH_API_KEY 从自己的环境提供，不写入脚本或仓库。
python "$PATCH/benchmark.py" verification --repeat 2
python "$PATCH/benchmark.py" four-queue --concurrency 4 --tokens 512
```

源码、参数和统计摘要均经发布前扫描；未上传控制台源码/配置、登录凭据、IP、GPU UUID、用户名路径、私有对话、原始性能 trace、编译缓存或模型权重。原始诊断 ZIP 仅用于本机留档，不是公开附件。

上游来源与固定版本见 [SOURCES.json](SOURCES.json)。继承文件保留 [Apache-2.0 LICENSE](UPSTREAM_LICENSE) 和 [NOTICE](UPSTREAM_NOTICE)，vLLM 补丁保留其原有许可。旧发布包和驱动来源不变。
