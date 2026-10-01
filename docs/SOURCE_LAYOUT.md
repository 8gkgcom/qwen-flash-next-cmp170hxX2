# 源码与校验

先下载并解压 [完整源码附件](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.01)，以下路径和命令以完整包为准。

## 推理覆盖层

`runtime/overlay/` 包含上游导入依赖、全部已修改的 vLLM 源文件、本地 PP/RAM 模块及 preload；其余基础代码由固定镜像提供。

对照 wheel RECORD 核验了 4758 个文件，发现 12 个修改文件及 1 个新增 vLLM 模块，全部保留。见 [源码差异](../provenance/vllm-source-diff.json) 和 [基础版本](../runtime/SOURCES.json)。安装器使用 `runtime/base-sha256.json` 防止覆盖不匹配的基础环境。

| 文件 | 作用 |
|---|---|
| `vllm_ple_mmap.py` / `ple_ram_gather.cpp` | 原始 FP8 mmap、锁页及字节 gather |
| `sm80_ram_hooks.py` | RAM/PP 接入 |
| `sm80_ple_local.py` | 同进程 PLE |
| `sm80_pp_draft_int8.py` | 草稿候选筛选与 BF16 重排 |
| `sm80_ple_cache.py` | PLE 可选导入依赖，启动脚本关闭缓存 |

GDS/TP 模块属于上游导入依赖，启动脚本仅启用 PP2/RAM 路径。

## 其他内容

- `drivers/` 保留完整已打补丁驱动源码，身份见 `BUILD_IDENTITY.json`。
- `benchmarks/` 保留已完成实验的数据，独立启动示例尚未在新主机从零验收。
- `SHA256SUMS` 校验最终发布文件；`provenance/RELEASE_CHECKS.json` 记录发布检查。
- 模型权重、基础编译扩展、认证文件和闭源用户态驱动另行准备。

第三方声明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
