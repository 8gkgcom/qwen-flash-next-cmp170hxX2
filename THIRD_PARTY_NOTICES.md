# 来源与许可

2026-10-06 驱动更新的完整源码保留 NVIDIA 的 `COPYING` 与各文件许可，cmpunlocker 的 GPLv2 许可保留在 [CMPUNLOCKER_LICENSE](drivers/610.57.04-sm74-p2p/CMPUNLOCKER_LICENSE)。SM 解锁增量来源、P2P 基础及兼容修改见 [PATCH_SOURCES.json](drivers/610.57.04-sm74-p2p/PATCH_SOURCES.json)；并非包含全部 v0.5 功能。

2026-10-05 增量更新使用固定上游提交 `11421471c050cccc036b004e91345583e33993b2`。继承的 M7 源码及 M3 适配保留 [Apache-2.0 许可](updates/20261005-pp-mtp2/UPSTREAM_LICENSE) 与 [NOTICE](updates/20261005-pp-mtp2/UPSTREAM_NOTICE)；文件来源见 [SOURCES.json](updates/20261005-pp-mtp2/SOURCES.json)。OrcaRouter 权重不随更新发布。

| 组件 | 来源 / 许可位置 |
|---|---|
| SM80 推理分支 | [Qwen-Flash-SM80-170HX](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX)，`licenses/sm80/LICENSE`、`NOTICE`，Apache-2.0 |
| PLE 社区集成 | `licenses/sm80/community/qwen3.8-Flash-DGX-LICENSE` |
| vLLM | [vLLM](https://github.com/vllm-project/vllm)，`licenses/vllm/licenses/LICENSE` 及源文件声明 |
| NVIDIA 内核模块 | [610.57.04](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.57.04)，`drivers/nvidia-open-610.57.04-patched/COPYING`；各文件通常 MIT，组合内核模块 MIT/GPLv2 |
| cmpunlocker | 用户提供的源码包，`drivers/cmpunlocker/LICENSE`，GPLv2 及文件头 |

第三方源码与本地修改保留各自声明，不统一重新授权。新增部署辅助代码未单独指定整个仓库的统一许可证。

模型权重从 [dealignai](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4) 获取并遵循其许可。闭源用户态驱动、GSP 固件、CUDA 安装包和模型权重未捆绑；上游 preload 计算内核保留原声明。
