# 来源与许可

| 组件 | 来源 / 许可位置 |
|---|---|
| SM80 推理分支 | [Qwen-Flash-SM80-170HX](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX)，`licenses/sm80/LICENSE`、`NOTICE`，Apache-2.0 |
| PLE 社区集成 | `licenses/sm80/community/qwen3.8-Flash-DGX-LICENSE` |
| vLLM | [vLLM](https://github.com/vllm-project/vllm)，`licenses/vllm/licenses/LICENSE` 及源文件声明 |
| NVIDIA 内核模块 | [610.57.04](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.57.04)，`drivers/nvidia-open-610.57.04-patched/COPYING`；各文件通常 MIT，组合内核模块 MIT/GPLv2 |
| cmpunlocker | 用户提供的源码包，`drivers/cmpunlocker/LICENSE`，GPLv2 及文件头 |

第三方源码与本地修改保留各自声明，不统一重新授权。新增部署辅助代码未单独指定整个仓库的统一许可证。

模型权重从 [dealignai](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4) 获取并遵循其许可。闭源用户态驱动、GSP 固件、CUDA 安装包和模型权重未捆绑；上游 preload 计算内核保留原声明。
