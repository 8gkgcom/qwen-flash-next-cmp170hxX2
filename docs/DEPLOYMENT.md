# 部署

先下载并解压 [完整源码附件](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.01)，以下路径和命令以完整包为准。

目标：NVFP4、双卡 PP2、RAM PLE。当前主机已验证；新主机的 Python/CUDA ABI 和完整窗口需重新验收。

## 1. 驱动与固定环境

先按 [驱动文档](DRIVER.md) 确认两张卡各有 65536 MiB 显存，并核验实际通信：

```bash
uname -r
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id,uuid --format=csv
nvidia-smi topo -m
lspci -tv
```

实测软件组合：

| 组件 | 版本 |
|---|---|
| Python / vLLM | 3.12.14 / `0.1.dev20073+g8e685d198` |
| PyTorch / Triton | 2.13.0 / 3.7.1 |
| Transformers / TileLang | 5.17.0 / 0.1.14 |
| flashinfer-python / cuda-python | 0.6.16.post3 / 13.4.1 |
| CUDA Toolkit | 13.0，nvcc 13.0.88 |
| NumPy / safetensors | 2.3.5 / 0.8.0 |

固定基础镜像：

```text
vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
```

已有匹配原生环境可直接复用。新主机可用 `skopeo`、`umoci` 获取基础二进制，无需启动 Docker：

```bash
skopeo copy --override-os linux --override-arch amd64 \
  docker://vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8 \
  oci:./vllm-base-oci:pinned
umoci unpack --image ./vllm-base-oci:pinned ./vllm-base-unpacked
```

从 `vllm-base-unpacked/rootfs/usr/local/lib/python3.12/dist-packages` 准备基础包。宿主机 Python、torch、CUDA 动态库必须与其中的编译扩展匹配；仅复制 vllm 目录不够。不要覆盖宿主机 `/lib`，运行时不要加载 CUDA stubs。

下文示例路径：

```bash
export SITE=/media/nvme2/qwen-flash-sm80-data/native-root/usr/local/lib/python3.12/dist-packages
export ENGINE_PYTHON=/opt/qwen-flash/venv/bin/python
```

`ENGINE_PYTHON` 指向自己准备的 Python 3.12 环境。新增内核依赖及哈希见 `examples/requirements-kernels.txt`；不要用 `pip install -U vllm` 替换固定分支。先检查：

```bash
PYTHONPATH="$SITE" "$ENGINE_PYTHON" -c \
  'import torch,triton,vllm; print(torch.__version__,triton.__version__,vllm.__version__); print(torch.cuda.device_count())'
find "$SITE/vllm" -maxdepth 1 -name '_C*.so' -exec ldd '{}' \;
```

## 2. 下载模型

模型约 125.91 GiB，另留基础环境和缓存空间。模型目录应在已挂载的数据盘上。

```bash
hf download dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --revision be794b990578ef3031eccf9f28e675a289a09ee9 \
  --local-dir /media/nvme/models/dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4
```

核对 206 个分片及 [元数据哈希](../provenance/model-source-dealignai.json)。RAM 路径直接读取原始 PLE，不需要额外的 GDS 转换文件。

## 3. 安装推理补丁

将本仓库放到 `/opt/qwen-flash`；已有目录时只复制仓库文件，不覆盖自己准备的 venv。以下命令从仓库根目录执行：

```bash
# 只检查，不写入。
python3 scripts/install_runtime_overlay.py --target "$SITE"

# 停止使用该目标环境的模型后应用。
sudo python3 scripts/install_runtime_overlay.py --target "$SITE" --apply
```

脚本核对基础源码哈希，备份覆盖文件，安装增量源码/preload，并用 g++ 编译 `ple_ram_gather.so`。其余 vLLM 代码及 `.so` 继续使用固定基础环境。

PLE 约 47.68 GiB，会预热并锁页。运行账户需要足够的 `MEMLOCK`，以及数据/缓存目录写权限。上游 preload 记录写入 `/evidence`，也需要可写。

## 4. 独立启动

按 `nvidia-smi` 的物理编号填写 UUID；`GPU1,GPU0` 顺序让后半层及 MTP 位于 GPU0。

```bash
export GPU0_UUID='GPU-替换为物理GPU0的完整UUID'
export GPU1_UUID='GPU-替换为物理GPU1的完整UUID'
export MODEL_API_KEY='自己设置的密钥'
export SM80_PYTHON="$ENGINE_PYTHON"
export SM80_SITE="$SITE"

bash examples/start-nvfp4.example.sh
```

脚本的环境变量可覆盖模型路径、缓存路径、端口、上下文、槽位及 MTP 数，默认值见 `examples/qwen-flash.env.example`。shell 直接启动前应确认 `ulimit -l` 足够；systemd 示例已设置 `LimitMEMLOCK=infinity`。

手动启动时先创建可写的 `/evidence` 和缓存目录。仅查看隐藏密钥后的命令可执行 `SM80_PRINT_COMMAND=1 bash examples/start-nvfp4.example.sh`，不会加载模型。

脚本保持主模型 NVFP4、原始 FP8 PLE 和 BF16 KV，关闭 GDS、GPU PLE cache、CPU KV offload 和自定义 all-reduce。

## 5. systemd（可选）

示例服务账户为 `model`，仓库位于 `/opt/qwen-flash`：

```bash
getent passwd model || sudo useradd --create-home --shell /bin/bash model
sudo install -d -o model -g model /evidence /media/nvme2/qwen-flash-sm80-data/logs \
  /media/nvme2/qwen-flash-sm80-data/cache/vllm-ram/pp2-optimized \
  /media/nvme2/qwen-flash-sm80-data/cache/triton
sudo install -m 0600 examples/qwen-flash.env.example /etc/qwen-flash.env
sudoedit /etc/qwen-flash.env
sudo install -m 0644 examples/qwen-flash.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start qwen-flash.service
journalctl -u qwen-flash.service -f
```

账户需可读模型/基础环境。确认启动成功后，可执行 `sudo systemctl enable qwen-flash.service`。修改配置后通过该服务重启；不要同时手工启动第二份占用相同 GPU/端口的进程。

## 6. CPU 性能模式

```bash
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
for governor in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
  printf '%s\n' performance | sudo tee "$governor" >/dev/null
done
```

本机用 `cpufrequtils` 的 `GOVERNOR="performance"` 持久化，保留原 CPU 亲和性。

## 7. 复测

```bash
export MODEL_API_KEY='与启动时相同的密钥'
curl -H "Authorization: Bearer $MODEL_API_KEY" http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $MODEL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"计算 17+25，只输出结果。"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'

python3 scripts/stable_benchmark.py local-check --temperature 1 --repeats 2
python3 scripts/long_exact_probe.py \
  --count 2 --prompt-tokens 262077 --max-tokens 67 --max-context 262144 \
  --require-overlap --output two-262k.local.json
```

在无其他请求时复测。长窗口脚本按 tokenizer 实际计数补足总窗口，检查两路标记回取、实际并发、零抢占与 usage；不同模板以结果计数为准。

| 问题 | 检查 |
|---|---|
| import / undefined symbol | 固定基础包、Python/torch/CUDA ABI、动态库路径 |
| PLE 持续读盘或锁页失败 | 预热、可用 RAM、MEMLOCK |
| OOM / 抢占 | 实际显存、KV 总池、prefill 缓冲、其他进程 |
| 输出达到上限 | 客户端 max_tokens 与总窗口预算 |
| P2P 错误数据 | NCCL 实际校验、自定义 all-reduce 是否关闭 |

源码备份不是完整环境备份。升级或回退时一起匹配基础镜像 digest、源码、编译扩展和依赖版本。
