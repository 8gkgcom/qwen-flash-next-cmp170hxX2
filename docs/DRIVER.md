# CMP 170HX 驱动

当前版本见 [64GB / P2P / 74 SM 源码与安装说明](../drivers/610.57.04-sm74-p2p/README.md)，[完整源码附件](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.06-driver-sm74)。

以下保留 **2026-10-01 的 70 SM 历史版本**说明；不要用其构建安装器覆盖新的 74 SM 版本。

## 2026-10-01 历史版本

先下载并解压 [完整源码附件](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.01)，以下路径和命令以完整包为准。

`drivers/cmpunlocker/` 为用户提供的安装器及补丁；`drivers/nvidia-open-610.57.04-patched/` 为当前驱动对应的完整已打补丁源码。

## 版本

| 项目 | 值 |
|---|---|
| NVIDIA / 内核 | 610.57.04 / 6.8.0-138-generic |
| 运行模块 srcversion | `A7AA83788B88DA9780FC086` |
| profile / P2P | `8gb` / `p2p=1` |
| 本机结果 | 两卡各 65536 MiB，BAR1 64GiB |

已安装 `nvidia.ko` 与该源码目录构建产物的 SHA256 相同，见 [BUILD_IDENTITY.json](../drivers/BUILD_IDENTITY.json)。`8gb` 是安装器的原始板卡分类，本机对应 64GB 解锁配置。

## 编译

需要匹配的 kernel headers、编译器，以及 **610.57.04 用户态驱动/GSP 固件**。基础源码为 [NVIDIA 610.57.04](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.57.04)。

```bash
uname -r
test -d "/lib/modules/$(uname -r)/build"
modinfo -F version nvidia

# 已打补丁的源码：只构建，不安装，也不要重复打补丁。
cd drivers/nvidia-open-610.57.04-patched
make -j4 modules SYSSRC="/lib/modules/$(uname -r)/build"
```

构建产物只适用于匹配的内核。Secure Boot 开启时还需完成模块签名/信任配置。

## 安装

安装器会替换模块、更新 initramfs 并处理 NVIDIA DKMS；在停止 GPU 任务并保留旧驱动/内核回退入口后执行。

确认用户态驱动为 **610.57.04**，再从仓库根目录运行：

```bash
cd drivers/cmpunlocker
sudo bash install.sh --profile=8gb --p2p --no-iommu --no-gen2-service --no-passthrough
```

注意：

- `install.sh` 使用检测到的已安装 NVIDIA 版本，不是简单读取外部版本变量。
- `driver/VERSION` 的第一项不是本机实测版本。直接调用构建安装脚本需锁定版本：

```bash
sudo env CMPUNLOCKER_DRIVER_VERSION=610.57.04 \
  CMPUNLOCKER_CARD_PROFILE=8gb CMPUNLOCKER_ENABLE_P2P=1 \
  bash driver/build.sh
```

`--no-iommu` 表示不修改启动参数。本机已使用 `intel_iommu=off iommu=off`；安装器默认 IOMMU 配置不同。关闭 IOMMU 会改变设备 DMA 隔离能力，需要虚拟机直通时应另行评估。

两卡已经为 Gen2 x4，因此示例不配置 Gen2 retrain 服务。P2P 构建后按脚本要求冷重启。

## 验证

1. 核对驱动版本、两卡显存、BAR1、PCIe 链路和 Above 4G 资源分配。
2. 使用非零/随机数据做双向传输与结果校验，不能只看 capability 或带宽。
3. 当前模型使用已通过数据检查的 NCCL P2P/CUMEM 路径；raw peer/custom all-reduce 历史上出现过错误数据，故保留 `--disable-custom-all-reduce`。
4. 再做模型短请求、两路完整窗口和持续负载测试。

原测试源码见 `drivers/cmpunlocker/tests/` 及其 `docs/P2P.md`。GPU↔GPU P2P 可用不代表 NVMe→GPU GDS 可用。

内核更新后需重新核对 headers、vermagic、签名和匹配组件，重建后完成以上验证再移除旧启动项。
