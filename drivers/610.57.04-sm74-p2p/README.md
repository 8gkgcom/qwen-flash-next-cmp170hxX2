# CMP 170HX：64GB / P2P / 74 SM 驱动源码

与本机当前安装模块对应的 **NVIDIA 610.57.04** 源码：原有 64GB、BAR1、P2P 修改，加上 cmpunlocker 的 SM 解锁增量。两卡由 70 SM 变为 74 SM。保留已有 Linux 6.8 DRM 兼容修改。

[下载完整源码包](https://github.com/8gkgcom/qwen-flash-next-cmp170hxX2/releases/tag/v2026.10.06-driver-sm74) · [完整补丁](stock-to-sm74-p2p.patch) · [源码及模块身份](BUILD_IDENTITY.json) · [验证记录](VALIDATION.json)

发布附件为 `cmp170hx-610.57.04-sm74-p2p-source.tar.gz`。仓库中的 GitHub 自动生成 Source code 包只含补丁和辅助文件，完整 NVIDIA 源码在此附件中。

## 已核对的范围

- NVIDIA **610.57.04**，内核 **6.8.0-138-generic**，GCC **12.3**。
- 3871 个源码文件逐一匹配当前构建目录；相对 NVIDIA 原版改变 23 个文件。
- 5 个已安装 `.ko` 与该目录现有构建产物 SHA256 一致。没有分发这些内核专用二进制。
- 2026-10-05 重启验收：两卡各 65536 MiB、74 SM；小型 BF16 计算、双向非零数据 NCCL P2P/CUMEM 及 all-reduce 校验通过。
- 2026-10-06 再次只读核对上述源码、模块哈希及 74 SM。不是重新编译或完整 64GB 显存压力测试。

该组合**没有加入 v0.5 的 ECC 改动，也没有 PCIe Gen3/Gen4 补丁**。74 SM 是本机结果，不代表每张卡都可解锁，也不代表推理速度提高 5.7%。当前 PCIe 仍为 Gen2。

## 编译

需要匹配的 kernel headers、GCC 12、make、Python 3；运行时还需 **610.57.04 用户态驱动与 GSP 固件**。二者不随本包分发。

```bash
# 解压完整发布附件后，进入其顶层目录。
cd cmp170hx-610.57.04-sm74-p2p-source
python3 verify-source.py open-gpu-kernel-modules-610.57.04
JOBS=3 bash build-only.sh
```

`build-only.sh` 只构建，不安装、不卸载模块、不重启。默认要求本机已验证的 `6.8.0-138-generic` headers，产物在源码目录的 `kernel-open/` 下。其他内核需自行适配，尤其是 DRM 回调签名；不能视作通用 DKMS 包。

附件里的源码**已打补丁**。若自行从 NVIDIA 原版重建，则在原版 **610.57.04** 根目录执行下面的命令，再用同一个清单校验；不要对附件重复打补丁：

```bash
patch --dry-run --fuzz=0 -p1 < /path/to/stock-to-sm74-p2p.patch
patch --fuzz=0 -p1 < /path/to/stock-to-sm74-p2p.patch
python3 /path/to/verify-source.py .
```

## 安装与回退

先完成编译并停止使用 NVIDIA GPU 的任务。以下是**已有本项目旧版 610.57.04 驱动、模块位于 `updates/cmpunlocker`** 时的替换步骤；安装前用 `modinfo -n nvidia` 确认布局。保留可启动的旧内核/驱动以及本地控制台。

```bash
kver=6.8.0-138-generic
moddir="/lib/modules/$kver/updates/cmpunlocker"
test "$(uname -r)" = "$kver" || exit 1
test "$(modinfo -F version nvidia)" = 610.57.04 || exit 1
test "$(modinfo -n nvidia)" = "$moddir/nvidia.ko" || exit 1
backup="/var/backups/cmp170hx-$(date +%Y%m%d-%H%M%S)"
sudo install -d -m 700 "$backup"
sudo cp -a "$moddir" "$backup/modules"
sudo cp -a "/boot/initrd.img-$kver" "$backup/"
sudo cp -a /etc/modprobe.d "$backup/"
echo "Backup: $backup"

for name in nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem; do
  test "$(modinfo -F version "open-gpu-kernel-modules-610.57.04/kernel-open/$name.ko")" = 610.57.04 || exit 1
  test "$(modinfo -F vermagic "open-gpu-kernel-modules-610.57.04/kernel-open/$name.ko" | cut -d' ' -f1)" = "$kver" || exit 1
done
sudo install -m 644 open-gpu-kernel-modules-610.57.04/kernel-open/nvidia{,-modeset,-uvm,-drm,-peermem}.ko "$moddir/"
sudo depmod -a "$kver"
modprobe -n -v nvidia
sudo update-initramfs -u -k "$kver"
```

记录输出的备份路径，确认 `modprobe -n -v` 仍指向新模块，再自行重启。Secure Boot 开启时，应先完成模块签名与信任配置。不要用旧安装器重新构建覆盖此次 SM 增量。

需回退时，从可用的系统/恢复控制台还原同一备份：

```bash
kver=6.8.0-138-generic
backup=/var/backups/cmp170hx-YYYYMMDD-HHMMSS  # 改成上面实际输出的路径
sudo cp -a "$backup/modules/." "/lib/modules/$kver/updates/cmpunlocker/"
sudo depmod -a "$kver"
sudo cp -a "$backup/initrd.img-$kver" "/boot/initrd.img-$kver"
```

上述替换不修改 modprobe 或 GRUB。本机沿用的参数如下，仅作为验证环境记录：

```text
NVreg_RegistryDwords="RM1457588=1;RM1774520=1;RmForceEnableGen2=1;RMPcieLinkSpeed=0x1;RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1"
intel_iommu=off iommu=off
```

关闭 IOMMU 会改变 DMA 隔离能力，且与虚拟机直通要求有关；本包不自动修改它，也不附带原系统的其他启动参数。

## 重启验证

先核对 `nvidia-smi`、`modinfo` 和内核错误日志。使用已安装 CUDA 版 PyTorch 的环境，可以运行小规模正确性检查：

```bash
mkdir -p p2p-results
CUDA_VISIBLE_DEVICES=0,1 python3 verify-p2p.py p2p-results 2>&1 | tee p2p-results/run.log
```

同时检查结果 JSON 无错误，以及 NCCL 日志**两个方向实际使用 P2P/CUMEM**；仅通过 collective 测试并不能排除网络/主机回退。测试包括非零数据，不能只凭 capability=1、带宽数字或全零结果判断 P2P 正常。

本机模型继续关闭 custom all-reduce。GPU P2P 可用不等于 raw peer 读写或 GDS 已验证。最终还需在自己的机器上验证长上下文推理与持续负载。

## 来源与许可

来源列于 [PATCH_SOURCES.json](PATCH_SOURCES.json)。完整 NVIDIA 源码保留 `COPYING` 和各文件声明；cmpunlocker 许可保留在 `CMPUNLOCKER_LICENSE`，原贡献说明保留在 `CREDITS.md`。不包含闭源用户态驱动、GSP 固件、模型、VBIOS、账号、地址或本机日志。
