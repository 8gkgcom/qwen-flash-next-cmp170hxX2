#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${SOURCE_DIR:-$here/open-gpu-kernel-modules-610.57.04}"
kernel="${KERNEL:-$(uname -r)}"
if [[ "$kernel" != 6.8.0-138-generic ]]; then
  echo "This source snapshot was validated only with 6.8.0-138-generic; review DRM compatibility for other kernels." >&2
  exit 2
fi
test -d "/lib/modules/$kernel/build"
command -v gcc-12 >/dev/null
python3 "$here/verify-source.py" "$source_dir"
make -C "$source_dir" -j"${JOBS:-3}" modules \
  SYSSRC="/lib/modules/$kernel/build" CC=gcc-12
for module in nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem; do
  file="$source_dir/kernel-open/$module.ko"
  test "$(modinfo -F version "$file")" = 610.57.04
  test "$(modinfo -F vermagic "$file" | cut -d' ' -f1)" = "$kernel"
  sha256sum "$file"
done
echo "Build complete. No modules installed, unloaded or reloaded. No reboot requested."
