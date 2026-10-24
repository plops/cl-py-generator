#!/bin/bash
# Startet das Minimal-Image in QEMU mit serieller Konsole.
# Nutzung:
#   ./setup05_run_qemu.sh                 # Boot vom Disk-Image (setup04)
#   ./setup05_run_qemu.sh --direct <dir>  # Direktstart Kernel+Initramfs aus <dir>
set -euo pipefail

KVM_FLAGS=""
if [ -e /dev/kvm ]; then
  KVM_FLAGS="-cpu host -enable-kvm"
else
  echo "Hinweis: /dev/kvm fehlt, nutze TCG-Emulation (langsamer)."
  KVM_FLAGS="-cpu max"
fi

if [ "${1:-}" = "--direct" ]; then
  # Kernel-Direktstart ohne GRUB/Loop: Kernel+Initramfs kommen aus <dir>,
  # das Squashfs liegt als Datei auf der vfat-Disk qemu/sda1.img (/dev/vda).
  SRC_DIR="${2:-$(ls -dt /dev/shm/gentoo-qemu-min-openrc_* 2>/dev/null | head -n1)}"
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  exec qemu-system-x86_64 \
    ${KVM_FLAGS} \
    -smp 16 \
    -m 8G \
    -device virtio-net-pci,netdev=n0 \
    -netdev user,id=n0 \
    -kernel "${SRC_DIR}/vmlinuz" \
    -initrd "${SRC_DIR}/initramfs_squash_sda1-x86_64.img" \
    -append "root=live:/dev/vda rd.live.dir=/ rd.live.squashimg=gentoo.squashfs console=ttyS0,115200" \
    -drive "file=${SCRIPT_DIR}/qemu/sda1.img,format=raw,if=virtio" \
    -nographic
else
  exec qemu-system-x86_64 \
    ${KVM_FLAGS} \
    -smp 16 \
    -drive file=qemu/sda1.img,format=raw,if=virtio \
    -m 8G \
    -device virtio-net-pci,netdev=n0 \
    -netdev user,id=n0 \
    -nographic
fi
