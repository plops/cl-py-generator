#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

QEMU_DIR="${QEMU_DIR:-${SCRIPT_DIR}/qemu}"
IMAGE_NAME="${IMAGE_NAME:-openrc-luks-squashfs.img}"
IMAGE_PATH="${QEMU_DIR}/${IMAGE_NAME}"
KERNEL_PATH="${QEMU_DIR}/vmlinuz"
INITRAMFS_PATH="${QEMU_DIR}/initramfs_squash_sda1-x86_64.img"
CMDLINE_PATH="${QEMU_DIR}/cmdline.txt"
MEMORY="${QEMU_MEMORY:-4096}"
SMP_COUNT="${QEMU_SMP:-4}"
CONSOLE_MODE="${QEMU_CONSOLE_MODE:-curses}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "missing image: ${IMAGE_PATH}" >&2
  echo "create it first with sudo ./setup04_create_qemu.sh" >&2
  exit 1
fi

if [[ ! -f "${KERNEL_PATH}" || ! -f "${INITRAMFS_PATH}" || ! -f "${CMDLINE_PATH}" ]]; then
  echo "missing kernel/initramfs/cmdline payloads in ${QEMU_DIR}; recreate the image first" >&2
  exit 1
fi

CMDLINE="$(tr '\n' ' ' < "${CMDLINE_PATH}")"

ACCEL_ARGS=()
if [[ -e /dev/kvm && -r /dev/kvm && -w /dev/kvm ]]; then
  ACCEL_ARGS=(-accel kvm -cpu host)
else
  ACCEL_ARGS=(-accel tcg -cpu max)
fi

DISPLAY_ARGS=()
case "${CONSOLE_MODE}" in
  serial)
    DISPLAY_ARGS=(-serial mon:stdio -display none)
    ;;
  curses)
    CMDLINE="$(
      printf '%s\n' "${CMDLINE}" \
        | sed -E 's/(^| )console=ttyS0,115200//g; s/(^| )console=tty0/ console=tty1/g' \
        | tr -s ' '
    ) consoleblank=0"
    DISPLAY_ARGS=(-display curses -serial none -monitor none)
    ;;
  *)
    echo "unsupported QEMU_CONSOLE_MODE: ${CONSOLE_MODE}" >&2
    echo "supported modes: serial, curses" >&2
    exit 1
    ;;
esac

exec qemu-system-x86_64 \
  -machine q35 \
  "${ACCEL_ARGS[@]}" \
  -smp "${SMP_COUNT}" \
  -m "${MEMORY}" \
  -kernel "${KERNEL_PATH}" \
  -initrd "${INITRAMFS_PATH}" \
  -append "${CMDLINE}" \
  -drive "file=${IMAGE_PATH},format=raw,if=virtio" \
  -netdev user,id=n1 \
  -device virtio-net-pci,netdev=n1 \
  "${DISPLAY_ARGS[@]}" \
  -no-reboot
