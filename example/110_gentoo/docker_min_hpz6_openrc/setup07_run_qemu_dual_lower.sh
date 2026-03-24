#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

QEMU_DIR="${QEMU_DIR:-${SCRIPT_DIR}/qemu}"
IMAGE_NAME="${IMAGE_NAME:-openrc-luks-dual-lower.img}"
IMAGE_PATH="${QEMU_DIR}/${IMAGE_NAME}"
EXT4_LOWER_NAME="${EXT4_LOWER_NAME:-gentoo.ext4}"
EXT4_LOWER_PATH="${QEMU_DIR}/${EXT4_LOWER_NAME}"
KERNEL_PATH="${QEMU_DIR}/vmlinuz"
INITRAMFS_PATH="${QEMU_DIR}/initramfs_dual_lower_sda1-x86_64.img"
CMDLINE_PATH="${QEMU_DIR}/cmdline_dual_lower.txt"
MEMORY="${QEMU_MEMORY:-4096}"
SMP_COUNT="${QEMU_SMP:-4}"
CONSOLE_MODE="${QEMU_CONSOLE_MODE:-curses}"
QEMU_DEBUG="${QEMU_DEBUG:-0}"
QEMU_DEBUG_SHELL="${QEMU_DEBUG_SHELL:-0}"
QEMU_LOG_PATH="${QEMU_LOG_PATH:-}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "missing image: ${IMAGE_PATH}" >&2
  echo "create it first with sudo ./setup06_create_qemu_dual_lower.sh" >&2
  exit 1
fi

if [[ ! -w "${IMAGE_PATH}" ]]; then
  echo "image is not writable: ${IMAGE_PATH}" >&2
  echo "QEMU opens the disk read-write by default; recreate it with sudo ./setup06_create_qemu_dual_lower.sh" >&2
  echo "or fix permissions manually before running this script." >&2
  exit 1
fi

if [[ ! -f "${EXT4_LOWER_PATH}" ]]; then
  echo "missing ext4 lower image: ${EXT4_LOWER_PATH}" >&2
  echo "create it first with sudo ./setup06_create_qemu_dual_lower.sh" >&2
  exit 1
fi

if [[ ! -f "${KERNEL_PATH}" || ! -f "${INITRAMFS_PATH}" || ! -f "${CMDLINE_PATH}" ]]; then
  echo "missing kernel/initramfs/cmdline payloads in ${QEMU_DIR}; recreate the image first" >&2
  exit 1
fi

CMDLINE="$(tr '\n' ' ' < "${CMDLINE_PATH}")"

append_cmdline_arg() {
  local arg="$1"
  if [[ " ${CMDLINE} " != *" ${arg} "* ]]; then
    CMDLINE="${CMDLINE} ${arg}"
  fi
}

if [[ "${QEMU_DEBUG}" == "1" ]]; then
  append_cmdline_arg "rd.debug"
  append_cmdline_arg "rd.udev.log_level=debug"
  append_cmdline_arg "log_buf_len=1M"
  append_cmdline_arg "systemd.log_level=debug"
  append_cmdline_arg "systemd.log_target=console"
  append_cmdline_arg "ignore_loglevel"
  if [[ "${QEMU_DEBUG_SHELL}" == "1" ]]; then
    append_cmdline_arg "rd.shell"
  fi
  if [[ -z "${QEMU_CONSOLE_MODE:-}" ]]; then
    CONSOLE_MODE="serial"
  fi
  if [[ -z "${QEMU_LOG_PATH}" ]]; then
    mkdir -p "${SCRIPT_DIR}/logs"
    QEMU_LOG_PATH="${SCRIPT_DIR}/logs/qemu_boot_dual_lower_$(date +%Y%m%d_%H%M%S).log"
  fi
fi

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

QEMU_CMD=(
  qemu-system-x86_64
  -machine q35
  "${ACCEL_ARGS[@]}"
  -smp "${SMP_COUNT}"
  -m "${MEMORY}"
  -kernel "${KERNEL_PATH}"
  -initrd "${INITRAMFS_PATH}"
  -append "${CMDLINE}"
  -drive "file=${IMAGE_PATH},format=raw,if=virtio"
  -drive "file=${EXT4_LOWER_PATH},format=raw,if=virtio,readonly=on"
  -netdev user,id=n1
  -device virtio-net-pci,netdev=n1
  "${DISPLAY_ARGS[@]}"
  -no-reboot
)

if [[ -n "${QEMU_LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${QEMU_LOG_PATH}")"
  echo "QEMU log: ${QEMU_LOG_PATH}" >&2
  printf 'Kernel cmdline: %s\n' "${CMDLINE}" >&2
  "${QEMU_CMD[@]}" 2>&1 | tee "${QEMU_LOG_PATH}"
else
  exec "${QEMU_CMD[@]}"
fi
