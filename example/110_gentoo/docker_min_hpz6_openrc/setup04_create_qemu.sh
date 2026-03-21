#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ${EUID} -ne 0 ]]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi

QEMU_DIR="${QEMU_DIR:-${SCRIPT_DIR}/qemu}"
IMAGE_NAME="${IMAGE_NAME:-openrc-luks-squashfs.img}"
IMAGE_PATH="${QEMU_DIR}/${IMAGE_NAME}"
IMAGE_SIZE="${IMAGE_SIZE:-8192M}"
BOOT_SIZE_MIB="${BOOT_SIZE_MIB:-1400}"
HOST_MAPPER_NAME="${HOST_MAPPER_NAME:-qemuenc}"
GUEST_MAPPER_NAME="${GUEST_MAPPER_NAME:-enc}"
CRYPT_NAME="${CRYPT_NAME:-${HOST_MAPPER_NAME}}"
LUKS_PASSPHRASE="openrc-test"
ARTIFACT_DIR="${ARTIFACT_DIR:-}"
BOOT_LABEL="${BOOT_LABEL:-QEMUBOOT}"
PERSIST_LABEL="${PERSIST_LABEL:-PERSIST}"
MOUNT_BOOT="/mnt/qemu-boot.$$"
MOUNT_LIVE="/mnt/qemu-live.$$"
LOOP_DEVICE=""
KERNEL_OUT="${QEMU_DIR}/vmlinuz"
INITRAMFS_OUT="${QEMU_DIR}/initramfs_squash_sda1-x86_64.img"
CMDLINE_OUT="${QEMU_DIR}/cmdline.txt"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
  set +e
  if mountpoint -q "${MOUNT_LIVE}"; then
    umount "${MOUNT_LIVE}"
  fi
  if mountpoint -q "${MOUNT_BOOT}"; then
    umount "${MOUNT_BOOT}"
  fi
  if cryptsetup status "${CRYPT_NAME}" >/dev/null 2>&1; then
    cryptsetup close "${CRYPT_NAME}"
  fi
  if [[ -n "${LOOP_DEVICE}" ]] && losetup "${LOOP_DEVICE}" >/dev/null 2>&1; then
    losetup -d "${LOOP_DEVICE}"
  fi
  rmdir "${MOUNT_LIVE}" "${MOUNT_BOOT}" 2>/dev/null || true
}

trap cleanup EXIT

find_artifact_dir() {
  if [[ -n "${ARTIFACT_DIR}" ]]; then
    printf '%s\n' "${ARTIFACT_DIR}"
    return 0
  fi

  local latest
  latest="$(
    find /dev/shm -maxdepth 1 -type d -name 'gentoo-z6-min-openrc_*' -printf '%T@ %p\n' \
      | sort -nr \
      | awk 'NR==1 {print $2}'
  )"

  if [[ -z "${latest}" ]]; then
    echo "no artifact directory found under /dev/shm; run ./setup03_copy_from_container.sh first" >&2
    exit 1
  fi

  printf '%s\n' "${latest}"
}

pick_squashfs() {
  local dir="$1"

  for candidate in \
    "${dir}/gentoo.squashfs_e14" \
    "${dir}/gentoo.squashfs"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "no squashfs artifact found in ${dir}" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "required file missing: ${path}" >&2
    exit 1
  }
}

ARTIFACT_DIR="$(find_artifact_dir)"
SQUASHFS_SRC="$(pick_squashfs "${ARTIFACT_DIR}")"
KERNEL_SRC="${ARTIFACT_DIR}/vmlinuz"
INITRAMFS_SRC="${ARTIFACT_DIR}/initramfs_squash_sda1-x86_64.img"
PACKAGES_SRC="${ARTIFACT_DIR}/packages.txt"

require_file "${KERNEL_SRC}"
require_file "${INITRAMFS_SRC}"
require_file "${SQUASHFS_SRC}"

mkdir -p "${QEMU_DIR}"
rm -f "${IMAGE_PATH}"

log "using artifacts from ${ARTIFACT_DIR}"
log "creating raw disk image ${IMAGE_PATH} (${IMAGE_SIZE})"
qemu-img create -f raw "${IMAGE_PATH}" "${IMAGE_SIZE}" >/dev/null

LOOP_DEVICE="$(losetup --find --partscan --show "${IMAGE_PATH}")"
log "loop device: ${LOOP_DEVICE}"

cat <<EOF | sfdisk "${LOOP_DEVICE}" >/dev/null
label: dos

start=1MiB, size=$((BOOT_SIZE_MIB - 1))MiB, type=83, bootable
start=${BOOT_SIZE_MIB}MiB, type=83
EOF

udevadm settle
mkfs.ext4 -F -L "${BOOT_LABEL}" "${LOOP_DEVICE}p1" >/dev/null

log "formatting ${LOOP_DEVICE}p2 as LUKS"
printf '%s' "${LUKS_PASSPHRASE}" | cryptsetup luksFormat --batch-mode --type luks2 "${LOOP_DEVICE}p2" -
printf '%s' "${LUKS_PASSPHRASE}" | cryptsetup open "${LOOP_DEVICE}p2" "${CRYPT_NAME}" -
mkfs.ext4 -F -L "${PERSIST_LABEL}" "/dev/mapper/${CRYPT_NAME}" >/dev/null

mkdir -p "${MOUNT_BOOT}" "${MOUNT_LIVE}"
mount "${LOOP_DEVICE}p1" "${MOUNT_BOOT}"
mount "/dev/mapper/${CRYPT_NAME}" "${MOUNT_LIVE}"

mkdir -p "${MOUNT_BOOT}/boot/grub" "${MOUNT_LIVE}/persistent/upper" "${MOUNT_LIVE}/persistent/work"
cp -av "${KERNEL_SRC}" "${MOUNT_BOOT}/vmlinuz"
cp -av "${INITRAMFS_SRC}" "${MOUNT_BOOT}/initramfs_squash_sda1-x86_64.img"
cp -av "${SQUASHFS_SRC}" "${MOUNT_BOOT}/gentoo.squashfs"
cp -av "${KERNEL_SRC}" "${KERNEL_OUT}"
cp -av "${INITRAMFS_SRC}" "${INITRAMFS_OUT}"

if [[ -f "${PACKAGES_SRC}" ]]; then
  cp -av "${PACKAGES_SRC}" "${MOUNT_BOOT}/packages.txt"
fi

LUKS_UUID="$(cryptsetup luksUUID "${LOOP_DEVICE}p2")"
cat > "${CMDLINE_OUT}" <<EOF
root=live:/dev/vda1 rd.live.dir=/ rd.live.squashimg=gentoo.squashfs rd.live.ram=1 rd.live.overlay.overlayfs=1 rd.luks.uuid=${LUKS_UUID} rd.luks.name=${LUKS_UUID}=${GUEST_MAPPER_NAME} console=tty0 console=ttyS0,115200
EOF

sync
log "image ready: ${IMAGE_PATH}"
log "kernel: ${KERNEL_OUT}"
log "initramfs: ${INITRAMFS_OUT}"
log "cmdline: ${CMDLINE_OUT}"
log "luks uuid: ${LUKS_UUID}"
log "host-side mapper: ${CRYPT_NAME}"
log "guest-side mapper: ${GUEST_MAPPER_NAME}"
log "default passphrase: ${LUKS_PASSPHRASE}"
