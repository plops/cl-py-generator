#!/bin/bash
# Erzeugt ein simples QEMU-Boot-Image (angelehnt an ../docker_min/setup04_create_qemu.sh).
# Minimal-Variante ohne LUKS/LVM: vfat-Partition mit Kernel, Initramfs, Squashfs.
# Erwartet die Artefakte aus setup03_copy_from_container.sh in /dev/shm/gentoo-qemu-min-openrc_*/.
set -euo pipefail

MTOOLS_MODE=""
if [ "${1:-}" = "--mtools" ]; then MTOOLS_MODE=1; shift; fi
SRC_DIR="${1:-$(ls -dt /dev/shm/gentoo-qemu-min-openrc_* 2>/dev/null | head -n1)}"
if [ -z "${SRC_DIR}" ] || [ ! -d "${SRC_DIR}" ]; then
  echo "Keine Build-Artefakte gefunden. Erst setup03_copy_from_container.sh laufen lassen." >&2
  exit 1
fi
echo "Quelle: ${SRC_DIR}"
ls -la "${SRC_DIR}"

mkdir -p qemu
IMG=qemu/sda1.img
rm -f "${IMG}"
qemu-img create -f raw "${IMG}" 4G

# Pfad ohne Loop-Geraete (z.B. im AI-Container): vfat direkt per mtools
# befüllen, Boot danach per setup05 --direct (Kernel-Direktstart, kein GRUB).
if [ -n "${MTOOLS_MODE}" ] || ! losetup -f >/dev/null 2>&1; then
  echo "Baue loop-freies vfat-Image per mtools ..."
  mformat -i "${IMG}" -h 64 -t 512 -v GENTOOQEMU ::
  mcopy -i "${IMG}" "${SRC_DIR}/vmlinuz" ::vmlinuz
  mcopy -i "${IMG}" "${SRC_DIR}/initramfs_squash_sda1-x86_64.img" ::initramfs.img
  mcopy -i "${IMG}" "${SRC_DIR}/gentoo.squashfs_qemu" ::gentoo.squashfs
  mdir -i "${IMG}" ::
  echo "Fertig: ${IMG} (Boot per ./setup05_run_qemu.sh --direct)"
  exit 0
fi

LOOP_DEVICE=""
cleanup() {
  if mountpoint -q /mnt; then umount /mnt; fi
  if [ -n "${LOOP_DEVICE}" ]; then losetup -d "${LOOP_DEVICE}" || true; fi
}
trap cleanup EXIT

losetup -fP "${IMG}"
LOOP_DEVICE=$(losetup -j "${IMG}" | awk '{print $1}' | sed 's/://')
echo "Loop: ${LOOP_DEVICE}"

parted -s "${LOOP_DEVICE}" mklabel msdos
parted -s "${LOOP_DEVICE}" mkpart primary fat32 1MiB 100%
parted -s "${LOOP_DEVICE}" set 1 boot on
mkfs.vfat "${LOOP_DEVICE}p1"

mount "${LOOP_DEVICE}p1" /mnt
mkdir -p /mnt/boot/grub
cp "${SRC_DIR}/vmlinuz" /mnt/
cp "${SRC_DIR}/initramfs_squash_sda1-x86_64.img" /mnt/
cp "${SRC_DIR}/gentoo.squashfs_qemu" /mnt/gentoo.squashfs
grub-install --target=i386-pc --boot-directory=/mnt/boot "${LOOP_DEVICE}"

cat > /mnt/boot/grub/grub.cfg <<'EOF'
set default=0
set timeout=5
set root=(hd0,1)

menuentry 'Gentoo Minimal QEMU (serial console)' {
    insmod part_msdos
    echo 'Loading Linux ...'
    linux /vmlinuz root=live:/dev/sda1 rd.live.dir=/ rd.live.squashimg=gentoo.squashfs console=ttyS0,115200
    initrd /initramfs_squash_sda1-x86_64.img
}
EOF

df -h /mnt
umount /mnt
losetup -d "${LOOP_DEVICE}"
LOOP_DEVICE=""
trap - EXIT
echo "Fertig: ${IMG}"
