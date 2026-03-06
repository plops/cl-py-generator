# Install Gentoo Image on ThinkPad E14 (using existing GRUB)

This guide describes how to install a new live Gentoo image on the ThinkPad E14 workstation by copying new artifacts and updating the GRUB entry on the secondary disk.

## 0. Identifiers for this Machine (E14)

- **Artifacts Partition (Kernel/Squashfs)**: `/dev/nvme0n1p2` (UUID `df544e10-90c0-4315-860c-92a58ec8499e`)
- **GRUB Config Partition**: `/dev/nvme1n1p2` (currently mounted at `/1p2`)
- **Persistent Partition**: `/dev/nvme0n1p4` (LUKS UUID `bbac9bb8-39d9-42fa-8d04-94610ced9839`)
- **Build Date Suffix**: `0306`

## 1. Copy New Build Artifacts

We copy the artifacts to the storage partition (currently at `/run/initramfs/live`). We must remount it as `rw` first.

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo cp -av ~/gentoo-z6-min_20260306/gentoo.squashfs /run/initramfs/live/gentoo.squashfs_0306
sudo cp -av ~/gentoo-z6-min_20260306/vmlinuz /run/initramfs/live/boot/vmlinuz_0306
sudo cp -av ~/gentoo-z6-min_20260306/initramfs_squash_sda1-x86_64.img /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0306
```

## 2. Update GRUB Configuration

The GRUB config on this machine is located on `/dev/nvme1n1p2` (mounted at `/1p2`). We create or append the new entry to its `custom.cfg`.

```bash
cat <<'EOF' | sudo tee -a /1p2/boot/grub/custom.cfg >/dev/null

menuentry 'Gentoo Dracut (E14 persist on nvme0n1p4 0306)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    # Search for the partition containing the artifacts (nvme0n1p2)
    search --no-floppy --fs-uuid --set=root df544e10-90c0-4315-860c-92a58ec8499e

    linux /boot/vmlinuz_0306 \
      root=live:UUID=df544e10-90c0-4315-860c-92a58ec8499e \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0306 \
      rd.live.ram=1 \
      rd.luks.uuid=bbac9bb8-39d9-42fa-8d04-94610ced9839 \
      rd.luks.name=bbac9bb8-39d9-42fa-8d04-94610ced9839=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      nvme_core.default_ps_max_latency_us=0

    initrd /boot/initramfs_squash_sda1-x86_64.img_0306
}
EOF
```

> [!NOTE]
> The `nvme_core.default_ps_max_latency_us=0` parameter is added to prevent "Unsafe Shutdown" logs on the DRAM-less NVMe drives (ADATA/Hynix) used in this machine. It ensures the controller is ready for the shutdown signal from systemd.


## 3. Reboot and Test

1. Reboot the machine.
2. Select `Gentoo Dracut (E14 persist on nvme0n1p4 0306)` from the GRUB menu.
3. Unlock the LUKS partition when prompted.
4. Verify that the system boots into the new image with existing persistence.
