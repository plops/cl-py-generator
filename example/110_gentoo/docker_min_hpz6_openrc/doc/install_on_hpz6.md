# Install Gentoo OpenRC Image on HP Z6 (using existing GRUB)

This guide describes how to install the new HP Z6 OpenRC live image on the HP Z6 workstation by copying new boot artifacts to the existing Gentoo artifacts partition and adding a new GRUB entry.

It follows the same partition layout already in use on this machine:

- Boot artifacts and squashfs images stay on `/dev/nvme0n1p3` (label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`)
- Persistent writable storage stays on `/dev/nvme0n1p5` via LUKS (`UUID=0d7c5e23-6bab-4dce-b744-a5d61d497aca`) and `/dev/mapper/enc`
- Existing GRUB installation is reused

Current deployment target:

- New build directory: `/dev/shm/gentoo-z6-min-openrc_20260407/`
- New HP Z6 squashfs source: `/dev/shm/gentoo-z6-min-openrc_20260407/gentoo.squashfs_nv`
- New version suffix: `0407`
- Current booted fallback that must be kept: `0324`

## 0. Current machine state

On this HP Z6, the relevant runtime state is:

- `/run/initramfs/live` is mounted from `/dev/nvme0n1p3`
- `/proc/cmdline` currently uses:
  - `root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23`
  - `rd.live.squashimg=gentoo.squashfs_0324`
  - `rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca`
  - `rd.overlay=/dev/mapper/enc:persistent`
- Current GRUB custom entry uses the `0324` kernel/initramfs/squashfs set

Currently present squashfs files on `/dev/nvme0n1p3`:

- `gentoo.squashfs_0225` about `1.4G`
- `gentoo.squashfs_0302` about `2.4G`
- `gentoo.squashfs_0306` about `2.7G`
- `gentoo.squashfs_0309` about `2.6G`
- `gentoo.squashfs_0311` about `2.4G`
- `gentoo.squashfs_0324` about `2.3G`

There is currently about `17G` free on `/dev/nvme0n1p3`, so there should be enough space to keep all existing squashfs files and add the new `0407` set.

## 1. Preflight checks

Run these first:

```bash
lsblk -f
df -h /run/initramfs/live
cat /proc/cmdline
```

Expected identifiers on this machine:

- Artifacts partition: `/dev/nvme0n1p3`, label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`
- Persistent LUKS partition: `/dev/nvme0n1p5`, UUID `0d7c5e23-6bab-4dce-b744-a5d61d497aca`
- Open persistence mapper: `/dev/mapper/enc`

Inspect the existing artifacts and GRUB entry:

```bash
ls -lh /run/initramfs/live/gentoo.squashfs_*
ls -lh /run/initramfs/live/boot/vmlinuz_* /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_* 2>/dev/null
sed -n '1,200p' /run/initramfs/live/boot/grub/custom.cfg
```

## 2. Remount the artifacts partition read-write

The live artifacts partition is usually mounted read-only while booted from it, so remount it before copying:

```bash
sudo mount -o remount,rw /run/initramfs/live
mount | rg '/run/initramfs/live'
```

Expected result:

- `/run/initramfs/live` remains mounted from `/dev/nvme0n1p3`
- mount flags now include `rw`

## 3. Verify the new build output

Check that the expected OpenRC HP Z6 artifacts exist:

```bash
ls -lh /dev/shm/gentoo-z6-min-openrc_20260407/
ls -lh \
  /dev/shm/gentoo-z6-min-openrc_20260407/gentoo.squashfs_nv \
  /dev/shm/gentoo-z6-min-openrc_20260407/vmlinuz \
  /dev/shm/gentoo-z6-min-openrc_20260407/initramfs_squash_sda1-x86_64.img \
  /dev/shm/gentoo-z6-min-openrc_20260407/packages.txt \
  /dev/shm/gentoo-z6-min-openrc_20260407/packages.tsv
```

Important for HP Z6:

- Use `gentoo.squashfs_nv`
- Do not use the E14-specific `gentoo.squashfs_e14`
- `packages.txt` is the human-readable package manifest to archive on the boot partition
- `packages.tsv` is the machine-readable package/build-time manifest to archive alongside it
- `copy_files.sh` may be kept with the exported bundle for reference, but the installation steps below copy the artifacts explicitly

## 4. Storage policy: what to keep and what to delete only if needed

Recommended keep set:

- Keep `gentoo.squashfs_0324` because that is the currently booted known-good fallback
- Keep the older squashfs files as long as free space remains comfortable
- Keep matching kernel and initramfs files for the versions you keep bootable in GRUB

With about `17G` free on `/dev/nvme0n1p3`, the current recommendation is:

- Keep all existing squashfs files
- Add the new `0407` files without deleting anything first

If the copy later fails because space becomes tight, delete in this order:

1. `gentoo.squashfs_0225` and its matching `boot/vmlinuz_0225` and `boot/initramfs_squash_sda1-x86_64.img_0225`
2. `gentoo.squashfs_0302` and its matching `boot/vmlinuz_0302` and `boot/initramfs_squash_sda1-x86_64.img_0302`
3. `gentoo.squashfs_0306` and its matching `boot/vmlinuz_0306` and `boot/initramfs_squash_sda1-x86_64.img_0306`
4. `gentoo.squashfs_0309` and its matching `boot/vmlinuz_0309` and `boot/initramfs_squash_sda1-x86_64.img_0309`
5. `gentoo.squashfs_0311` and its matching `boot/vmlinuz_0311` and `boot/initramfs_squash_sda1-x86_64.img_0311`

Do not delete:

- `gentoo.squashfs_0324`
- `boot/vmlinuz_0324`
- `boot/initramfs_squash_sda1-x86_64.img_0324`

Optional cleanup commands if needed later:

```bash
sudo rm -v /run/initramfs/live/gentoo.squashfs_0225
sudo rm -v /run/initramfs/live/boot/vmlinuz_0225
sudo rm -v /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0225
```

Re-check free space after each cleanup step:

```bash
df -h /run/initramfs/live
```

## 5. Copy the new `0407` artifacts

Copy the new OpenRC HP Z6 artifacts with versioned names:

```bash
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260407/gentoo.squashfs_nv \
  /run/initramfs/live/gentoo.squashfs_0407
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260407/vmlinuz \
  /run/initramfs/live/boot/vmlinuz_0407
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260407/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0407
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260407/packages.txt \
  /run/initramfs/live/packages_0407.txt
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260407/packages.tsv \
  /run/initramfs/live/packages_0407.tsv
```

Verify:

```bash
ls -lh /run/initramfs/live/gentoo.squashfs_0407
ls -lh /run/initramfs/live/boot/vmlinuz_0407
ls -lh /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0407
ls -lh /run/initramfs/live/packages_0407.txt /run/initramfs/live/packages_0407.tsv
df -h /run/initramfs/live
```

## 6. Add a new GRUB entry and keep `0324` as fallback

Append a new entry to `/run/initramfs/live/boot/grub/custom.cfg` and keep the existing `0324` entry unchanged.

Recommended new entry:

```bash
cat <<'EOF' | sudo tee -a /run/initramfs/live/boot/grub/custom.cfg >/dev/null

menuentry 'Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0407 \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0407 \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/initramfs_squash_sda1-x86_64.img_0407
}
EOF
```

Verify that both the old and new entries are present:

```bash
rg -n "0324|0407" /run/initramfs/live/boot/grub/custom.cfg
```

## 7. Final checks before reboot

Run:

```bash
sync
ls -lh /run/initramfs/live/gentoo.squashfs_0324 /run/initramfs/live/gentoo.squashfs_0407
ls -lh /run/initramfs/live/boot/vmlinuz_0324 /run/initramfs/live/boot/vmlinuz_0407
ls -lh /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0324 /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0407
ls -lh /run/initramfs/live/packages_0407.txt /run/initramfs/live/packages_0407.tsv
sed -n '1,220p' /run/initramfs/live/boot/grub/custom.cfg
```

Checklist:

- `0324` files still exist
- `0407` files exist and have non-zero size
- `packages_0407.txt` and `packages_0407.tsv` are present if you copied the manifests
- `custom.cfg` contains both the existing `0324` entry and the new `0407` entry
- `/dev/mapper/enc` is still the overlay target referenced by the new entry

## 8. Reboot and select the new entry

Reboot the machine:

```bash
sudo reboot
```

At the GRUB menu:

- Select `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)` for the new image
- Keep using the `0324` entry as rollback fallback if the new image has a regression

## 9. Re-establish network configuration after boot

On the currently working HP Z6 setup, network state looks like this:

- `eno1` uses DHCP and provides the default route
- `enp65s0` has static addresses `192.168.254.123/24` and `192.168.178.122/24`
- DNS normally comes from:
  - Tailscale-managed `/etc/resolv.conf`, or
  - fallback `/etc/resolv.pre-tailscale-backup.conf`

Relevant config files on this machine:

- `/etc/systemd/network/10-eno1.network`
- `/etc/systemd/network/20-enp65s0.network`
- `/etc/systemd/resolved.conf`
- `/etc/resolv.conf`
- `/etc/resolv.pre-tailscale-backup.conf`

The current network definitions are:

```ini
# /etc/systemd/network/10-eno1.network
[Match]
Name=eno1

[Network]
DHCP=yes

[DHCPv4]
UseDomains=lgs-net.com
```

```ini
# /etc/systemd/network/20-enp65s0.network
[Match]
Name=enp65s0

[Network]
Address=192.168.254.123/24
Address=192.168.178.122/24
```

The currently working routes are:

```text
default via 10.60.120.5 dev eno1 proto dhcp src 10.60.120.97 metric 1024
10.60.120.0/23 dev eno1 proto kernel scope link src 10.60.120.97 metric 1024
192.168.178.0/24 dev enp65s0 proto kernel scope link src 192.168.178.122
192.168.254.0/24 dev enp65s0 proto kernel scope link src 192.168.254.123
```

### Standard recovery sequence

If the new image boots but the network is missing, run:

```bash
sudo ip link set eno1 up
sudo ip link set enp65s0 up
sudo systemctl restart systemd-networkd
sudo systemctl restart systemd-resolved
sleep 3
networkctl status eno1 enp65s0
ip a show eno1
ip a show enp65s0
ip route
resolvectl status
```

Expected result:

- `eno1` receives a DHCP lease on `10.60.120.0/23`
- default route appears via `10.60.120.5`
- `enp65s0` gets `192.168.254.123/24` and `192.168.178.122/24`

### If DNS is broken

Check whether `/etc/resolv.conf` still contains the current Tailscale-generated nameservers:

```bash
sed -n '1,120p' /etc/resolv.conf
```

Current working content on this machine:

```text
nameserver 194.11.90.70
nameserver 100.100.100.100
search tail6e39c9.ts.net lgs-net.com
```

If Tailscale is not up yet and DNS resolution fails, restore the non-Tailscale fallback:

```bash
sudo cp -av /etc/resolv.pre-tailscale-backup.conf /etc/resolv.conf
sed -n '1,120p' /etc/resolv.conf
```

Expected fallback content:

```text
nameserver 194.11.90.70
nameserver 194.11.92.70
search lgs-net.com
```

Then test:

```bash
ping -c 2 194.11.90.70
ping -c 2 8.8.8.8
getent hosts gentoo.org
```

### If `eno1` did not get DHCP

Check the networkd config and logs:

```bash
sed -n '1,120p' /etc/systemd/network/10-eno1.network
sudo journalctl -b -u systemd-networkd --no-pager | tail -n 120
networkctl status eno1
```

Then retry by cycling the link and restarting networkd:

```bash
sudo ip link set eno1 down
sleep 2
sudo ip link set eno1 up
sudo systemctl restart systemd-networkd
sleep 3
ip a show eno1
ip route
```

### If `enp65s0` lost its static addresses

Check the static config and re-apply it:

```bash
sed -n '1,120p' /etc/systemd/network/20-enp65s0.network
sudo systemctl restart systemd-networkd
sleep 3
ip a show enp65s0
```

As a temporary manual fallback, the two static addresses can be re-added directly:

```bash
sudo ip addr add 192.168.254.123/24 dev enp65s0
sudo ip addr add 192.168.178.122/24 dev enp65s0
ip a show enp65s0
```

### Optional: restart Tailscale after base networking works

Once plain networking and DNS are back, restore the Tailscale-managed resolver state if needed:

```bash
sudo systemctl restart tailscaled
sudo tailscale up
sed -n '1,120p' /etc/resolv.conf
```

## 10. Rollback

If the new image does not boot cleanly:

- Reboot
- Select the existing `0324` menu entry
- Leave the `0407` files on disk for later inspection unless you need the space immediately

If you want to remove the `0407` deployment after a failed test:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -v /run/initramfs/live/gentoo.squashfs_0407
sudo rm -v /run/initramfs/live/boot/vmlinuz_0407
sudo rm -v /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0407
sudo rm -v /run/initramfs/live/packages_0407.txt
sudo rm -v /run/initramfs/live/packages_0407.tsv
```

Then remove the corresponding `0407` GRUB entry from `/run/initramfs/live/boot/grub/custom.cfg`.
