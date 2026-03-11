# Install Gentoo Image on HP Z6 (using existing GRUB)

This guide describes how to install a new live Gentoo image on an HP Z6 workstation **without** reinstalling GRUB.

It is not a clean-slate install:
- Existing bootloader (GRUB) is reused.
- Existing Linux installation is used as the working environment.
- A new encrypted persistent partition is prepared on `nvme0n1p5`.
- Kernel, initramfs, and squashfs remain on an unencrypted partition.

Current image set used in this guide:
- Build directory: `gentoo-z6-min_20260311`
- Version suffix for copied boot artifacts: `0311`

## 0. Preflight and safety checks

Run these checks first:

```bash
lsblk -f
```

On the target HP Z6 used for this guide, the relevant identifiers are:
- Boot/squashfs partition: `/dev/nvme0n1p3` (label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`)
- Persistent encrypted partition: `/dev/nvme0n1p5` (LUKS UUID `0d7c5e23-6bab-4dce-b744-a5d61d497aca`)
- Open mapper for persistence: `/dev/mapper/enc` (ext4 label `persist`)

Before editing GRUB config directly, always create a backup:

```bash
sudo cp -av /0p3/boot/grub/grub.cfg /0p3/boot/grub/grub.cfg.bak_$(date +%Y%m%d_%H%M%S)
```

## 1. Inspect disks and choose persistent partition

Boot into an existing Linux installation and inspect disks:

```bash
lsblk
```

Example layout (shortened):

```text
nvme0n1
├─nvme0n1p3   (contains existing Gentoo boot artifacts and squashfs)
└─nvme0n1p5   (will be repurposed as encrypted persistent storage)
```

Decision:
- Wipe `nvme0n1p5`
- Create `LUKS2 -> ext4` for persistence

Rationale:
- Keep boot artifacts (`vmlinuz`, `initramfs`, `squashfs`) on unencrypted storage.
- Encrypt only the persistent writable layer.

## 2. Verify where current Gentoo boots from

Check current kernel command line:

```bash
cat /proc/cmdline
```

Current setup references:
- `squashfs.part=/dev/disk/by-label/gentoo`

Verify label target:

```bash
ls -ltr /dev/disk/by-label/gentoo
```

Expected result:
- `/dev/disk/by-label/gentoo -> ../../nvme0n1p3`

Mount it to inspect contents:

```bash
sudo mkdir -p /0p3
mountpoint -q /0p3 || sudo mount /dev/nvme0n1p3 /0p3
ls /0p3
ls /0p3/boot
```

You should see previous dated artifacts such as:
- `/0p3/gentoo.squashfs_0306`
- `/0p3/boot/vmlinuz_0306`
- `/0p3/boot/initramfs_squash_sda1-x86_64.img_0306`

## 3. Check existing GRUB entry

Inspect GRUB config on that partition:

```bash
grep -i gentoo -A 10 /0p3/boot/grub/grub.cfg | head -n 12
```

Confirm:
- `search --fs-uuid` points to the partition that hosts `/boot` and squashfs (`UUID=4f708c84-...` in this setup).

## 4. Copy new build artifacts with date suffix

Artifacts were copied from container output (`setup03_copy_from_container.sh`) and are located in:
- `gentoo-z6-min_20260311/`
- `/dev/shm/gentoo-z6-min_20260311/`

Example listing:

```bash
ls -tlrh /dev/shm/gentoo-z6-min_20260311/
```

Expected files:
- `gentoo.squashfs`
- `vmlinuz`
- `initramfs_squash_sda1-x86_64.img`
- `packages.txt`

Important:
- For HP Z6, copy only `gentoo.squashfs` (the `_e14` suffix image is for the laptop).

Copy to boot partition with rollback-friendly suffixes:

```bash
sudo cp -av /dev/shm/gentoo-z6-min_20260311/gentoo.squashfs /0p3/gentoo.squashfs_0311
sudo cp -av /dev/shm/gentoo-z6-min_20260311/vmlinuz /0p3/boot/vmlinuz_0311
sudo cp -av /dev/shm/gentoo-z6-min_20260311/initramfs_squash_sda1-x86_64.img /0p3/boot/initramfs_squash_sda1-x86_64.img_0311
```

Verify copied files:

```bash
ls -lh /0p3/gentoo.squashfs_0311
ls -lh /0p3/boot/vmlinuz_0311 /0p3/boot/initramfs_squash_sda1-x86_64.img_0311
```

## 5. Create encrypted persistent partition (`nvme0n1p5`)

If persistence on `nvme0n1p5` is already configured and working, **skip this section** to avoid destructive reformatting.

Set variables:

```bash
export PERSIST_PART=/dev/nvme0n1p5
export CRYPT_NAME=persist
```

Create LUKS2 container and ext4 filesystem:

```bash
sudo cryptsetup luksFormat --type luks2 "$PERSIST_PART"
sudo cryptsetup open "$PERSIST_PART" "$CRYPT_NAME"
sudo mkfs.ext4 -L persist "/dev/mapper/$CRYPT_NAME"
sudo mkdir -p /mnt/persist
sudo mount "/dev/mapper/$CRYPT_NAME" /mnt/persist
sudo blkid "$PERSIST_PART" "/dev/mapper/$CRYPT_NAME"
```

Example UUIDs from this machine:

```text
/dev/nvme0n1p5: UUID="0d7c5e23-6bab-4dce-b744-a5d61d497aca" TYPE="crypto_LUKS"
/dev/mapper/persist: UUID="3ca5dfb2-35c9-4ed5-906a-f965dbcd1c7b" TYPE="ext4"
```

Boot partition UUID (where squashfs is loaded from) stays unchanged:

```bash
sudo blkid /dev/disk/by-label/gentoo
```

In this setup:
- Boot/squashfs UUID: `4f708c84-185d-437b-a03a-7a565f598a23`

## 6. Add GRUB menu entry for dracut live boot + encrypted overlay

Preferred location:
- Put custom entries into `/0p3/boot/grub/custom.cfg` (it is sourced by `grub.cfg`).
- This survives `grub-mkconfig` regeneration better than editing `grub.cfg` directly.

Use this entry (adapt UUIDs/paths if your system differs):

```cfg
menuentry 'Gentoo Dracut (persist on nvme0n1p5)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0311 \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0311 \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/initramfs_squash_sda1-x86_64.img_0311
}
```

Recommended menu title convention:
- `Gentoo Dracut (persist on nvme0n1p5 0311)`

Example append command:

```bash
cat <<'EOF' | sudo tee -a /0p3/boot/grub/custom.cfg >/dev/null
menuentry 'Gentoo Dracut (persist on nvme0n1p5 0311)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0311 \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0311 \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/initramfs_squash_sda1-x86_64.img_0311
}
EOF
```

### Migrate an existing entry from `grub.cfg` to `custom.cfg`

If you already appended the entry directly to `grub.cfg`, move it to `custom.cfg` for cleaner long-term maintenance:

```bash
# 1) Backup grub.cfg
sudo cp -av /0p3/boot/grub/grub.cfg /0p3/boot/grub/grub.cfg.bak_migrate_$(date +%Y%m%d_%H%M%S)

# 2) Write the entry into /boot/grub/custom.cfg
cat <<'EOF' | sudo tee /0p3/boot/grub/custom.cfg >/dev/null
menuentry 'Gentoo Dracut (persist on nvme0n1p5 0311)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0311 \
    root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
    rd.live.dir=/ \
    rd.live.squashimg=gentoo.squashfs_0311 \
    rd.live.ram=1 \
    rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
    rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
    rd.overlay=/dev/mapper/enc:persistent \
    rd.live.overlay.overlayfs=1 \
    modprobe.blacklist=hp_bioscfg
    initrd /boot/initramfs_squash_sda1-x86_64.img_0311
}
EOF

# 4) Verify the entry exists in custom.cfg and not in grub.cfg
sudo rg -n "Gentoo Dracut \\(persist on nvme0n1p5 0311\\)" /0p3/boot/grub/custom.cfg /0p3/boot/grub/grub.cfg
```

Notes:
- This setup uses stock dracut behavior.
- `modprobe.blacklist=hp_bioscfg` is included to avoid a kernel issue seen on this platform.

`hp_bioscfg` is HP's BIOS configuration driver (sysfs interface under `/sys/class/firmware_attributes/hp-bioscfg/`). If BIOS configuration from Linux is not needed, blacklisting is acceptable.

After editing, validate syntax if available:

```bash
sudo grub-script-check /0p3/boot/grub/grub.cfg
```

On minimal live environments this tool may be absent. In that case, do a careful visual check and keep your backup file for rollback.

## 7. Prepare required overlay directories on persistent volume

Dracut overlay mode expects specific directories on the unlocked persistent filesystem.

Mount and create them:

```bash
sudo mount "/dev/mapper/$CRYPT_NAME" /mnt
sudo mkdir -p /mnt/persistent/upper
sudo mkdir -p /mnt/persistent/work
```

If `enc` is already open and mounted (for example at `/run/enc`), you can use that mountpoint directly:

```bash
sudo mkdir -p /run/enc/persistent/upper
sudo mkdir -p /run/enc/persistent/work
```

Meaning:
- `upper`: writable OverlayFS layer
- `work`: OverlayFS work directory (must be on same filesystem as `upper`)

## 8. Seed user/account data into overlay (optional but practical)

If you want existing users/passwords/SSH config available immediately, pre-seed files into `upper`:

```bash
sudo mkdir -p /mnt/persistent/upper/etc
sudo cp -av /etc/shadow /mnt/persistent/upper/etc/shadow
sudo cp -av /etc/passwd /mnt/persistent/upper/etc/passwd
sudo cp -av /etc/group /mnt/persistent/upper/etc/group

sudo mkdir -p /mnt/persistent/upper/home/kiel
sudo chown -R kiel:kiel /mnt/persistent/upper/home/kiel
sudo cp -av /home/kiel/.ssh /mnt/persistent/upper/home/kiel
sudo cp -av /home/kiel/.xinitrc /mnt/persistent/upper/home/kiel
sudo cp -av /home/kiel/.bashrc /mnt/persistent/upper/home/kiel
```

Finally unmount:

```bash
sudo umount /mnt
```

## 9. Reboot and test

1. Reboot and select `Gentoo Dracut (persist on nvme0n1p5 0309)` from GRUB.
2. Unlock LUKS when prompted.
3. Verify persistence by creating a test file and rebooting again.
4. Keep older `_0306` artifacts for rollback until the new boot path is validated.

## 10. Audio configuration (PipeWire, HP Z6 specific)

Short orientation (why this matters on both HP Z6 and E14):
- `ALSA` is the kernel/user-space hardware layer (`snd_*` drivers + `libasound`): it exposes the real sound cards/devices.
- Direct ALSA access can be enough for simple/single-app scenarios; with multiple desktop apps at once, a sound server is typically needed for safe mixing/routing.
- `PipeWire` is the session audio/video server on top of ALSA: routing, per-app volumes, device switching, low-latency graph.
- `PulseAudio` was the older userspace audio server for similar desktop tasks. In this image, the real PulseAudio daemon is replaced by PipeWire.
- `pipewire-pulse` is a compatibility server that provides a PulseAudio-compatible socket/API.
- Apps like Chrome usually use the PulseAudio client API; here they connect to `pipewire-pulse` and are handled by PipeWire internally.
- `pavucontrol` still works as before because it configures PulseAudio-compatible streams/devices, which are backed by PipeWire here.
- `dbus` is important for session integration (service discovery, policy/session coordination, RTKit interaction). With `-dbus`, device/session behavior can degrade or appear incomplete.

Can PulseAudio be removed completely?
- The separate package `media-sound/pulseaudio-daemon` can be removed (and was removed here).
- The PulseAudio compatibility interface in PipeWire (`pipewire-pulse`, `pipewire[pulseaudio]`) should usually stay enabled, otherwise apps expecting PulseAudio (including Chrome in typical Linux builds) may lose audio integration.
- There is no common "native PipeWire audio API path" used by Chrome for normal playback; PipeWire in Chrome is mainly used for features like screen capture, not as the primary playback API.

Validated on this HP Z6 image (`2026-03-09`):
- ALSA cards: `HDA NVidia` (GA104 HDMI/DP audio) and `HD-Audio Generic` (ALC222 analog codec)
- Working default output/input after fix:
  - Sink: `alsa_output.pci-0000_05_00.7.analog-stereo`
  - Source: `alsa_input.pci-0000_05_00.7.analog-stereo`

### 10.1 Critical fix: PipeWire USE flags

If `wpctl status` only shows `Dummy Output` or no sources, check PipeWire build flags first.

Problem observed on this machine:
- `media-video/pipewire` was built with `-dbus -pipewire-alsa -pulseaudio -sound-server`
- Result: ALSA devices existed in kernel, but were not exposed as usable PipeWire sinks/sources

Required `/etc/portage/package.use/package.use` entries:

```text
media-video/pipewire X dbus pipewire-alsa pulseaudio readline sound-server ssl systemd -system-service
media-video/wireplumber systemd -system-service
```

Rebuild:

```bash
sudo emerge -1v media-video/pipewire media-video/wireplumber
```

Note:
- `media-sound/pulseaudio-daemon` may be removed due to PipeWire `sound-server` soft block. This is expected in this setup.

### 10.2 Enable and start user services

```bash
systemctl --user enable --now pipewire pipewire-pulse wireplumber
systemctl --user --no-pager status pipewire pipewire-pulse wireplumber
```

### 10.3 Verify devices and set defaults

```bash
wpctl status
```

On this host, working IDs were:

```bash
wpctl set-default 52
wpctl set-default 53
wpctl set-volume @DEFAULT_AUDIO_SINK@ 0.40
wpctl set-mute @DEFAULT_AUDIO_SINK@ 0
wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.70
wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0
```

Final verification:

```bash
wpctl status
```

Expected in `Settings -> Default Configured Devices`:
- `Audio/Sink -> alsa_output.pci-0000_05_00.7.analog-stereo`
- `Audio/Source -> alsa_input.pci-0000_05_00.7.analog-stereo`

### 10.4 Low-level diagnostics used on this machine

Kernel/driver layer:

```bash
cat /proc/asound/cards
lsmod | rg '^snd|^sof|^audio'
lspci -nnk | rg -i "audio|multimedia|nvidia" -n -A4
```

PipeWire internal objects:

```bash
pw-cli ls Device
pw-cli ls Node
```

If `aplay`/`arecord` are missing, install `media-sound/alsa-utils` first.

### 10.5 Optional warning cleanup

The following warning is harmless for basic playback/recording but appeared in this image:
- `RTKit error: org.freedesktop.DBus.Error.ServiceUnknown`

Optional fix:

```bash
sudo emerge -1v sys-auth/rtkit
```
