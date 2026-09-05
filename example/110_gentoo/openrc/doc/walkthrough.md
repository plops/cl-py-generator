# HP Z6 OpenRC Squashfs Deployment Walkthrough

This walkthrough explains how to deploy a dated Gentoo OpenRC squashfs bundle
to the HP Z6 without overwriting the running system. The concrete example is
the August 26, 2026 deployment from
`/dev/shm/gentoo-z6-min-openrc_20260826` into `/boot/0826`.

For the historical record, exact hashes, and older deployments, see
[install_on_hpz6.md](install_on_hpz6.md).

## Storage Layout

The boot artifacts and GRUB configuration live on the Btrfs filesystem with
UUID `4f708c84-185d-437b-a03a-7a565f598a23`. While a live image is running,
that filesystem is mounted at:

```text
/run/initramfs/live
```

The persistent overlay is the LUKS partition with UUID
`0d7c5e23-6bab-4dce-b744-a5d61d497aca`.

NVMe kernel names can change between boots. Use these stable references when
identifying the disks:

```text
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part5
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part6
```

Do not deploy into the running root's ordinary `/boot`. Use
`/run/initramfs/live/boot`.

## 1. Define the New Deployment

For the August deployment:

```bash
SOURCE_DATE=20260826
SLOT=0826
ARTIFACT_DIR=/dev/shm/gentoo-z6-min-openrc_20260826
TARGET_DIR=/run/initramfs/live/boot/0826
```

Keep the date and slot explicit and inspect their values before using them:

```bash
printf 'source date: %s\nslot: %s\nsource: %s\ntarget: %s\n' \
  "$SOURCE_DATE" "$SLOT" "$ARTIFACT_DIR" "$TARGET_DIR"
```

The HP Z6 artifact is `gentoo.squashfs_nv`. Do not accidentally install the
ThinkPad E14 artifact, `gentoo.squashfs_e14`.

## 2. Verify the Mount and Running Slot

Inspect the artifact filesystem and stable device mapping:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
readlink -f \
  /dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3
lsblk -f
```

Read the running slot from the kernel command line:

```bash
cat /proc/cmdline
```

Look for both `BOOT_IMAGE=/boot/<slot>/vmlinuz` and
`rd.live.dir=/boot/<slot>`. During the August installation both identified
`0625`. A previously expected value of `0624` was not trusted over the running
kernel's evidence.

Never delete:

- The slot named by `/proc/cmdline`.
- A slot the operator explicitly asked to preserve.
- The direct-disk fallback.
- Every older known-good squashfs; retain at least one fallback.

For the August installation, both `0624` and `0625` were treated as protected,
even though no `0624` directory existed.

## 3. Inspect and Identify the Source Bundle

Require the five deployment artifacts:

```bash
ls -lah "$ARTIFACT_DIR"
stat -c '%n %s bytes %y' \
  "$ARTIFACT_DIR/gentoo.squashfs_nv" \
  "$ARTIFACT_DIR/vmlinuz" \
  "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img" \
  "$ARTIFACT_DIR/packages.txt" \
  "$ARTIFACT_DIR/packages.tsv"
```

Describe the image and kernel:

```bash
file "$ARTIFACT_DIR/gentoo.squashfs_nv"
file "$ARTIFACT_DIR/vmlinuz"
strings "$ARTIFACT_DIR/vmlinuz" | grep -m 2 'Linux version'
```

Compute source hashes before copying:

```bash
sha256sum \
  "$ARTIFACT_DIR/gentoo.squashfs_nv" \
  "$ARTIFACT_DIR/vmlinuz" \
  "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img" \
  "$ARTIFACT_DIR/packages.txt" \
  "$ARTIFACT_DIR/packages.tsv"
```

When the container image still exists, its history can distinguish similar
build projects:

```bash
sudo docker image inspect gentoo-z6-min-openrc:latest \
  --format 'created={{.Created}} id={{.Id}}'
sudo docker history --no-trunc --format '{{.CreatedBy}}' \
  gentoo-z6-min-openrc:latest | grep -E \
  'reverse-ssh|config6|KVER_PURE|gentoo-sources'
```

For `0826`, direct `COPY config/reverse-ssh-*.initd` layers confirmed that the
image came from:

```text
/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc
```

The installed kernel is `6.18.41-gentoo-dist`, built from
`sys-kernel/gentoo-sources-6.18.41` with GCC 15.3.0 and GNU ld 2.46.1.

## 4. Check Space Before Writing

Compare free space with the source bundle and list the dated slots:

```bash
df -h /run/initramfs/live /dev/shm
du -sh /run/initramfs/live/boot/* 2>/dev/null | sort -h
du -sh "$ARTIFACT_DIR"
```

The August deployment began with 2.1 GiB free. The new NV squashfs was
2,398,760,960 bytes, so cleanup was required before copying.

The oldest dated squashfs slot was `0427`, about 2.2 GiB. It was not running,
was not operator-protected, and newer fallbacks `0428`, `0608`, and `0625`
remained. Its contents were listed before deletion:

```bash
find /run/initramfs/live/boot/0427 -mindepth 1 -maxdepth 1 \
  -printf '%y %f %s bytes\n' | sort
```

## 5. Remount, Back Up GRUB, and Free Space

The live artifact filesystem normally starts read-only. Temporarily remount it
read-write and back up GRUB before changing either slots or menu entries:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-0826
```

Delete only the previously inspected, unprotected old slot. For the August
deployment this was `/run/initramfs/live/boot/0427`. Verify that it is absent,
sync, and check the recovered space before continuing:

```bash
sudo rm -rf -- /run/initramfs/live/boot/0427
sync
test ! -e /run/initramfs/live/boot/0427
df -h /run/initramfs/live
```

This increased available space to about 4.2 GiB. The deletion was direct, not
a move to trash; recovery requires the old source artifacts or another backup.
Remove the matching GRUB menu entry when installing the new configuration so
the menu does not point to missing files.

## 6. Copy the New Artifacts

Require a new target rather than silently merging with a partial deployment:

```bash
test ! -e "$TARGET_DIR"
sudo mkdir "$TARGET_DIR"
```

Copy and rename the HP Z6 squashfs while preserving the other filenames:

```bash
sudo cp -av "$ARTIFACT_DIR/gentoo.squashfs_nv" \
  "$TARGET_DIR/gentoo.squashfs"
sudo cp -av "$ARTIFACT_DIR/vmlinuz" "$TARGET_DIR/vmlinuz"
sudo cp -av "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img" \
  "$TARGET_DIR/initramfs_squash_sda1-x86_64.img"
sudo cp -av "$ARTIFACT_DIR/packages.txt" "$TARGET_DIR/packages.txt"
sudo cp -av "$ARTIFACT_DIR/packages.tsv" "$TARGET_DIR/packages.tsv"
sync
```

## 7. Add `SOURCE.txt`

Every new dated directory should contain a human-readable `SOURCE.txt`. Record:

- Installation date, source bundle, and target slot.
- Build-project path and repository commit.
- Whether relevant build files had uncommitted modifications.
- Container tag, image ID, and creation time when available.
- Why `gentoo.squashfs_nv` was selected.
- Gentoo stage3, Portage snapshot, and profile.
- Kernel source version, release, configuration source, compiler, and linker.
- The role of the accompanying initramfs.
- SHA-256 hashes using the installed filenames.

The deployed example is:

```text
/run/initramfs/live/boot/0826/SOURCE.txt
```

Including the dirty-build warning matters: a Git commit cannot reproduce an
image if `Dockerfile` or kernel configuration changes were not committed.

## 8. Add the GRUB Entry

Add the new entry before older entries in
`/run/initramfs/live/boot/grub/custom.cfg`:

```grub
menuentry 'Gentoo Dracut (persist on luks overlay 0826 OpenRC NV folder)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/0826/vmlinuz \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/boot/0826 \
      rd.live.squashimg=gentoo.squashfs \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      pcie_aspm=off acpi_mask_gpe=0x08 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/amd-uc.img /boot/0826/initramfs_squash_sda1-x86_64.img
}
```

The directory in `linux`, `rd.live.dir`, and `initrd` must match. Keep
`rd.live.squashimg=gentoo.squashfs` aligned with the installed filename. The
AMD microcode image must precede the dated initramfs.

## 9. Verify Before Returning to Read-Only

List the installed files and compare every source/target hash pair:

```bash
ls -lah "$TARGET_DIR"
sha256sum "$TARGET_DIR/gentoo.squashfs" \
  "$ARTIFACT_DIR/gentoo.squashfs_nv"
sha256sum "$TARGET_DIR/vmlinuz" "$ARTIFACT_DIR/vmlinuz"
sha256sum "$TARGET_DIR/initramfs_squash_sda1-x86_64.img" \
  "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img"
sha256sum "$TARGET_DIR/packages.txt" "$ARTIFACT_DIR/packages.txt"
sha256sum "$TARGET_DIR/packages.tsv" "$ARTIFACT_DIR/packages.tsv"
```

Each pair must have identical hashes. Check the new entry, retained fallbacks,
and backup:

```bash
grep -nE 'menuentry.*(0826|0625|0608|0428)|rd.live.dir=/boot/(0826|0625|0608|0428)' \
  /run/initramfs/live/boot/grub/custom.cfg
test -d /run/initramfs/live/boot/0625
test -f /run/initramfs/live/boot/grub/custom.cfg.before-0826
```

If `grub-script-check` is installed, run it against `custom.cfg`. It was not
installed during the August deployment, so the entry was checked by inspecting
its structure and verifying every referenced file.

Sync, restore read-only mode, and confirm it:

```bash
sync
sudo mount -o remount,ro /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
df -h /run/initramfs/live
```

The final August state had about 1.9 GiB free. `df` displayed 100% utilization
because of rounding, but all artifacts had already been copied and verified.

## 10. Reboot and Validate

On the next reboot select:

```text
Gentoo Dracut (persist on luks overlay 0826 OpenRC NV folder)
```

After login, confirm that the new slot supplied the kernel and live root:

```bash
cat /proc/cmdline
uname -r
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

Expected values include:

```text
BOOT_IMAGE=/boot/0826/vmlinuz
rd.live.dir=/boot/0826
6.18.41-gentoo-dist
```

Do not remove `0625` merely because `0826` copied successfully. Keep it until
the new image has booted and its networking, overlay, graphics, and required
services have been validated.

## Rollback

If `0826` fails to boot, select `0625` from GRUB. To remove only the new slot
and restore the previous menu after booting a fallback:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -rf -- /run/initramfs/live/boot/0826
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg.before-0826 \
  /run/initramfs/live/boot/grub/custom.cfg
sync
sudo mount -o remount,ro /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

The additional fallbacks are `0608`, `0428`, and the direct `Gentoo OpenRC
disk` entry.

## 0826 Post-Boot Overlay Audit (2026-08-26)

The `0826` deployment booted successfully. The LUKS volume was unlocked, the
user `kiel` logged in, and X11 was started after running:

```text
~/activate
sudo chmod a+rwx /dev/tty*
startx
```

`Handy_0.9.4_amd64.AppImage` successfully captured microphone input and
performed speech-to-text. This validates the application and an audio-input
path, but not the intended PipeWire/Pulse session setup: at audit time no
`pipewire`, `pipewire-pulse`, or `wireplumber` process was running, and
`pactl info` failed with `Connection refused`.

No remediation described below was performed during this audit.

### Why the new squashfs does not replace old programs

The running root is mounted as:

```text
lowerdir=/run/rootfsbase
upperdir=/run/overlayfs -> /run/enc/persistent/upper
workdir=/run/ovlwork    -> /run/enc/persistent/work
```

`/run/rootfsbase` is the read-only `0826` squashfs and the encrypted LUKS
directory is the OverlayFS upper layer. OverlayFS always chooses an upper
object when the same path is present in both layers. Installing a newer
squashfs therefore cannot supersede a binary, library, symlink, package
database record, home-directory file, or opaque directory already copied into
the persistent upper layer.

The merged Portage database currently exposes both the old LUKS record and the
new squashfs record for four upgraded applications:

| Package | LUKS upper | 0826 squashfs | Runtime result |
| --- | --- | --- | --- |
| `app-editors/emacs` | `30.2-r3` | `30.2-r6` | The old upper `/usr/bin/emacs-30` wins. |
| `media-video/ffmpeg` | `8.1.1` | `8.1.2` | Old upper `ffmpeg` and `ffprobe` win, along with some copied-up library links. |
| `app-text/mupdf` | `1.26.3` with `X` | `1.27.2` without `X` | Old upper `mutool`, `mupdf`, `mupdf-x11`, and `libmupdf.so` win. The squashfs has a newer `mutool` but no GUI executable. |
| `net-misc/freerdp` | `3.26.0` with `X` | `3.30.0` without `X` | The old upper `xfreerdp` executable loads newer 3.30 squashfs libraries and reports 3.30.0. This mixed installation is the highest-risk case. |

The directly observed differing executable overlaps were:

```text
/usr/bin/ctags-emacs-30
/usr/bin/ebrowse-emacs-30
/usr/bin/emacs-30
/usr/bin/emacsclient-emacs-30
/usr/bin/etags-emacs-30
/usr/bin/ffmpeg
/usr/bin/ffprobe
/usr/bin/mutool
```

`/usr/bin/xfreerdp` is also stale, although it is not a same-path overlap: the
0826 FreeRDP build omitted the executable because its `X` USE flag was off.
Do not simply remove the upper MuPDF or FreeRDP files before rebuilding the
squashfs with the required GUI USE flags, or those two GUI programs will
disappear.

Two same-version package records also occur in both layers:
`app-emacs/emacs-common-1.14` and `x11-libs/libXtst-1.2.5`. They are not an
observed version regression, but they demonstrate that Portage state is being
merged from two independently managed installations.

The LUKS upper contains another 61 package records that have no corresponding
package name in the squashfs. These are not candidates for automatic
replacement; they are local additions that must either remain persistent or
be selected explicitly for future images:

```text
app-arch: 7zip, brotli, deb2targz, dpkg, upx-bin, zip
app-containers: nvidia-container-toolkit
app-eselect: eselect-repository
app-misc: screen
app-text: qpdf
dev-cpp: ada, nlohmann_json, simdutf
dev-debug: gdb
dev-go: go-md2man
dev-lang: go, rust, rust-bin, rust-common
dev-libs: cJSON, jemalloc, simde, simdjson, uthash
dev-python: defusedxml
dev-qt: qtsvg
dev-util: difftastic, include-what-you-use, nvidia-cuda-toolkit, xxd
dev-vcs: git-lfs
kde-frameworks: extra-cmake-modules
llvm-core: clang, clang-common, clang-linker-config,
  clang-toolchain-symlinks
llvm-runtimes: clang-rtlib-config, clang-runtime, clang-stdlib-config,
  clang-unwindlib-config, compiler-rt, compiler-rt-sanitizers, openmp
media-libs: libsamplerate, libva, rnnoise, x264
media-sound: alsa-utils
media-video: obs-studio
net-dialup: picocom
net-libs: mbedtls, nodejs, rpcsvc-proto
sci-mathematics: cadical, lean
sys-libs: libnvidia-container
sys-process: numactl
www-client: httrack
x11-libs: libXaw3d, libXres
x11-misc: xcb
```

This list is based on `/var/db/pkg` records, not a dependency closure. Some are
support packages rather than user-facing programs, and some may now be
unneeded. Review them before adding them wholesale to `config/world`.

### Other squashfs state hidden by LUKS

The most consequential hidden directory is `/etc/runlevels/default`. Its LUKS
upper copy has `trusted.overlay.opaque=y`, so its contents replace rather than
merge with the squashfs directory. Only `cgroups`, `reverse-ssh-eu`, and
`reverse-ssh-us` are present there. These seven 0826 default-runlevel links are
therefore invisible:

```text
dbus
iwd
local
netmount
sshd
user-runtime
user.kiel
```

This explains much of the work currently done by `~/activate`. It also means
that adding `rc-update` commands to the Dockerfile alone will not repair an
existing shared upper layer.

The following user files are also old LUKS copies and mask the Dockerfile
versions:

| Visible path | LUKS copy | Squashfs copy | Important difference |
| --- | --- | --- | --- |
| `~/activate` | 4,370 bytes, modified May 11 | 1,877 bytes | The LUKS script contains Z6 network setup but points first at `/usr/local/share/e14-bringup`, from a different build project. |
| `~/.xinitrc` | 1,145 bytes, modified April 29 | 792 bytes | The LUKS version has the four-monitor NVIDIA layout and its own D-Bus launch, but does not start `~/start-pipewire.sh`. |
| `~/start2` | 133 bytes, modified April 28 | 2,005 bytes | The LUKS version merely redirects to `~/activate`; the image contains the newer split setup logic. |

These files contain machine/user policy, so treating the squashfs versions as
unconditional replacements would also discard deliberate local changes. They
need a versioned migration policy rather than silent overwrite.

The persistent upper also masks the squashfs copies of:

```text
/etc/portage/package.use/package.use
/etc/portage/package.accept_keywords/package.accept_keywords
/var/lib/portage/world
```

Consequently, package operations performed after boot use old persistent
configuration even though the squashfs was built from the current repository
files.

Running `env-update` and `ldconfig` from `~/activate` copied several generated
library symlinks into the upper layer at boot. This can create new shadowing
even when the library payload itself still comes from the squashfs. Generated
linker state should not be refreshed unconditionally into a long-lived upper
layer.

### TTY and graphical-session finding

After the broad `chmod`, `/dev/tty`, `/dev/tty0`, `/dev/tty1`, and `/dev/tty2`
were all mode `0777`. User `kiel` is already a member of group `tty`, and
`/dev/tty1` is owned by `kiel:tty`. `sys-auth/elogind` is installed but its
service is stopped, and the current default runlevel does not include it. The
successful Xorg log uses VT 7 and contains no permission error, but the chmod
changed the evidence before that log was written, so this audit cannot prove
which original device access failed.

Do not automate `chmod a+rwx /dev/tty*`: it grants every local process access
to every terminal. Fix seat/session ownership through elogind (or another
deliberately selected seat manager), PAM/OpenRC session setup, and normal udev
permissions. Add a boot test that starts X as `kiel` without changing device
modes.

### Recommended installation and image changes

1. **Stop sharing one operating-system upper layer across releases.** Add an
   explicit release/slot argument to `mount-overlayfs.sh`, for example a custom
   `rd.live.overlay.slot=0826`, and use separate
   `persistent/slots/0826/{upper,work}` directories. A new squashfs should boot
   with a fresh upper. Retain the previous slot's upper for rollback.

2. **Separate persistent data from persistent OS mutations.** Keep selected
   data such as `/home/kiel`, SSH material, Docker data, and other declared
   state in dedicated encrypted directories mounted or bound after the root is
   assembled. Do not use one unrestricted upper as both the package manager
   and the home/data partition. This is the structural fix that allows new
   squashfs files to become visible reliably.

3. **Provide an explicit, reversible migration step.** Before first boot of a
   new slot, generate a report of upper paths that collide with the new lower,
   back up the old upper, seed a fresh slot, and migrate only an allowlist. Do
   not edit an active upper while it is mounted as `/`. Record the source slot,
   destination slot, migrated paths, and hashes in the deployment directory.

4. **Fix the two incomplete GUI builds in `openrc/config/package.use`.** Add
   explicit `X` USE flags for at least `app-text/mupdf` and
   `net-misc/freerdp`, then make the Docker build fail unless the expected
   commands exist. Suitable checks include `test -x /usr/bin/mupdf-x11` and
   `test -x /usr/bin/xfreerdp`. Also run version smoke tests from the finished
   squashfs before export.

5. **Choose which LUKS-only applications belong in the immutable image.** Add
   frequently required programs to `config/world` and rebuild them with the
   squashfs. Keep large or experimental toolchains out unless their boot-to-
   boot availability is required. Avoid normal `emerge` upgrades into the
   shared upper; they recreate the mixed-package problem seen above.

6. **Move root boot work out of `~/activate`.** Split module loading, network
   setup, reverse tunnels, runtime-directory creation, and linker maintenance
   into small idempotent OpenRC services with dependencies. Enable the needed
   services, including the chosen elogind/session path, in the image. Keep
   user-specific display layout in a user-owned configuration file.

7. **Make the user session single-owner and automatic.** Decide whether D-Bus
   and PipeWire are managed by OpenRC user services or by `.xinitrc`, not both.
   The Dockerfile currently creates the D-Bus user link but leaves PipeWire,
   PipeWire Pulse, and WirePlumber links commented out. After validating those
   services, enable them and remove the ad-hoc process startup. Optionally add
   a console-login guard that invokes `startx` once on the intended VT.

8. **Treat home files as managed defaults.** Store `activate`, `start2`,
   `.xinitrc`, and audio/session defaults under a versioned
   `/usr/local/share/openrc-host-config` tree. On a release transition, compare
   the installed user copy, preserve local edits, and offer or apply a
   three-way migration. A Dockerfile `COPY` into `/home/kiel` cannot update a
   path already present in the persistent upper.

9. **Do not run `env-update`/`ldconfig` on every login.** Run them in the image
   build and, if necessary, once in a controlled release migration. Add an
   audit that reports generated upper symlinks and package files after first
   boot.

10. **Make export/deployment release-driven and atomic.** Replace independent
    `date` calls in `setup03_copy_from_container.sh` and `copy_files.sh` with a
    required release ID passed through both scripts. Export into a temporary
    directory, verify all required artifacts and hashes, write `SOURCE.txt`,
    then rename into place. Avoid `chmod -R a+rwx`; use explicit ownership and
    read modes. A deployment helper can then install the dated slot, update
    GRUB, verify references, and restore the live filesystem read-only in one
    checked workflow.

11. **Add an overlay-collision artifact to every build.** Alongside
    `packages.txt` and `packages.tsv`, produce a machine-readable manifest of
    managed executables, libraries, service links, and configuration hashes.
    A pre-reboot audit can compare that manifest with the selected LUKS upper
    and block deployment when stale binaries or opaque runlevel directories
    would hide the new image.

The first implementation priority should be per-slot upper directories plus a
small, explicit persistent-data allowlist. Dockerfile service and package
fixes are still necessary, but a shared unrestricted upper will continue to
mask those improvements on every later squashfs update.
