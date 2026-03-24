# Post-Build Plan For Dual-Lower DAX/PMEM Boot Flow

This document is a handoff for the next agent after the Docker build finishes. It assumes no prior context.

## What Changed Before The Build

These changes are already present in the worktree and are intended to be consumed by the next build:

1. [`Dockerfile`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/Dockerfile)
   Uses:
   `gentoo/portage:20260323`
   `gentoo/stage3:nomultilib-20260323`

2. [`Dockerfile`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/Dockerfile)
   Before kernel build, the `CONFIG_OPTS` block now enables PMEM/NVDIMM-related kernel options:
   `X86_PMEM_LEGACY`
   `LIBNVDIMM`
   `BLK_DEV_PMEM`
   `BTT`
   `ND_BTT`
   `NVDIMM_PFN`
   `ND_PFN`
   `NVDIMM_DAX`
   `DEV_DAX`
   `DEV_DAX_PMEM`
   `ND_BLK`

3. [`Dockerfile`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/Dockerfile)
   The dracut command now includes:
   `--add-drivers "libnvdimm nd_pmem nd_btt nd_blk dax device_dax dax_pmem dax_pmem_core nd_e820"`

4. [`config/mount-overlayfs.sh`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/config/mount-overlayfs.sh)
   The initramfs overlay hook now supports an optional second lower layer from kernel args:
   `rd.live.overlay.lower.ext4dev=...`
   `rd.live.overlay.lower.ext4opts=...`

5. [`setup06_create_qemu_dual_lower.sh`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/setup06_create_qemu_dual_lower.sh)
   Creates a QEMU image for the new dual-lower layout and copies `gentoo.ext4` into `qemu/`.

6. [`setup07_run_qemu_dual_lower.sh`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/setup07_run_qemu_dual_lower.sh)
   Runs QEMU with the ext4 lower image attached as a second virtio disk.

## Goal After Build

Confirm that the rebuilt exported artifacts support:

1. Kernel-level PMEM/NVDIMM support.
2. Initramfs inclusion of PMEM/NVDIMM drivers.
3. Dual-lower overlay root:
   ext4 lower first
   squashfs lower second
   encrypted ext4 upper/work on `/dev/mapper/enc`
4. If possible, actual PMEM-backed DAX mount behavior in the guest.

## Expected Build Output Location

After a successful rebuild and export, artifacts should exist under a directory like:

`/dev/shm/gentoo-z6-min-openrc_YYYYMMDD/`

The key expected files are:

1. `gentoo.squashfs` or `gentoo.squashfs_e14`
2. `gentoo.ext4`
3. `vmlinuz`
4. `initramfs_squash_sda1-x86_64.img`
5. `packages.txt`

## Step 1: Run The Build And Export

Run:

```bash
./setup01_run_with_log.sh
./setup03_copy_from_container.sh
```

If the build fails, inspect the latest log under:

`logs/setup01_run_*.log`

Most likely failure points:

1. The stage3 or portage tag may not exist.
2. One or more kernel config symbols in `CONFIG_OPTS` may have been renamed or removed in `6.12.77`.
3. Dracut may fail if one of the requested drivers does not exist under that built kernel.

## Step 2: Confirm The Exported Kernel Config

Check the built kernel config inside the artifact tree or inside the built image if available.

Minimum required result:

```bash
zgrep -E 'CONFIG_(LIBNVDIMM|BLK_DEV_PMEM|FS_DAX|DAX|ZONE_DEVICE|X86_PMEM_LEGACY|DEV_DAX|DEV_DAX_PMEM|NVDIMM_DAX|ND_PFN|ND_BTT|ND_BLK)' /proc/config.gz
```

If checking from the built rootfs image instead of a running system, inspect the installed config from the built container or kernel build directory.

Success criteria:

1. `CONFIG_LIBNVDIMM` is enabled, ideally `y`.
2. `CONFIG_BLK_DEV_PMEM` is enabled, ideally `y`.
3. `CONFIG_DAX` and `CONFIG_FS_DAX` are enabled.
4. `CONFIG_X86_PMEM_LEGACY` is enabled if using `memmap=...!...` to create type-12 persistent memory.

If these are not present, the PMEM path is still blocked at kernel level.

## Step 3: Confirm The New Initramfs Contains The Required Hook And Drivers

Use the freshly exported initramfs from `/dev/shm/gentoo-z6-min-openrc_YYYYMMDD/initramfs_squash_sda1-x86_64.img`.

Check the overlay hook:

```bash
lsinitrd -f var/lib/dracut/hooks/pre-pivot/10-mount-overlayfs.sh /dev/shm/gentoo-z6-min-openrc_YYYYMMDD/initramfs_squash_sda1-x86_64.img
```

It must contain:

1. `LOWER_EXT4_DEV`
2. `LOWER_EXT4_OPTS`
3. mount of `/run/rootfsdax`
4. `ovlfs=lowerdir=/run/rootfsdax:/run/rootfsbase`

Check for PMEM/NVDIMM-related modules:

```bash
lsinitrd /dev/shm/gentoo-z6-min-openrc_YYYYMMDD/initramfs_squash_sda1-x86_64.img | \
  rg 'usr/lib/modules/.*/(libnvdimm|nd_pmem|nd_btt|nd_blk|dax|device_dax|dax_pmem|dax_pmem_core|nd_e820)\.ko'
```

Success criteria:

1. The modified overlay hook is present in the initramfs.
2. The PMEM/NVDIMM modules listed above are present.

If the hook is missing, the build did not pick up the updated `config/mount-overlayfs.sh`.
If the modules are missing, dracut may still be omitting them or the kernel may not have built them.

## Step 4: Create The Dual-Lower QEMU Image

Run:

```bash
sudo ARTIFACT_DIR=/dev/shm/gentoo-z6-min-openrc_YYYYMMDD ./setup06_create_qemu_dual_lower.sh
```

Expected outputs under `qemu/`:

1. `openrc-luks-dual-lower.img`
2. `gentoo.ext4`
3. `vmlinuz`
4. `initramfs_dual_lower_sda1-x86_64.img`
5. `cmdline_dual_lower.txt`

Check the generated cmdline:

```bash
cat qemu/cmdline_dual_lower.txt
```

It must contain:

1. `rd.live.overlay.overlayfs=1`
2. `rd.live.overlay.lower.ext4dev=/dev/vdb`
3. `rd.live.overlay.lower.ext4opts=ro,dax=always`
4. `rd.luks.uuid=...`
5. `rd.luks.name=...=enc`

## Step 5: Boot In QEMU

Run:

```bash
QEMU_DEBUG=1 QEMU_CONSOLE_MODE=serial ./setup07_run_qemu_dual_lower.sh
```

The script attaches:

1. `qemu/openrc-luks-dual-lower.img` as `vda`
2. `qemu/gentoo.ext4` as `vdb`

At boot, enter the LUKS passphrase:

`openrc-test`

## Step 6: Verify Guest Boot Behavior

Inside the boot log or guest shell, verify:

### Overlay lower stack

```bash
cat /proc/mounts | grep -E 'overlay|rootfsdax|rootfsbase|/run/enc'
```

Success criteria:

1. `/run/rootfsbase` is the squashfs lower.
2. `/run/rootfsdax` is mounted from the ext4 lower image.
3. `/sysroot` or `/` is an overlay with:
   `lowerdir=/run/rootfsdax:/run/rootfsbase`

### PMEM presence

If the PMEM path is really working, also check:

```bash
dmesg | grep -Ei 'pmem|nvdimm|dax'
ls /dev/pmem* /dev/dax* 2>/dev/null
```

Success criteria:

1. The kernel reports PMEM or NVDIMM initialization.
2. `/dev/pmem0` or similar exists if the memory reservation path is active.

If only `/dev/vdb` exists and no `/dev/pmem*` exists, then the system is only using ext4 as a second lower layer, not true PMEM-backed XIP.

### DAX mount behavior

Check:

```bash
cat /proc/mounts | grep rootfsdax
```

Desired result:

The ext4 lower is mounted with DAX-related options.

If mount fell back to plain `ro`, the hook’s fallback path was used. That means DAX mount failed but boot continued.

## Step 7: If True PMEM Is Still Missing, Add A QEMU PMEM Device

The current runner attaches `gentoo.ext4` as a readonly virtio block device on `vdb`.
That is enough to test the dual-lower overlay path, but may not be enough for real PMEM-backed DAX/XIP.

If `/dev/pmem*` does not appear after boot, modify [`setup07_run_qemu_dual_lower.sh`](/home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/setup07_run_qemu_dual_lower.sh) to use a QEMU NVDIMM/PMEM backend instead of a plain virtio disk.

The likely direction is:

1. Create an object backend from `qemu/gentoo.ext4`
2. Expose it via `-device nvdimm,...` or the matching QEMU PMEM-capable device
3. Adjust the guest cmdline if needed:
   add a `memmap=...!...` reservation only if the chosen PMEM mechanism requires it

Do not assume the current `vdb` path yields actual XIP.

## Step 8: Final Validation For XIP

Only do this if `/run/rootfsdax` is a true DAX-capable mount and `/dev/pmem*` exists.

Check:

```bash
filefrag -v /run/rootfsdax/usr/bin/bash
```

And optionally:

```bash
LD_LIBRARY_PATH=/run/rootfsdax/lib64:/run/rootfsdax/usr/lib64 /run/rootfsdax/usr/bin/bash
```

Interpretation:

1. If the binary can be executed directly from the DAX-backed mount and kernel logs indicate DAX/PMEM behavior, XIP testing can continue.
2. If the overlay root works but direct execution from the DAX mount is required, use `/run/rootfsdax/...` paths instead of overlay paths.

## Known Caveats

1. OverlayFS may hide or defeat some DAX/XIP properties. Direct execution from `/run/rootfsdax/...` is a more reliable test than execution through `/`.
2. The current initramfs hook intentionally falls back from `ro,dax=always` to plain `ro` so boot can proceed.
3. The exact QEMU PMEM plumbing has not yet been implemented in the runner.
4. The Docker build has not been rerun after these edits, so none of the new PMEM assumptions are validated yet.

## Minimal Command Sequence For The Next Agent

```bash
./setup01_run_with_log.sh
./setup03_copy_from_container.sh

ls -ltrh /dev/shm/gentoo-z6-min-openrc_$(date +%Y%m%d)/

lsinitrd -f var/lib/dracut/hooks/pre-pivot/10-mount-overlayfs.sh \
  /dev/shm/gentoo-z6-min-openrc_YYYYMMDD/initramfs_squash_sda1-x86_64.img

lsinitrd /dev/shm/gentoo-z6-min-openrc_YYYYMMDD/initramfs_squash_sda1-x86_64.img | \
  rg 'usr/lib/modules/.*/(libnvdimm|nd_pmem|nd_btt|nd_blk|dax|device_dax|dax_pmem|dax_pmem_core|nd_e820)\.ko'

sudo ARTIFACT_DIR=/dev/shm/gentoo-z6-min-openrc_YYYYMMDD ./setup06_create_qemu_dual_lower.sh
QEMU_DEBUG=1 QEMU_CONSOLE_MODE=serial ./setup07_run_qemu_dual_lower.sh
```
