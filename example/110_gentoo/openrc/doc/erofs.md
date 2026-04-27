# EROFS Support and Usage

This document describes the EROFS support added for the ThinkPad E14 build, how it differs from the manually edited kernel config at `/mnt/usr/src/linux/.config`, and how to use the resulting image.

## Kernel Config Comparison

The repo kernel config is [config/config6.18.18](/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/config/config6.18.18). The manual reference config is `/mnt/usr/src/linux/.config`.

Both configs enable the DAX prerequisites already used by this project:

- `CONFIG_ZONE_DEVICE=y`
- `CONFIG_DAX=y`
- `CONFIG_DEV_DAX=y`
- `CONFIG_DEV_DAX_PMEM=y`
- `CONFIG_DEV_DAX_KMEM=y`
- `CONFIG_FS_IOMAP=y`
- `CONFIG_FS_DAX=y`
- `CONFIG_FS_DAX_PMD=y`

Both configs also enable the core EROFS options needed for basic read-only mounting:

- `CONFIG_EROFS_FS=y`
- `CONFIG_EROFS_FS_XATTR=y`
- `CONFIG_EROFS_FS_POSIX_ACL=y`

The manual config at `/mnt/usr/src/linux/.config` enables these additional EROFS options that are not yet mirrored in the repo config:

- `CONFIG_EROFS_FS_SECURITY=y`
- `CONFIG_EROFS_FS_BACKED_BY_FILE=y`
- `CONFIG_EROFS_FS_ZIP=y`
- `CONFIG_EROFS_FS_ZIP_ZSTD=y`

It also explicitly leaves these off:

- `# CONFIG_EROFS_FS_DEBUG is not set`
- `# CONFIG_EROFS_FS_ZIP_LZMA is not set`
- `# CONFIG_EROFS_FS_ZIP_DEFLATE is not set`
- `# CONFIG_EROFS_FS_ZIP_ACCEL is not set`
- `# CONFIG_EROFS_FS_ONDEMAND is not set`
- `# CONFIG_EROFS_FS_PCPU_KTHREAD is not set`

## Current Build Output

The Docker build now creates an additional E14 artifact:

- `/gentoo.squashfs_e14`: the existing trimmed E14 squashfs image
- `/gentoo.erofs_e14`: a new EROFS image for binaries, libraries, and `/opt`

The EROFS image is created from:

- `/bin`
- `/sbin`
- `/lib`
- `/lib64`
- `/usr/bin`
- `/usr/sbin`
- `/usr/lib`
- `/usr/lib64`
- `/opt`

This keeps the image focused on code and shared objects that benefit from direct executable mapping.

## Why This Image Is Uncompressed

`/gentoo.erofs_e14` is built uncompressed with:

```sh
mkfs.erofs -b 4096 /gentoo.erofs_e14 /tmp/erofs_e14_root
```

That choice is deliberate:

- DAX on EROFS applies to uncompressed images.
- 4 KiB blocks match the normal page size expected by file DAX.
- This is the closest fit for execute-in-place style access to binaries and libraries.

## Important DAX Caveat

EROFS `dax=always` only works when the underlying block device supports DAX.

That means:

- Mounting an EROFS file through a normal loop device on ordinary SSD storage usually will not give true DAX/XIP behavior.
- The initramfs helper falls back to plain read-only mounting if the DAX mount fails.
- The image is still usable without DAX; it just loses the execute-in-place benefit.

## Initramfs Support

The initramfs overlay helper in [config/mount-overlayfs.sh](/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/config/mount-overlayfs.sh:1) now supports an EROFS lower layer.

New kernel command line arguments:

- `rd.live.overlay.lower.erofsdev=<device>`
- `rd.live.overlay.lower.erofsopts=<mount-options>`

Default EROFS mount options are:

```text
ro,dax=always
```

If that mount fails, the script retries with:

```text
ro
```

The helper prefers EROFS over the ext4 lower image if both are specified.

## Boot Usage

The root still comes from the normal live squashfs flow. The EROFS image is an additional lower layer intended to sit ahead of `/run/rootfsbase` in the overlay stack.

Typical pattern:

1. Put `gentoo.erofs_e14` somewhere the initramfs can access as a block device or DAX-capable filesystem-backed device.
2. Keep the existing live squashfs boot arguments.
3. Add the EROFS lower-image argument so the overlay script mounts it before the squashfs root.

Example kernel command line fragment:

```text
rd.live.overlay.overlayfs=1 \
rd.live.overlay.lower.erofsdev=/dev/pmem0p1 \
rd.live.overlay.lower.erofsopts=ro,dax=always
```

If you are still using the older ext4 lower image instead, the existing argument remains:

```text
rd.live.overlay.lower.ext4dev=<device>
```

## Build-Time Requirements

The build now requires:

- `sys-fs/erofs-utils`
- kernel EROFS support
- dracut support for the `erofs` filesystem

These are wired in here:

- [config/world](/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/config/world:79)
- [Dockerfile](/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/Dockerfile:279)
- [Dockerfile](/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/Dockerfile:357)

## Notes

- The repo config currently supports the uncompressed EROFS image that the Dockerfile builds.
- Your manual kernel config enables extra features for compressed or file-backed EROFS use cases too.
- If you want the repo config to match the manual config exactly, the next step would be to copy over the extra `CONFIG_EROFS_*` settings listed above.
