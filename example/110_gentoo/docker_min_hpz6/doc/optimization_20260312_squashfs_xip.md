# Optimization and Build Stats (2026-03-12)

## Build Performance
- **Build Duration:** 3933s (~1h 5m)
- **Start Time:** 2026-03-11T22:09:27Z
- **Package Count:** 384
- **SquashFS Size:** 1.9GB (33.89% compression ratio)

## SquashFS Compression vs. XIP (Execute-in-Place)

### Compression Trade-offs
SquashFS is read-only and compressed.
*   **Compressed (Default):** Saves disk space and I/O bandwidth (good for slow USB/SD). Costs CPU cycles to decompress on read.
*   **Uncompressed:** Removes CPU decompression overhead but increases I/O load. Faster on high-speed storage (NVMe) where CPU is the bottleneck.

### Execute-in-Place (XIP) via DAX on RAM Disk
This solves the "Double Caching" problem where a SquashFS image loaded into RAM (`rd.live.ram=1`) is cached again in the Page Cache when files are read.

**The Strategy: brd + ext4 + DAX**
Instead of mounting a SquashFS image, create a RAM-backed block device (`brd`), format it with a DAX-capable filesystem (`ext4`), and synchronize files into it during boot.

1.  **Initialize RAM Disk:** Use `brd` kernel module to create a raw memory block device.
2.  **Format with DAX:** Format as `ext4` with block size matching page size (usually 4K).
3.  **Mount with `-o dax`:** Tells kernel to map memory addresses directly, bypassing page cache.
4.  **Populate:** Copy files from SquashFS (on physical disk) into this DAX RAM-disk.

#### Kernel Requirements
*   `CONFIG_BLK_DEV_RAM` (for brd)
*   `CONFIG_FS_DAX`
*   `CONFIG_EXT4_FS` + `CONFIG_EXT4_FS_POSIX_ACL`

#### Implementation Logic (Dracut Module)
Intercept the boot process before the overlay is mounted to manually handle memory allocation.

```bash
# 1. Create a RAM disk large enough for uncompressed rootfs (e.g., 4GB)
# brd is better than /dev/shm for DAX support
modprobe brd rd_nr=1 rd_size=4194304

# 2. Format the RAM disk for DAX
mkfs.ext4 -b 4096 /dev/ram0

# 3. Mount it with DAX enabled
mkdir -p /run/rootfs_dax
mount -o dax,noatime /dev/ram0 /run/rootfs_dax

# 4. Mount your SquashFS from the NVMe (temporary)
mkdir -p /run/squash_source
mount -t squashfs /run/initramfs/live/gentoo.squashfs /run/squash_source

# 5. Sync files (This replaces rd.live.ram=1)
# This is the "loading" phase.
cp -a /run/squash_source/* /run/rootfs_dax/

# 6. Clean up source
umount /run/squash_source
```

#### Overlay Configuration
Change the `lowerdir` to point to the new DAX-enabled mount point:

```bash
# Build overlay lowerdir using the DAX mount
ovlfs=lowerdir=/run/rootfs_dax
mount -t overlay LiveOS_rootfs -o "$ROOTFLAGS,$ovlfs",upperdir=/run/overlayfs,workdir=/run/ovlwork "$NEWROOT"
```

#### Mixed Compression Strategy (Nested Overlays)
To save RAM, keep large, rarely-used data (firmware, locales) compressed on NVMe while keeping binaries uncompressed for XIP.
Use multiple `lowerdir` paths in OverlayFS:

1.  **Lower 1 (DAX):** `ext4`-DAX RAM disk containing binaries (`/usr/bin`, `/usr/lib`).
2.  **Lower 2 (Compressed):** SquashFS mounted directly from NVMe (not copied to RAM).
3.  **Upper:** Persistent encrypted layer.

```bash
# Example Overlay Mount
mount -t overlay -o lowerdir=/run/rootfs_dax:/run/squash_compressed ...
```

#### RAM Usage Comparison
*   **Current way:** SquashFS-in-RAM (2GB) + Page Cache (2GB) = **4GB used**.
*   **DAX way:** Ext4-in-RAM (Uncompressed, say 3.5GB) + 0MB Page Cache = **3.5GB used**.
*   **Result:** You "waste" RAM on the uncompressed filesystem, but you never pay the "double tax" when programs actually execute.

