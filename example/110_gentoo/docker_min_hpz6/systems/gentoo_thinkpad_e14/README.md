# Gentoo System Information - ThinkPad E14 (0308 Update)

This directory contains system snapshots of the Gentoo installation on the ThinkPad E14 workstation, refreshed on **2026-03-10** (Build `0308`).

## Latest Snapshot (0308 — 2026-03-10)

- **Summary:** several runtime and hardware snapshot files were refreshed on 2026-03-10 to capture current power, NVMe, and system-state metrics.
- **Refreshed snapshots:** `cpuinfo.txt`, `df.txt`, `dmesg.txt`, `dmidecode.txt`, `fastfetch.txt`, `fdisk.txt`, `free.txt`, `hostnamectl.txt`, `lsblk_all.txt`, `lsblk_devices.txt`, `lscpu.txt`, `lshw.txt`, `lsmod.txt`, `meminfo.txt`, `sensors.txt`, `smartctl_nvme0n1.txt`, `smartctl_nvme1n1.txt`, `tlp-stat.txt`, `uname.txt`.

- **NVMe safety notes:** the SMART counters show that unsafe shutdown counts have increased slightly since the 0306 snapshot (details below).

## Health delta since 0306

- `nvme0n1` (ADATA LEGEND 800, 1TB): Unsafe Shutdowns increased from **465** (0306) to **471** (0308) — +6.
- `nvme1n1` (SKHynix 512GB): Unsafe Shutdowns increased from **88** (0306) to **94** (0308) — +6.

Conclusion: unsafe shutdown occurrences have not reduced since the 0306 snapshot; they have in fact increased slightly in the latest 0308 refresh.

## Version Summary

- **Kernel**: `6.12.74-gentoo` (Build date: 2026-03-05)
- **Base Build**: `gentoo-z6-min_20260306`
- **Boot Method**: Dracut Live (SquashFS) with LUKS persistence on `nvme0n1p4`.

## Key Changes (0226 -> 0306)

### 1. Kernel and Build
- Upgraded kernel from `6.12.58` to `6.12.74`.
- Updated compiler/toolchain signatures in `dmesg` (Gentoo 16.0.1 experimental).

### 2. Boot Configuration
- Boot artifacts now use the `_0306` suffix:
    - `vmlinuz_0306`
    - `initramfs_squash_sda1-x86_64.img_0306`
    - `gentoo.squashfs_0306`
- Active command line updated to reference these artifacts.

### 3. Storage and Memory
- **Image Size**: The SquashFS image size increased significantly (~1.4G -> ~2.7G), as reflected in `df.txt` and `lsblk`.
- **Rootfs Usage**: `/run/rootfsbase` (the mounted squashfs) now consumes 2.7G of RAM (when `rd.live.ram=1` is used).
- **Persistent Storage**: `nvme0n1p4` (decrypted to `enc`) remains the primary persistent layer.

### 4. Hardware and Runtime
- **Battery**: Significant health drop noted since Jan snapshot (97% -> 90.9%). Thresholds are active (75/80).
- **NVMe Anomalies**:
    - `nvme0n1` (1TB) suffers from **Unsafe Shutdowns** due to DRAM-less architecture.
    - `nvme1n1` (512GB) shows **pathological power cycling** (23k+ cycles) due to aggressive APST.
    - Both addressed via `nvme_core.default_ps_max_latency_us=0` kernel parameter.
- **CPU**: `cpuinfo.txt` reflects active frequency scaling during the snapshot.
- **Modules**: Minor adjustments in loaded kernel modules.

## Snapshot Files
- `blkid.txt`: Partition UUIDs and labels.
- `cpuinfo.txt` / `lscpu.txt`: Processor details (AMD Ryzen 7 7735HS).
- `df.txt` / `lsblk_all.txt`: Filesystem and block device status.
- `dmesg.txt`: Kernel ring buffer after successful boot.
- `tlp-stat.txt`: Power management and battery health status.
- `uname.txt`: Current kernel signature.
