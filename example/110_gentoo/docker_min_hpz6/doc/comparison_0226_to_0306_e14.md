# Comparison Report: Gentoo ThinkPad E14 (0226 vs 0306)

This report documents the changes observed after updating the ThinkPad E14 Gentoo installation to the **0306** image.

## Summary of Software Updates

| Component | Version (0226) | Version (0306) | Change |
|-----------|----------------|----------------|--------|
| **Kernel** | 6.12.58-gentoo | 6.12.74-gentoo | Upgraded |
| **Toolchain** | GCC 16.0.1 (2026.02.22) | GCC 16.0.1 (2026.03.01) | Minor Update |
| **SquashFS Size** | 1.4 GB | 2.7 GB | **+1.3 GB** |

## Filesystem Status (`df.txt`)

The increased squashfs size has a direct impact on the live RAM usage:
- `/run/rootfsbase` increased from **1.4G** to **2.7G**.
- `/run/initramfs/live` (artifact storage) usage increased from **66%** to **81%** (12G to 15G used).
- Persistent storage (`/run/enc`) usage dropped slightly (**99%** -> **93%**), likely due to cleanup or optimization in the new image.

## Kernel Boot Log (`dmesg.txt`)

Significant boot entries and changes:
- **Command Line**: Successfully updated to `rd.live.squashimg=gentoo.squashfs_0306`.
- **Memory Maps**: Minor shifts in efi memory allocation ranges (`0x70aad018` vs `0x70aad018`).
- **Initialization**: Firmware attributes and EFI services initialized with identical hardware signatures.

## Battery Health Trends (Jan - Mar)

Analysis of the system's Git history (`cfa74177` -> `0306`) reveals steady degradation:

| Period | Cycle Count | Full Capacity | Capacity % |
|--------|-------------|---------------|------------|
| Early Jan | 67 | 55.3 Wh | 97.0% |
| Late Feb | 86 | 54.3 Wh | 95.3% |
| **Mar 06** | **95** | **51.8 Wh** | **90.9%** |

**Observation**: Despite using a 75%/80% charge threshold, the battery's full energy capacity is starting to decline more rapidly as it approaches 100 cycles.

## NVMe Drive Endurance and Health

The SMART data for both primary and secondary drives shows interesting activity since the last collection:

| Metric | nvme0n1 (1TB) | nvme1n1 (512GB) | Change (Approx) |
|--------|---------------|-----------------|-----------------|
| **Power On Hours** | 15,670 -> 15,763 | 2,626 -> 2,721 | +93-95 hours |
| **Power Cycles** | 2,143 -> 2,167 | 23,306 -> 23,330 | +24 cycles |
| **Data Written** | 20.7 TB -> 20.9 TB | 5.67 TB -> 5.70 TB | +230 GB total |
| **Unsafe Shutdowns** | 460 -> 465 | 80 -> 88 | +5 / +8 |
| **Temperature** | 31°C -> 47°C | 23°C -> 41°C | +16-18°C elevated |

**Notes**:
- **Unsafe Shutdowns**: Both drives are DRAM-less (ADATA Legend 800 and SK Hynix BC901) and are sensitive to the shutdown handshake. The ADATA drive has logged **33 unsafe shutdowns** since early January, correlating almost 1:1 with system power cycles.
- **Pathological Cycling on nvme1n1**: The secondary Hynix drive is cycling at an extreme rate of **8.6 cycles/hour** (~23,330 cycles total). This is likely due to aggressive APST (Autonomous Power State Transitions) firmware.
- **Recommendation**: The parameter `nvme_core.default_ps_max_latency_us=0` is critical for this machine to stop both the unsafe shutdowns and the excessive cycling on the secondary drive.

## Loaded Modules (`lsmod.txt`)

- Observed minor variations in module load order and resident memory usage for graphics and wireless drivers, matching the kernel upgrade.
