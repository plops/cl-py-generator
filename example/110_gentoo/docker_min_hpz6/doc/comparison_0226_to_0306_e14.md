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

## Hardware Health (`tlp-stat.txt` & `sensors.txt`)

- **Battery Aging**:
    - Cycle Count: 86 -> **95**
    - Full Charge Energy: 54300 mWh -> **54130 mWh**
    - Capacity: 95.3% -> **95.0%**
- **Thermal Status**: Idle temperatures remained consistent within expected ranges.

## NVMe Drive Endurance and Health

The SMART data for both primary and secondary drives shows the following activity over the last ~95 operational hours:

| Metric | nvme0n1 (1TB) | nvme1n1 (512GB) | Change (Approx) |
|--------|---------------|-----------------|-----------------|
| **Power On Hours** | 15,670 -> 15,763 | 2,626 -> 2,721 | +93-95 hours |
| **Power Cycles** | 2,143 -> 2,167 | 23,306 -> 23,330 | +24 cycles |
| **Data Written** | 20.7 TB -> 20.9 TB | 5.67 TB -> 5.70 TB | +230 GB total |
| **Unsafe Shutdowns** | 460 -> 465 | 80 -> 88 | +5 / +8 |
| **Temperature** | 31°C -> 47°C | 23°C -> 41°C | +16-18°C elevated |

**Notes**:
- The temperature elevation is consistent with sustained write activity during the SquashFS transfer and image deployment.
- **Unsafe Shutdowns**: Both drives are DRAM-less (ADATA Legend 800 and SK Hynix BC901) and are sensitive to the shutdown handshake. 
- **Recommendation**: Consider adding `nvme_core.default_ps_max_latency_us=0` to the GRUB command line to ensure the controller is always ready for the final shutdown signal.

## Loaded Modules (`lsmod.txt`)

- Observed minor variations in module load order and resident memory usage for graphics and wireless drivers, matching the kernel upgrade.
