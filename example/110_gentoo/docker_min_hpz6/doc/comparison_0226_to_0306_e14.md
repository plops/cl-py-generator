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

## Loaded Modules (`lsmod.txt`)

- Observed minor variations in module load order and resident memory usage for graphics and wireless drivers, matching the kernel upgrade.
