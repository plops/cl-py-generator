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

### Execute-in-Place (XIP)
Normal execution copies code from storage -> RAM (Page Cache) -> Execution. XIP runs code directly from its storage address, avoiding the copy.
*   **Requirement:** Requires uncompressed data on memory-mappable storage (NOR flash, DAX-capable RAM disk).
*   **Filesystems:** AXFS, CramFS, or ext4/XFS with `-o dax` on persistent memory.
*   **Note:** If you mount a SquashFS image into a RAM disk (`/dev/ram`), you are still "double loading" (RAM disk -> Page Cache) unless using a DAX/XIP-aware setup.
