# Gentoo Minimal Docker Build Configuration

## Overview
This Docker configuration builds a minimal, debloated Gentoo system that produces a squashfs image designed to run from RAM. The final output is a bootable system image optimized for size and performance.

## Quick Start
```bash
cd /home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min
./setup01_build_image.sh
```
Output: `gentoo.squashfs` in the Docker container (extracted via setup03_copy_from_container.sh)

## Project Structure

### Key Files
- **Dockerfile** - Multi-stage build configuration (316 lines, optimized for caching)
- **config/** - Configuration files copied into the image
  - `make.conf` - Portage build configuration (includes steve jobserver settings)
  - `package.use` - USE flag configurations
  - `package.accept_keywords` - Package keyword acceptance (~amd64)
  - `package.license` - License acceptance
  - `package.mask.nosystemd` - Mask systemd (use systemd-utils instead)
  - `package.use.ssl` - SSL/TLS related USE flags
  - `world` - List of packages to install
  - `dwm-6.5` - DWM window manager saved config
  - `simple_password.expect` - Password setup script
  - `list_files.py` - Generates exclusion list for squashfs
  - `mount-overlayfs.sh` - Dracut module for overlayfs mounting
  - `52persistent-overlay/` - Dracut module directory

### Build Scripts
- `setup01_build_image.sh` - Build the Docker image
- `setup02_run_container.sh` - Run container from built image
- `setup03_copy_from_container.sh` - Extract squashfs from container
- `setup04_create_qemu.sh` - Create QEMU disk image
- `setup05_run_qemu.sh` - Run system in QEMU
- `build_and_run.sh` - Combined build and run script

## Build System Features

### Steve Jobserver Integration
The build uses **steve** jobserver for coordinated parallel builds:
- **32 job tokens** (16 cores × 2 threads)
- **Load average limit**: 32
- **Coordination**: Prevents overscheduling across parallel emerges
- **CUSE kernel module**: Required for /dev/steve character device
- **Wrapper script**: `/usr/local/bin/with-steve.sh` manages daemon lifecycle

### Build Configuration (make.conf)
```bash
MAKEOPTS="-j32 -l32"                              # For non-jobserver tools
NINJAOPTS="-l32"                                   # Ninja load average
MAKEFLAGS="-l32 --jobserver-auth=fifo:/dev/steve" # Jobserver (NO -j flag!)
EMERGE_DEFAULT_OPTS="--jobs 24 --load-average 32" # Parallel emerges
FEATURES="jobserver-token"                         # Portage integration
```

### Compiler Configuration
- **GCC 16** (latest) with znver3 optimization
- **COMMON_FLAGS**: `-march=znver3 -fomit-frame-pointer -O2 -pipe`
- **CPU_FLAGS_X86**: Optimized for Threadripper 7955WX and Ryzen 7735HS

## Critical Development Guidelines

### ⚠️ Docker Layer Caching Strategy
**NEVER merge `--fetchonly` with build commands!**

❌ **WRONG** (re-downloads on build failure):
```dockerfile
RUN emerge @world --fetchonly && emerge @world
```

✅ **CORRECT** (cached downloads persist):
```dockerfile
RUN emerge @world --fetchonly
RUN emerge @world
```

### Dockerfile Optimization Principles

1. **Order by Frequency of Change** (top to bottom):
   - Base image selection
   - System configuration (rarely changes)
   - Package configuration (changes occasionally)
   - Build operations (changes during development)
   - Final image creation (rarely changes)

2. **Separate Long-Running Operations**:
   ```dockerfile
   RUN emerge @world --fetchonly   # Downloads (cached)
   RUN emerge @world               # Build (can fail/retry)
   RUN emerge --depclean           # Cleanup (separate)
   ```

3. **Group by Logical Units**:
   - Initial configuration → single RUN
   - Kernel configuration → single RUN
   - Each major build step → separate RUN

4. **Keep Download Steps Separate**:
   - Downloads in their own RUN command
   - Never combine with compilation
   - Allows retry without re-download

5. **Use Comments for Layer Purpose**:
   ```dockerfile
   # Download all packages first (cached layer - won't re-download if build fails)
   RUN with-steve.sh emerge -e @world --fetchonly
   ```

### When to Modify Early vs Late Stages

**Early Stages** (modify rarely, causes full rebuild):
- Base image FROM lines
- Initial COPY of config files
- System profile selection
- GCC installation

**Late Stages** (modify freely during development):
- User creation
- Final package installations
- Squashfs creation
- Initramfs generation

**Middle Stages** (modify carefully):
- World updates - separate fetchonly from build
- Kernel builds - separate compilation from installation

### Consolidation Strategy

**Weekly**: When stage3 image updates
- Combine RUN commands for operations that always succeed together
- Remove temporary files in same layer they're created
- Merge related configuration steps

**Keep Separate**:
- Any operation that might fail (builds, compilations)
- Download operations (--fetchonly)
- Operations you're actively debugging
- Cleanup steps that could mask errors

## Build Process Overview

### Stage 1: Base System (Fast, Cached)
1. Copy portage tree
2. Configure system profile
3. Install GCC 16
4. Install kernel sources

### Stage 2: Kernel Configuration (Fast)
1. Configure kernel (disable unnecessary drivers)
2. Enable required features (CUSE for steve!)
3. Configure modules

### Stage 3: World Rebuild (SLOW - 40-60 min)
1. **Download packages** (separate layer!)
2. **Rebuild @world** (can retry without re-download)
3. **Cleanup** (separate for debugging)

### Stage 4: Kernel Build (Medium - 3-5 min)
1. **Compile kernel** (with steve coordination)
2. **Install modules and kernel** (separate layer)

### Stage 5: Finalization (Fast)
1. User setup
2. File list generation
3. Squashfs creation
4. Initramfs generation

## Build Time Expectations

- **Full build**: ~40-70 minutes (depending on cache)
- **World rebuild**: ~40-60 minutes
- **Kernel build**: ~3-5 minutes
- **GCC installation**: ~5-10 minutes
- **Squashfs creation**: ~2-3 minutes

## Goals

### Primary Goals
- ✅ **Debloat**: No systemd (use systemd-utils for udev)
- ✅ **Minimal**: Only essential packages
- ✅ **Fast builds**: Steve jobserver coordination
- ✅ **Reproducible**: Version-pinned base images

### Development Workflow Goals
- ✅ **Fast iteration**: Separate downloads from builds
- ✅ **Cache-friendly**: Order by change frequency
- ✅ **Debug-friendly**: Separate logical steps
- ✅ **Retry-friendly**: Don't lose downloads on build failure

### Size Goals
- **Compressed**: ~2-3 GB squashfs (zstd level 19)
- **Uncompressed**: ~6-7 GB
- **Compression ratio**: ~30-35%

## Troubleshooting

### Build Fails During World Rebuild
- Downloads are cached in previous layer
- Fix the issue, rebuild from that point
- No re-download needed!

### Build Fails During Kernel Compilation
- Kernel compilation is separate from installation
- Adjust config, rebuild from compilation step
- Installation layer is separate for quick testing

### Need to Add a Package
- Modify `config/world` file
- Rebuild will start from world fetchonly layer
- Downloads cached, only new packages fetched

## Package Management

### Current Package Set
- **Base system**: @system + gentoolkit, eix
- **Compiler**: GCC 16
- **Kernel**: gentoo-sources 6.12.58
- **Tools**: steve (jobserver), expect (password setup)
- **Python**: 3.13 (single target)

### Adding Packages
1. Edit `config/world`
2. Optionally update `config/package.use` for USE flags
3. Rebuild from world fetchonly layer

### Removing Packages
1. Edit `config/world`
2. Rebuild will handle depclean automatically

## Notes

- **Stage3 updates**: Weekly (gentoo/stage3:nomultilib-YYYYMMDD)
- **Portage snapshot**: Weekly (gentoo/portage:YYYYMMDD)
- **Architecture**: x86_64 no-multilib only
- **Profile**: default/linux/amd64/23.0/no-multilib
