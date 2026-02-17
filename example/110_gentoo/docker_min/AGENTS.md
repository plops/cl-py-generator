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

### ⚠️ MANDATORY: Ask User About Build Mode Before Adding Packages

**ALWAYS ask the user which mode they want before adding new packages:**

#### Core Principle: ALWAYS Split fetchonly and emerge

**For @world builds, ALWAYS keep separate:**
```dockerfile
# Download all packages first (cached layer - won't re-download if build fails)
RUN with-steve.sh emerge -e @world --fetchonly

# Rebuild world with steve jobserver coordination (separate layer for debugging)
RUN with-steve.sh emerge -e @world
```
This is **NOT negotiable** - never merge these two steps!

#### 🐛 Debugging Mode (Default for Active Development)

**Use when:**
- Testing new packages
- Iterating on configurations
- Daily development work
- Package might not work as expected

**How to add packages:**
Add new packages as **late-stage RUN commands** (before mksquashfs):

```dockerfile
# ... existing world build ...
RUN with-steve.sh emerge --depclean

# Build kernel with steve jobserver coordination
RUN with-steve.sh make -j32

# Install kernel modules and kernel
RUN make modules_install && make install

# === NEW PACKAGE (DEBUG MODE) ===
# Testing: app-editors/vim with python support
RUN emerge --ask=n app-editors/vim
# ================================

# Configure sudo and create user with password
RUN mkdir -p /etc/sudoers.d && \
# ... rest of Dockerfile ...
```

**Why this works:**
- ✅ Uses cached world build (fast iteration)
- ✅ Only rebuilds from new package onwards
- ✅ No modification to config/world (no early rebuild)
- ✅ Can test package without committing to world
- ✅ Easy to remove if package doesn't work

**Location:** Add new package installations **after kernel build, before mksquashfs**

#### 🚀 Consolidation Mode (Weekly Maintenance)

**Use when:**
- Weekly stage3 update time
- Package has been tested and works well
- Moving tested packages to permanent world set
- End-of-week maintenance

**How to consolidate packages:**
Move tested packages from late-stage RUN to `config/world`:

**Before (Debug Mode):**
```dockerfile
# Late in Dockerfile
RUN emerge --ask=n app-editors/vim
RUN emerge --ask=n dev-vcs/git
```

**After (Consolidation Mode):**
```bash
# Edit config/world file
app-editors/vim
dev-vcs/git
```

```dockerfile
# Remove late-stage RUN commands
# (packages now built during world rebuild)
```

**Result:** Packages are now part of the stable world set and built early in the process.

#### Decision Tree for Package Addition

**Choose Debugging Mode if:**
- "I want to test this package first"
- "Not sure if this package will work"
- "Need quick iteration to test configuration"
- "Daily development work"
- "Package might need USE flag tweaks"

**Choose Consolidation Mode if:**
- "It's the weekly stage3 update"
- "Package has been working for several days"
- "Moving all tested packages to world"
- "End-of-week cleanup"

#### Package Addition Workflow

**Daily (Debug Mode):**
1. User: "Add package X"
2. Agent asks: "Debug mode (late-stage RUN) or Consolidation mode (config/world)?"
3. If debug: Add `RUN emerge --ask=n X` before mksquashfs
4. Build only rebuilds from that point (fast!)

**Weekly (Consolidation Mode):**
1. Review all late-stage package additions
2. Add tested packages to `config/world`
3. Remove late-stage RUN commands
4. Full rebuild with updated stage3 (scheduled anyway)

#### Example Conversation

**User:** "Add vim to the build"

**Agent:** "Before I add vim, which mode do you prefer?

🐛 **DEBUGGING MODE** (recommended for testing)
   - Add as late-stage RUN command (before mksquashfs)
   - Fast rebuild (only from vim installation onwards)
   - Easy to remove if it doesn't work
   - Location: After kernel build, before user setup

🚀 **CONSOLIDATION MODE** (for weekly updates)
   - Add to config/world file
   - Slow rebuild (from world rebuild stage)
   - Package becomes part of stable world set
   - Use during weekly stage3 updates

**Recommendation:** Debugging mode unless it's weekly maintenance time."

#### Consolidation Checklist (Weekly)

**When updating stage3 (weekly):**
1. ✅ Update FROM lines (stage3 and portage dates)
2. ✅ Review all late-stage RUN package additions
3. ✅ Move tested packages to `config/world`
4. ✅ Remove corresponding late-stage RUN commands
5. ✅ Update package.use if needed
6. ✅ Full rebuild (cache invalidated by FROM change anyway)

**Keep as late-stage RUN:**
- Packages still being tested
- Experimental packages
- Packages you might remove soon

#### Visual Workflow: Package Addition Locations

```
DOCKERFILE STRUCTURE:
┌─────────────────────────────────────────────────────┐
│ FROM gentoo/stage3:20260216                         │ ← Changes weekly
├─────────────────────────────────────────────────────┤
│ COPY config/world                                   │ ← 🚀 CONSOLIDATION
│ (contains: sudo, btrfs-progs, cryptsetup, ...)     │    Add here weekly
├─────────────────────────────────────────────────────┤
│ Install GCC 16                                      │
├─────────────────────────────────────────────────────┤
│ Install kernel sources                              │
├─────────────────────────────────────────────────────┤
│ Configure kernel                                    │
├─────────────────────────────────────────────────────┤
│ RUN emerge -e @world --fetchonly                   │ ← NEVER merge!
│ RUN emerge -e @world                                │ ← NEVER merge!
│ RUN emerge --depclean                               │
├─────────────────────────────────────────────────────┤
│ Build kernel                                        │
├─────────────────────────────────────────────────────┤
│ Install kernel modules                              │
├─────────────────────────────────────────────────────┤
│ 🐛 DEBUG MODE: Add new packages HERE               │ ← Fast iteration!
│ RUN emerge --ask=n app-editors/vim                 │   Uses cached world
│ RUN emerge --ask=n dev-vcs/git                     │   Only rebuilds from
│ RUN emerge --ask=n sys-apps/htop                   │   here downwards
├─────────────────────────────────────────────────────┤
│ Configure sudo and create user                      │
├─────────────────────────────────────────────────────┤
│ Generate file list                                  │
├─────────────────────────────────────────────────────┤
│ Create squashfs                                     │
├─────────────────────────────────────────────────────┤
│ Create initramfs                                    │
└─────────────────────────────────────────────────────┘

WORKFLOW:
Day 1-6: Add packages in 🐛 debug zone (fast, iterative)
Day 7:   Move tested packages to 🚀 config/world (weekly maintenance)
```

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
- **Tools**: steve (jobserver)
- **Python**: 3.14 (single target)

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
