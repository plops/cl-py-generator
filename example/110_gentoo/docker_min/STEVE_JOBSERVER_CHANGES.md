# Steve Jobserver Integration for 16-core/32-thread Build Server

This document describes the changes made to integrate steve jobserver for coordinated parallel builds across all 16 cores and 32 threads.

## Overview

Steve is a system-wide jobserver implementation that coordinates parallel job execution across multiple build processes. It prevents overscheduling by managing a shared pool of job tokens that build tools (make, ninja, cargo, gcc, etc.) can acquire before starting work.

## Changes Made

### 1. Kernel Configuration (Dockerfile)

**Enabled CONFIG_CUSE** in the kernel configuration:
- CUSE (Character Device in Userspace) is required for steve to create the `/dev/steve` character device
- Added `CUSE` to the list of enabled kernel modules

### 2. Steve Installation (Dockerfile)

**Installed steve package**:
```bash
RUN emerge --ask=n dev-build/steve
```

**Created wrapper script** `/usr/local/bin/with-steve.sh`:
- Automatically starts steve daemon with 32 jobs (16 cores × 2 threads)
- Configured with load-average limit of 32
- Manages daemon lifecycle per build command
- Cleans up properly even if build fails

### 3. Make Configuration (config/make.conf)

**Updated parallel build settings**:
```bash
# Keep -j32 for packages that don't support jobserver
MAKEOPTS="-j32 -l32"
NINJAOPTS="-l32"

# Configure jobserver for make - NO -j flag so it uses the jobserver
MAKEFLAGS="-l32 --jobserver-auth=fifo:/dev/steve"
```

**Key points**:
- `MAKEOPTS` keeps `-j32` for tools that extract the value but don't support jobserver
- `MAKEFLAGS` configures jobserver without `-j` flag (critical - `-j` disables jobserver)
- Load average limits set to 32 to match thread count

**Enabled Portage jobserver integration**:
```bash
FEATURES="noman nodoc noinfo jobserver-token"
```

This enables Portage >=3.0.74 to acquire jobserver tokens per emerge job, solving the "implicit slot" problem where each make process gets one free job slot.

### 4. Build Commands (Dockerfile)

**Wrapped intensive build operations with steve**:
```bash
RUN with-steve.sh emerge -e @world --fetchonly
RUN with-steve.sh emerge -e @world
RUN with-steve.sh emerge --depclean
RUN with-steve.sh make -j32
```

## How It Works

1. **Token Pool**: Steve maintains a pool of 32 job tokens
2. **Token Acquisition**: Build tools acquire a token before starting a job
3. **Token Release**: Tokens are returned when jobs complete (or processes exit)
4. **Dynamic Coordination**: Multiple parallel emerges share the same 32-token pool
5. **Load Management**: Additional load-average limiting prevents thrashing

## Benefits

1. **No Overscheduling**: Maximum 32 jobs system-wide, not per-package
2. **Better Resource Utilization**: Idle threads pick up work from the shared pool
3. **Improved Interactivity**: Parallel emerges don't spawn 100+ processes
4. **Memory Management**: Fewer concurrent C++ compilers = less memory pressure
5. **Cross-tool Coordination**: Works with make, ninja, cargo, gcc LTO, etc.

## Supported Tools

The jobserver protocol is supported by:
- GNU Make (original implementation)
- Ninja build system
- Cargo (Rust package manager)
- GCC (LTO parallelism)
- LLVM (LTO parallelism)

## Build Server Specifications

- **CPU**: 16 cores / 32 threads
- **Token Pool**: 32 tokens (matches thread count)
- **Load Average Limit**: 32 (matches thread count)
- **Parallel Emerges**: 24 (from EMERGE_DEFAULT_OPTS)

## References

- [Gentoo Wiki: steve](https://wiki.gentoo.org/wiki/Steve)
- [One jobserver to rule them all](https://blogs.gentoo.org/mgorny/2025/11/30/one-jobserver-to-rule-them-all/)
- [GNU Make Jobserver](https://www.gnu.org/software/make/manual/html_node/Job-Slots.html)

