# Optimization Log: GCC 15, LLVM, and Shaderc Pruning (2026-03-11)

## Context
The `gentoo-z6-min` Docker image build was taking too long and producing a large image. Analysis of the dependency tree revealed several heavy dependency chains that were not strictly necessary for the target use case (minimal Nvidia workstation).

## Identified Heavyweights
Using `emerge --pretend --tree`, we identified the following major contributors to build time and size:
1.  **GCC 15**: The system was attempting to upgrade to `sys-devel/gcc-15.0.0` (unstable/experimental), triggering a full toolchain rebuild.
2.  **LLVM & Clang**: Pulled in by `media-libs/mesa[llvm]` (default) and `dev-lang/rust` (source).
3.  **Shaderc**: A Google shader compiler, pulling in heavy C++ build artifacts. Pulled in by `media-libs/libplacebo[vulkan]` -> `media-video/mpv[vulkan]`.
4.  **Rust (Source)**: Needed for `librsvg` (GTK icons) and `firefox` (source).

## Actions Taken

### 1. Blocking Unwanted Upgrades
-   **GCC 15**: Created `config/package.mask` to block `>=sys-devel/gcc-15.0.0`. This forces the system to stay on the stable GCC 14.
-   **LLVM/Clang**: Added hard masks for `sys-devel/llvm` and `sys-devel/clang` to prevent them from slipping back in.

### 2. Pruning Graphics Dependencies
-   **Mesa**: Disabled `llvm` support in `config/package.use` (`media-libs/mesa -llvm`).
    -   *Impact*: Disables software rasterizers (llvmpipe) and some AMD drivers (radeonsi).
    -   *Mitigation*: We are using `x11-drivers/nvidia-drivers`, which provides its own OpenGL implementation.
-   **MPV & Libplacebo**: Disabled `vulkan` and `shaderc`.
    -   `media-video/mpv -vulkan`
    -   `media-libs/libplacebo -vulkan -shaderc -opengl`
    -   *Impact*: Removes Vulkan video output.
    -   *Mitigation*: MPV works fine with `vo=gpu` (OpenGL) or `vo=xv` on Nvidia.

### 3. Optimization of Tools
-   **QEMU**:
    -   Removed: `spice`, `vnc`, `smartcard`, `curl`.
    -   Kept: `usbredir` (requested feature).
    -   Result: `app-emulation/qemu` is now much smaller and builds faster.
-   **Firefox**: Switched to `www-client/firefox-bin` to avoid building Firefox from source (which requires Rust source, Clang, Node.js, etc.).

## Verification Results
-   **Total Packages**: Reduced to ~425.
-   **Build Time**: Significantly reduced by skipping LLVM, Clang, Shaderc, and GCC 15 builds.
-   **Largest Packages (Size)**:
    1.  `linux-firmware` (~570MB)
    2.  `nvidia-drivers` (~420MB)
    3.  `rust-bin` (~180MB) - *Build dependency only, removed in final stage.*
    4.  `gentoo-sources` (~150MB)

## Validation Commands
To verify the dependency tree in the future:
```bash
./scripts/analyze_deps.sh
python3 scripts/find_largest_pkgs.py
```
To test the base build:
```bash
./build_gcc_only.sh
```

## Files Modified
-   `config/package.mask`: Added masks.
-   `config/package.use`: Updated flags for mesa, mpv, libplacebo, qemu.
-   `Dockerfile`: Updated to copy `package.mask`.
