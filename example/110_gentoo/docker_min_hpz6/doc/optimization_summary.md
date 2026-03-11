# Gentoo Docker Image Optimization Summary

## Goal
Optimize the Gentoo Docker image for size and compilation time, specifically:
- Remove large compilation dependencies: LLVM, Mesa (source), Rust (source).
- Remove large runtime dependencies: Qt, NodeJS, detailed documentation toolchains (xmlto/fop).
- Maintain functionality: NVIDIA graphics with X11, MPV for media, a modern web browser.

## Optimization Strategy

### 1. Removing Source-Based Heavyweights
- **LLVM & Mesa:** Disabled `llvm` support in Mesa (`media-libs/mesa -llvm`). This removes the need to compile LLVM/Clang, saving massive build time. NVIDIA drivers provide their own OpenGL implementation, so software rasterizers (llvmpipe) are not needed.
- **Rust:** Switched to `dev-lang/rust-bin`. This avoids compiling the Rust compiler itself (which takes hours).
- **Firefox:** Switched to `www-client/firefox-bin`. Compiling modern browsers from source is resource-intensive.

### 2. Dependency pruning
We iteratively removed large packages and their dependencies:
- **Qt5/Qt6:** Removed by dropping `virt-manager`, `wireshark`, and `qemu` GUI tools.
- **NodeJS:** Removed entirely.
- **Bluetooth/NetworkManager:** Removed `bluez`, `blueman`, `networkmanager`. Using simpler networking.
- **PulseAudio Tools:** Replaced `pavucontrol` (GTK based) with `pulsemixer` (CLI based).
- **Documentation Tools:** Removed `app-text/xmlto`, which pulls in a heavy Java/DocBook toolchain. This required switching from Chrome (which needs `xdg-utils` -> `xmlto`) to Firefox.

### 3. Verification of "Hard" Dependencies
Some dependencies could not be removed due to the requirement for a graphical web browser (`firefox-bin`):
- **GTK+ 3:** Required by Firefox.
- **librsvg:** Required by GTK+ to load icons/assets.
- **Adwaita Icon Theme:** Required by GTK+ as a fallback cursor/icon theme.
- **Rust (Binary):** `librsvg` is written in Rust, so the binary Rust runtime is required.

## Key Learnings & Configuration
- **Mesa without LLVM:** Works fine for NVIDIA users. Add `media-libs/mesa -llvm` to `package.use`.
- **Rust Binary:** Essential for build speed. Add `dev-lang/rust-bin` to world and set `dev-lang/rust system-llvm` (if needed) or just rely on the bin.
- **Browser Choice:** `google-chrome` (Gentoo package) pulls in `xdg-utils`, which pulls in `xmlto`, which pulls in `fop`/`docbook`. `firefox-bin` is cleaner in this specific dependency graph.
- **Audio:** `pulsemixer` is a lightweight, ncurses-based alternative to `pavucontrol` that avoids GTK dependencies.

## Final Status
- **Compiler Stack:** GCC (source), Rust (bin), Clang (removed).
- **Graphics Stack:** NVIDIA (proprietary), X11 (source), Mesa (minimal, no llvm).
- **Desktop:** i3 (assumed), dmenu, simple terminal, no full DE.
- **Apps:** `firefox-bin`, `mpv`, `pulsemixer`.
