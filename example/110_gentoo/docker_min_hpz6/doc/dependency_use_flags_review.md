# Dependency USE Flag Review (Top Packages)

Date: 2026-03-06
Source list: `doc/dependency_audit.md` (largest installed world packages)
Context: review of installed USE flags and `config/package.use` intent

## Summary

This review focuses on the top packages by installed size and whether current
USE choices are aligned with expected features, runtime behavior, and startup
time.

## Package Review

### sys-kernel/linux-firmware (~1725 MiB)

- Current USE: `initramfs redistributable`
- Optional IUSE of interest: `compress-zstd`, `compress-xz`, `deduplicate`
- Impact:
  - Turning on compression/dedup can reduce installed footprint.
  - Runtime/boot impact is usually small; may add minor decompression overhead.

### sys-kernel/gentoo-sources (~1410 MiB)

- Current USE: no special feature flags enabled.
- Optional IUSE of interest: `symlink`, `experimental`, `build`
- Impact:
  - Mostly kernel source management/build workflow.
  - No direct userland runtime/load-time impact.

### net-im/signal-desktop-bin (~442 MiB)

- Current USE: no meaningful feature toggles (binary package).
- Impact:
  - Size/features are mostly fixed by upstream binary.

### sys-devel/gcc (~357 MiB)

- Current USE includes: `fortran`, `graphite`, `jit`, `openmp`, `sanitize`, `cet`
- Impact:
  - `jit`, `fortran`, `openmp`, `sanitize` increase compiler capability and size.
  - Mostly affects build/toolchain performance and available language/runtime
    features for software you compile, not end-user app startup directly.

### www-client/google-chrome (~339 MiB)

- Current USE: mostly locale (`l10n_en-GB`), no major feature toggles in use.
- Optional IUSE of interest: `qt6`, `selinux`.
- Impact:
  - Limited Gentoo-side control for binary package.

### app-editors/emacs (~318 MiB)

- Current USE includes: `gui X Xaw3d athena jit dbus systemd sqlite tree-sitter`
- Largest size drivers:
  - bundled lisp tree
  - native-compiled lisp (`jit`)
- Impact:
  - `jit` improves runtime responsiveness at cost of package size/build time.
  - `tree-sitter` improves language parsing/highlighting quality.
  - `sqlite` enables sqlite-backed functionality in packages/features.

### dev-lang/rust (~266 MiB)

- Current USE includes: `clippy rustfmt system-llvm llvm_targets_X86 llvm_targets_AMDGPU`
- Impact:
  - `clippy`/`rustfmt` add dev tooling (size increase, no runtime benefit).
  - extra LLVM targets increase build/toolchain footprint.
  - mostly impacts developer workflow, not runtime of unrelated apps.

### www-client/firefox (~243 MiB)

- Current USE includes:
  - `X dbus hwaccel pulseaudio openh264 jumbo-build`
  - `system-av1 system-harfbuzz system-icu system-jpeg system-libevent system-libvpx system-pipewire system-webp`
  - disabled: `wayland`, `pgo`, `telemetry`, `wifi`, `wasm-sandbox`
- Impact:
  - `hwaccel` and `system-pipewire` improve media/runtime behavior.
  - `jumbo-build` helps build time; minimal runtime effect.
  - `-pgo` keeps build simpler but can leave some performance on the table.
  - `-wayland` forces X11 path; if you move to Wayland, enable `wayland`.

### net-analyzer/wireshark (~115 MiB)

- Current USE includes:
  - `gui`, `lua`, `lua_single_target_lua5-4`
  - `http2`, `brotli`, `snappy`, `zlib`, `zstd`, `ssl`
  - `plugins`, `pcap`, `sharkd`, `sdjournal`, `sshdump`, `wifi`
- Requirement check:
  - GUI required: satisfied (`gui` enabled)
  - Parse gRPC traffic (compressed): satisfied (`http2`, `brotli`, `snappy`, `zlib`, `zstd`)
  - Encrypted traffic support preferred: satisfied (`ssl` enabled)
  - Custom Lua dissectors: satisfied (`lua` + lua target enabled)
- Notes:
  - For TLS decryption in practice, capture/export session keys is still needed;
    USE flags alone do not decrypt unknown TLS sessions.
  - Optional: consider `http3` if you expect gRPC over QUIC/HTTP3.

### dev-python/uv (~62 MiB)

- Current USE: minimal; optional `test`/`debug` are off.
- Impact:
  - Current config is already lean.

### dev-db/postgresql (~53 MiB)

- Current USE includes: `server ssl icu llvm lz4 zstd readline systemd uring`
- Impact:
  - Good general-purpose server feature set.
  - Disabling `icu`/`llvm` can reduce footprint but may reduce capabilities
    (collation/JIT-related behavior).

### app-text/mupdf (~52 MiB)

- Current USE: `X ssl jpeg2k`, with `-javascript`, `-opengl`.
- Impact:
  - `-javascript` reduces attack surface.
  - `-opengl` keeps dependencies smaller/simpler.

### dev-lisp/sbcl (~48 MiB)

- Current USE: `threads unicode zstd`.
- Impact:
  - Reasonable defaults; no obvious bloat flags enabled.

### dev-vcs/git (~39 MiB)

- Current USE includes: `curl gpg nls pcre tk`, with `-perl`.
- Impact:
  - `-perl` reduces dependency size but removes some legacy scripts/features.
  - `tk` enables `gitk`; disable if not needed.

### net-analyzer/nmap (~24 MiB)

- Current USE includes: `nse ssl nls ipv6`.
- Impact:
  - `nse` is high value for extensibility.
  - Current profile is appropriate for troubleshooting/security workflows.

## Recommended Changes (Optional)

1. Keep Wireshark flags as-is for your stated requirements.
2. Consider enabling `wireshark http3` only if you need HTTP/3 or QUIC capture.
3. If size pressure is high, evaluate:
   - `app-editors/emacs -jit` (largest single Emacs size reduction),
   - `dev-lang/rust -clippy -rustfmt` (if host is not used for Rust development),
   - `dev-vcs/git -tk` (if `gitk` is unnecessary).
