# OpenRC E14 Bringup Notes

## Scope
- This directory contains the OpenRC-based Gentoo live image bringup for the ThinkPad E14.
- Installation and bringup documentation belongs in `doc/install_on_e14.md` and related `doc/install*e14*` files.
- Runtime hotfixes for the already-booted image belong in `config/activate`, which is copied into the image as `/activate` and may also be mirrored manually to `~/kiel/activate` on the live system.

## Working Rules
- Prefer fixing current boot/runtime regressions before broad cleanup.
- Do not hand-edit generated or exported artifacts on the target machine when the repo copy is the real source of truth.
- Treat the running live image and the Docker build as separate states:
  - `config/activate` is for immediate rescue on the live system.
  - `Dockerfile`, `config/world`, and `config/package.use` are for making the next rebuilt image correct by default.

## Current Bringup Focus
- OpenRC service activation on first boot.
- E14-specific graphics, WiFi, input, and audio bringup.
- Dynamic linker/runtime issues after toolchain changes.

## Known Runtime Issue
- If `emacs`, `rg`, AppImages, or other binaries fail with `libgcc_s.so.1` or `libstdc++.so.6` errors, first suspect a stale dynamic linker configuration rather than a missing package.
- Check:
  - `find /usr/lib /usr/lib64 -name 'libgcc_s.so*' -o -name 'libstdc++.so*' 2>/dev/null`
  - `ldconfig -p | grep 'libgcc_s\\.so\\.1\\|libstdc++\\.so\\.6'`
  - `cat /etc/ld.so.conf.d/05gcc-x86_64-pc-linux-gnu.conf`
- If the cache points at an old GCC slot, refresh it with `env-update && ldconfig`.

## Tooling Notes
- On a broken live image, `rg` itself may fail because it also depends on `libgcc_s.so.1`.
- In that state, use `find`, `grep`, `sed`, and `awk` until the loader cache is repaired.

## Validation
- For live-system hotfixes, validate on the running machine with the smallest possible checks first, for example:
  - `ldconfig -p | grep libgcc_s`
  - `emacs --version`
  - `rg --version`
- For rebuild fixes, keep the E14 install doc in sync with the actual recovery steps.
