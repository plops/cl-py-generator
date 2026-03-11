# Copilot Instructions for `example/110_gentoo/docker_min_hpz6`

> **Note:** These instructions apply **only** to the `example/110_gentoo/docker_min_hpz6` directory. Ignore all other parts of the repository.

---

## Build, Test, and Lint Commands

- **Full build workflow:**
  - `./build_and_run.sh` — Runs the complete build, extraction, QEMU image creation, and boots in QEMU.
  - Stepwise scripts:
    - `./setup01_build_image.sh` — Builds the Docker image (accepts SSH key args for remote unlock).
    - `./setup03_copy_from_container.sh` — Extracts build artifacts from the container to the host.
    - `./setup04_create_qemu.sh` — Creates a raw disk image and installs GRUB.
    - `./setup05_run_qemu.sh` — Boots the image in QEMU.
- **Audit world package sizes:**
  - `./scripts/audit_world_sizes.sh [world-file]` — Lists largest packages in the world set (requires `qsize`).
- **No formal test or lint suite** is present; validation is by successful image boot and artifact inspection.

## High-Level Architecture

- **Purpose:** Builds a minimal, reproducible Gentoo Linux image (squashfs + kernel + initramfs) for ThinkPad E14 and HP Z6, using Docker for isolation and repeatability.
- **Artifacts:**
  - `gentoo.squashfs` (root FS), `vmlinuz` (kernel), `initramfs_*.img` (initramfs variants), `packages.txt` (installed packages)
  - Artifacts are exported to `/dev/shm/gentoo-z6-min_YYYYMMDD/` on the host
- **Config:**
  - All build and system config in `config/` (make.conf, world, package.use, kernel config, etc.)
  - Hardware inventory and helper scripts in `systems/`
- **Initramfs:**
  - Two main variants: RAM-copy and disk-mount, both using encrypted overlayfs for persistence
  - Optional remote-unlock (SSH) initramfs is present but not production-ready
- **QEMU:**
  - Disk image creation and boot scripts for local testing

## Key Conventions & Patterns

- **Dockerfile iteration:**
  - Temporary changes are appended to the end of the Dockerfile for fast iteration; clean up before final builds
- **Kernel module selection:**
  - Use `lsmod` from a modern live OS to guide kernel config (see `config/config6.*` and scripts)
- **Partitioning & GRUB:**
  - Kernel and GRUB config live on unencrypted partitions; squashfs and persistence are encrypted and referenced by UUID
  - Example GRUB entries in `grub.txt` and install docs
- **Hardware support:**
  - `systems/` contains scripts and logs for hardware probing and anonymization
- **Build args:**
  - `INITRAMFS_AUTH_KEY` and `TUNNEL_KEY` can be passed to inject SSH keys for remote unlock
- **Image size management:**
  - Large packages are commented out in `config/world` for laptop builds; full set for workstation

## Documentation

- See `doc/README.md`, `doc/spec.md`, and install guides in `doc/` for detailed workflows, hardware notes, and troubleshooting.
- For remote GUI streaming, see `doc/remote_gui.md`.

---

**This file is for Copilot and other AI assistants. If you add new scripts, workflows, or conventions, update this file to keep Copilot effective.**
