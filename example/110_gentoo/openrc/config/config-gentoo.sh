#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GENTOO_DEP_TUI="${GENTOO_DEP_TUI:-${HOME}/Downloads/gentoo-dep-tui}"
REPO_DIR="${REPO_DIR:-/var/db/repos/gentoo}"
BINHOST_DIR="${BINHOST_DIR:-/tmp/Packages}"

exec "${GENTOO_DEP_TUI}" \
    --repo "${REPO_DIR}" \
    --portage-dir "${SCRIPT_DIR}/etc/portage" \
    --world "${SCRIPT_DIR}/var/lib/portage/world" \
    --binhost "${BINHOST_DIR}"
