#!/usr/bin/env bash
# Baut das minimale QEMU-Testimage (openrc2, ohne ASan).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AUTHKEY=""
if [ -f "${HOME}/.ssh/authorized_keys" ]; then
  AUTHKEY=$(grep "kiel@localhost" "${HOME}/.ssh/authorized_keys" | head -n1 || true)
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILD_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  --build-arg BUILD_REPOSITORY="openrc2-minimal-qemu" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t gentoo-qemu-min-openrc --progress=plain .
