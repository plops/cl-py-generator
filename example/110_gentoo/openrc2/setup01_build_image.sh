#!/usr/bin/env bash
# Baut das minimale QEMU-Testimage (openrc2, ohne ASan).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# SSH_PUBKEY darf aus der Umgebung kommen; sonst Rueckfall auf den
# bekannten Gast-Schluessel (sonst kein SSH in den Gast).
if [ -z "${SSH_PUBKEY:-}" ] && [ -f /tmp/qemu-guest-key.pub ]; then
  SSH_PUBKEY=$(head -n1 /tmp/qemu-guest-key.pub || true)
fi
if [ -z "${SSH_PUBKEY:-}" ] && [ -f "${HOME}/.ssh/authorized_keys" ]; then
  SSH_PUBKEY=$(grep -h "qemu-guest\|kiel@localhost" "${HOME}/.ssh/authorized_keys" 2>/dev/null | head -n1 || true)
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILD_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  --build-arg BUILD_REPOSITORY="openrc2-minimal-qemu" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg SSH_PUBKEY="${SSH_PUBKEY:-}" \
  -t gentoo-qemu-min-openrc --progress=plain .
