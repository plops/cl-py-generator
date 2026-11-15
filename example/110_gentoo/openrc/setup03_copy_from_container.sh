#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TODAY="$(date +%Y%m%d)"
EXPORT_BASE="${EXPORT_BASE:-/dev/shm}"
mkdir -p "${EXPORT_BASE}"

AVAIL_BYTES="$(df -B1 --output=avail "${EXPORT_BASE}" | tail -n 1 | tr -d ' ')"
MIN_BYTES=$((6 * 1024 * 1024 * 1024))
if [ "${AVAIL_BYTES}" -lt "${MIN_BYTES}" ]; then
  echo "[FEHLER] ${EXPORT_BASE} hat nur ${AVAIL_BYTES} Bytes frei (< 6 GiB)." >&2
  echo "Im Agent-Container z. B. setzen: EXPORT_BASE=${SCRIPT_DIR}/out_export bash $0" >&2
  exit 1
fi

if [ -f logs/last_build_args.env ]; then
  # shellcheck disable=SC1091
  . logs/last_build_args.env
fi
BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT:-unknown}"
BUILD_REPOSITORY="${BUILD_REPOSITORY:-local}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

TARGET="${EXPORT_BASE}/gentoo-z6-min-openrc_${TODAY}"
TMP_TARGET="${TARGET}.tmp.$$"

rm -rf "${TMP_TARGET}"
mkdir -p "${TMP_TARGET}"

DOCKER_BUILDKIT=1 docker build \
  --target export \
  --build-arg BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT}" \
  --build-arg BUILD_REPOSITORY="${BUILD_REPOSITORY}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --output "type=local,dest=${TMP_TARGET}" \
  --progress=plain \
  .

rm -rf "${TARGET}"
mv "${TMP_TARGET}" "${TARGET}"
ls -lh "${TARGET}"
