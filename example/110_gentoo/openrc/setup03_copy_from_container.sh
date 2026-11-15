#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TODAY="$(date +%Y%m%d)"
EXPORT_BASE="${EXPORT_BASE:-/dev/shm}"
EXPORT_BINPKGS="${EXPORT_BINPKGS:-1}"
MIN_GIB="${MIN_GIB:-8}"
mkdir -p "${EXPORT_BASE}"

AVAIL_BYTES="$(df -B1 --output=avail "${EXPORT_BASE}" | tail -n 1 | tr -d ' ')"
MIN_BYTES=$((MIN_GIB * 1024 * 1024 * 1024))
if [ "${AVAIL_BYTES}" -lt "${MIN_BYTES}" ]; then
  echo "[FEHLER] ${EXPORT_BASE} hat nur ${AVAIL_BYTES} Bytes frei (< ${MIN_GIB} GiB)." >&2
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
TMP_BINPKGS="${TARGET}.binpkgs.$$"

rm -rf "${TMP_TARGET}" "${TMP_BINPKGS}"
mkdir -p "${TMP_TARGET}"

DOCKER_BUILDKIT=1 docker build \
  --target export \
  --build-arg BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT}" \
  --build-arg BUILD_REPOSITORY="${BUILD_REPOSITORY}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --output "type=local,dest=${TMP_TARGET}" \
  --progress=plain \
  .

if [ "${EXPORT_BINPKGS}" = "1" ]; then
  mkdir -p "${TMP_BINPKGS}"
  DOCKER_BUILDKIT=1 docker build \
    --target export-binpkgs-out \
    --build-arg BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT}" \
    --build-arg BUILD_REPOSITORY="${BUILD_REPOSITORY}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --output "type=local,dest=${TMP_BINPKGS}" \
    --progress=plain \
    .
  mv "${TMP_BINPKGS}/binpkgs" "${TMP_TARGET}/binpkgs"
  mv "${TMP_BINPKGS}/upstream_binpkgs.tsv" "${TMP_TARGET}/upstream_binpkgs.tsv"
  mv "${TMP_BINPKGS}/upstream_binpkgs.txt" "${TMP_TARGET}/upstream_binpkgs.txt"
  rmdir "${TMP_BINPKGS}"
fi

rm -rf "${TARGET}"
mv "${TMP_TARGET}" "${TARGET}"
ls -lh "${TARGET}"
if [ -d "${TARGET}/binpkgs" ]; then
  du -sh "${TARGET}/binpkgs"
fi
