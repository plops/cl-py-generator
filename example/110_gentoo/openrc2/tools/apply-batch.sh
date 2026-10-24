#!/bin/bash
# Erzeugt config/package.env und config/sanitized-packages.txt aus den Batches
# bis zur gewuenschten Stufe: ./tools/apply-batch.sh <1..5>
# Stufe 0 = keine Sanitizer (Referenz). Idempotent, deterministisch.
set -euo pipefail
OPENRC2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LEVEL="${1:-}"
if ! [[ "${LEVEL}" =~ ^[0-5]$ ]]; then
  echo "Nutzung: $0 <stufe 0..5>" >&2
  exit 1
fi

HELD=$(grep -vE '^\s*(#|$)' "${OPENRC2_DIR}/config/batches/hold.txt" | awk '{print $1}' || true)
{
  echo "# Erzeugt von tools/apply-batch.sh, Stufe ${LEVEL}."
  echo "# NICHT von Hand editieren; Batches + hold.txt liegen in config/batches/."
  echo "# Toolchain/libc bleiben immer unsanitisiert (ASan braucht saubere libc)."
  echo "sys-libs/glibc nosanitizer"
  echo "sys-devel/gcc nosanitizer"
  echo "sys-devel/binutils nosanitizer"
  for h in ${HELD}; do
    echo "${h} nosanitizer"
  done
  for n in 1 2 3 4 5; do
    [ "${n}" -gt "${LEVEL}" ] && break
    f="${OPENRC2_DIR}/config/batches/batch-0${n}.txt"
    echo "# --- batch-0${n} ---"
    grep -vE '^\s*(#|$)' "${f}" | while read -r atom; do
      if echo "${HELD}" | grep -qxF "${atom}"; then
        echo "# HOLD (s. hold.txt): ${atom}"
      else
        echo "${atom} sanitizers"
      fi
    done
  done
} > "${OPENRC2_DIR}/config/package.env"

{
  for n in 1 2 3 4 5; do
    [ "${n}" -gt "${LEVEL}" ] && break
    grep -vE '^\s*(#|$)' "${OPENRC2_DIR}/config/batches/batch-0${n}.txt"
  done | while read -r atom; do
    echo "${HELD}" | grep -qxF "${atom}" || echo "${atom}"
  done
} > "${OPENRC2_DIR}/config/sanitized-packages.txt"

echo "Stufe ${LEVEL}: $(grep -cv '^#' "${OPENRC2_DIR}/config/sanitized-packages.txt") Sanitizer-Pakete"
