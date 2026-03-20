#!/usr/bin/env bash
set -euo pipefail

WORLD_FILE="${1:-/var/lib/portage/world}"
TOP_N="${TOP_N:-40}"

if ! command -v qsize >/dev/null 2>&1; then
  echo "error: qsize not found (install app-portage/portage-utils)" >&2
  exit 1
fi

if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "error: world file not found: ${WORLD_FILE}" >&2
  exit 1
fi

while IFS= read -r atom; do
  [[ -z "${atom}" ]] && continue
  [[ "${atom}" =~ ^[[:space:]]*# ]] && continue

  line="$(qsize -m "${atom}" 2>/dev/null | head -n 1 || true)"
  [[ -z "${line}" ]] && continue

  pkg="$(awk -F: '{print $1}' <<<"${line}")"
  size="$(sed -n 's/.*,[[:space:]]*\([0-9.]\+\)[[:space:]]\+MiB$/\1/p' <<<"${line}")"
  [[ -z "${size}" ]] && continue

  printf "%s\t%s\t%s\n" "${size}" "${pkg}" "${atom}"
done < "${WORLD_FILE}" |
  sort -nr -k1,1 |
  head -n "${TOP_N}" |
  awk -F'\t' '{printf "%8.1f MiB  %s  [%s]\n", $1, $2, $3}'
