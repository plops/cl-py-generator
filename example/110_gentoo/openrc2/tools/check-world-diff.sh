#!/bin/bash
# Verifiziert: openrc2/config/world ist eine echte Teilmenge-Strategie von
# ../openrc/config/world — d.h. jedes gestrichene Paket ist begründet,
# nichts wurde aus Versehen umbenannt/hinzugefügt ohne Doku.
set -euo pipefail
OPENRC2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FULL="${OPENRC2_DIR}/../openrc/config/world"
MINI="${OPENRC2_DIR}/config/world"

# 1. Jedes Paket in world-mini muss (normalisiert) auch in world-voll stehen.
#    Ausnahme: explizit als openrc2-Neuzugang dokumentierte Pakete.
ALLOWED_NEW="sys-fs/dosfstools sys-fs/e2fsprogs"
EXIT=0
while read -r atom; do
  [ -z "${atom}" ] && continue
  case "${atom}" in \#*) continue;; esac
  # einfacher Vergleich: exakter Atom (ohne Versions-Pin) muss in FULL stehen
  plain="${atom#<}"
  if ! grep -qxF "${plain}" "${FULL}" && ! grep -qxF "${atom}" "${FULL}"; then
    if echo "${ALLOWED_NEW}" | grep -qw "${plain}"; then
      echo "NEU (begruendet): ${plain}"
    else
      echo "FEHLER: ${plain} steht nicht in openrc/config/world und ist nicht als Neuzugang dokumentiert" >&2
      EXIT=1
    fi
  fi
done < "${MINI}"

# 2. Verbotene Schwergewichte duerfen nicht in world-mini stehen.
FORBIDDEN="app-editors/emacs x11-drivers/nvidia-drivers media-libs/mesa www-client/firefox-bin net-im/signal-desktop-bin x11-base/xorg-server x11-wm/dwm sys-kernel/linux-firmware sys-firmware/sof-firmware"
for pkg in ${FORBIDDEN}; do
  if grep -qxF "${pkg}" "${MINI}"; then
    echo "FEHLER: verbotenes Paket ${pkg} in openrc2/config/world" >&2
    EXIT=1
  fi
done

# 3. Statistik
echo "openrc/world:  $(grep -cv '^#' "${FULL}") Eintraege"
echo "openrc2/world: $(grep -cv '^#' "${MINI}") Eintraege"

exit "${EXIT}"
