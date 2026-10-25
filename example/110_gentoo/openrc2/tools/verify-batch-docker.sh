#!/bin/bash
# Schnelle Batch-Verifikation im Docker-Container (statt QEMU/SSH).
# Nutzung: ./tools/verify-batch-docker.sh [image]  (Default: gentoo-qemu-min-openrc)
# Prueft: ASan-Coverage der Batch-Binaries, funktionale Laeufe, LSan/UBSan-Demos.
set -euo pipefail
IMG="${1:-gentoo-qemu-min-openrc}"
FAIL=0
say() { echo "$1"; }

check() { # name, command...
  local name="$1"; shift
  # LSAN-Policy global (Run-42): System-Vorladung braucht exitcode=0 auch hier.
  if docker run --rm -e LSAN_OPTIONS=exitcode=0 "${IMG}" bash -c "$*" >/dev/null 2>&1; then
    say "PASS: ${name}"
  else
    say "FAIL: ${name}"; FAIL=$((FAIL+1))
  fi
}

say "== Image: ${IMG} =="
# Coverage: Batch-Binaries enthalten AddressSanitizer-Symbolik.
# Link-Wahrheit via readelf (Run-42): ldd zeigt seit /etc/ld.so.preload bei
# JEDEM Binary libasan, beweist also keine Linkung mehr.
for bin in /bin/ls /usr/bin/grep /usr/bin/sed /bin/mount; do
  check "asan-linked $(basename "${bin}")" "readelf -d ${bin} | grep -q libasan"
done
check "bash-plain (Interpreter-zuletzt)" "! readelf -d /bin/bash | grep -q libasan"
# Funktional: Kernwerkzeuge laufen (Exit 0, kein Sanitizer-Abbruch)
check "run ls" "ls / >/dev/null"
check "run grep" "[ \"$(echo hello | grep ell)\" = hello ]"
check "run sed" "echo hi | sed -e s/h/H/"
check "run mount" "mount --version"
check "run bash -c" "bash -c true"
# LSan-Demo: gewollter Leak muss gemeldet werden (Detektion aktiv)
check "lsan-demo" "printf '#include <stdlib.h>\nint main(void){malloc(7);return 0;}\n' > /tmp/l.c && gcc -fsanitize=address /tmp/l.c -o /tmp/l && ASAN_OPTIONS=detect_leaks=1 /tmp/l 2>&1 | grep -q 'detected memory leaks'"
# UBSan-Demo: gewollter Overflow muss gemeldet werden
check "ubsan-demo" "printf '#include <stdio.h>\nint main(void){int x=2147483647; x=x+1; printf(\"%d\",x);return 0;}\n' > /tmp/u.c && gcc -fsanitize=undefined /tmp/u.c -o /tmp/u && /tmp/u 2>&1 | grep -q 'runtime error'"
# Schwergewichte duerfen nicht drin sein
check "no-heavyweights" "! qlist -I 2>/dev/null | grep -qE 'llvm-core/llvm|x11-drivers/nvidia-drivers|media-libs/mesa|app-editors/emacs|www-client/firefox-bin'"

say "== Ergebnis: FAIL=${FAIL} =="
exit "${FAIL}"
