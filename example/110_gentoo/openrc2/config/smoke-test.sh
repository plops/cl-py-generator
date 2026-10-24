#!/bin/bash
# Smoke-Test für den QEMU-Gast (openrc2, ohne ASan).
# Aufruf im Gast als root nach dem Boot. Exit-Code != 0 bei Fehlern.
set -euo pipefail

PASS=0
FAIL=0
check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "PASS: ${name}"; PASS=$((PASS+1))
  else
    echo "FAIL: ${name}"; FAIL=$((FAIL+1))
  fi
}

check "kernel" uname -r
check "root-mount" mountpoint -q /
check "virtio-blk" sh -c "ls /sys/bus/virtio/devices/ | grep -q ."
check "network-tool" ip link
check "gcc" gcc --version
check "git" git --version
check "picocom" picocom --help
check "sshd-service" rc-service sshd status
check "vfat-support" sh -c "grep -q vfat /proc/filesystems"
check "no-heavyweights" sh -c "! qlist -I | grep -qE 'llvm|nvidia|mesa|emacs|firefox|firmware'"
check "bash-plain" sh -c "! ldd /bin/bash | grep -q libasan"
check "asan-ls" sh -c "ldd /bin/ls | grep -q libasan"
check "asan-grep" sh -c "ldd /usr/bin/grep | grep -q libasan"
check "asan-sed" sh -c "ldd /usr/bin/sed | grep -q libasan"

echo "Ergebnis: ${PASS} PASS, ${FAIL} FAIL"
[ "${FAIL}" -eq 0 ]
