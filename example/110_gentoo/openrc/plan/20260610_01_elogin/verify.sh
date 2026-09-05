#!/usr/bin/env bash
set -euo pipefail

uid="$(id -u)"
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$uid}"
bus_path="$runtime_dir/bus"

echo "USER=$(id -un)"
echo "UID=$uid"
echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-}"
echo "DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-}"
echo

echo "Runtime directory:"
ls -ld /run/user "$runtime_dir" 2>/dev/null || true
echo

echo "Bus socket:"
ls -l "$bus_path" 2>/dev/null || true
echo

echo "loginctl:"
loginctl 2>&1 || true
echo

echo "dbus session probe:"
dbus-send --session \
  --dest=org.freedesktop.DBus \
  --type=method_call \
  --print-reply \
  /org/freedesktop/DBus \
  org.freedesktop.DBus.ListNames >/dev/null 2>&1 \
  && echo "session bus reachable" \
  || echo "session bus unreachable"
