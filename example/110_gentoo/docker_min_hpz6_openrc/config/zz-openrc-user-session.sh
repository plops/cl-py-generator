#!/bin/sh

uid="$(id -u 2>/dev/null || printf '')"

if [ -n "${uid}" ] && [ "${uid}" -ge 1000 ] 2>/dev/null; then
  : "${XDG_RUNTIME_DIR:=/run/user/${uid}}"
  export XDG_RUNTIME_DIR

  if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
  fi

  if [ -z "${PULSE_SERVER:-}" ]; then
    export PULSE_SERVER="unix:${XDG_RUNTIME_DIR}/pulse/native"
  fi
fi
