#!/usr/bin/env bash
set -euo pipefail

uid="$(id -u)"
choose_runtime_dir() {
  if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR:-}" && -w "${XDG_RUNTIME_DIR:-}" ]]; then
    printf '%s\n' "$XDG_RUNTIME_DIR"
    return 0
  fi

  if [[ -d "/run/user/$uid" && -w "/run/user/$uid" ]]; then
    printf '%s\n' "/run/user/$uid"
    return 0
  fi

  local fallback="/tmp/pipewire-runtime-$uid"
  mkdir -p "$fallback"
  chmod 700 "$fallback"
  printf '%s\n' "$fallback"
}

export XDG_RUNTIME_DIR="$(choose_runtime_dir)"
export PULSE_SERVER="unix:$XDG_RUNTIME_DIR/pulse/native"

cleanup_stale_runtime() {
  mkdir -p "$XDG_RUNTIME_DIR/pulse"

  if ! pgrep -u "$uid" -x pipewire >/dev/null 2>&1; then
    rm -f \
      "$XDG_RUNTIME_DIR/pipewire-0" \
      "$XDG_RUNTIME_DIR/pipewire-0.lock" \
      "$XDG_RUNTIME_DIR/pipewire-0-manager" \
      "$XDG_RUNTIME_DIR/pipewire-0-manager.lock"
  fi

  if ! pgrep -u "$uid" -x pipewire-pulse >/dev/null 2>&1; then
    rm -f \
      "$XDG_RUNTIME_DIR/pulse/native" \
      "$XDG_RUNTIME_DIR/pulse/pid"
  fi
}

write_pulse_client_conf() {
  local pulse_dir="$HOME/.config/pulse"
  local client_conf="$pulse_dir/client.conf"

  mkdir -p "$pulse_dir"

  if [[ -f "$client_conf" ]] && grep -q '^[[:space:]]*default-server[[:space:]]*=' "$client_conf"; then
    return 0
  fi

  if ! cat >"$client_conf" <<EOF
default-server = $PULSE_SERVER
autospawn = no
daemon-binary = /bin/true
enable-shm = false
EOF
  then
    echo "Konnte $client_conf nicht schreiben."
  fi
}

start_if_needed() {
  local name="$1"
  shift

  if pgrep -u "$uid" -x "$name" >/dev/null 2>&1; then
    echo "$name laeuft bereits."
    return 0
  fi

  echo "Starte $name ..."
  nohup "$@" >/tmp/"$name".log 2>&1 &
}

cleanup_stale_runtime
write_pulse_client_conf

start_if_needed pipewire pipewire
sleep 1
start_if_needed pipewire-pulse pipewire-pulse
start_if_needed wireplumber wireplumber

echo "PipeWire-Stack gestartet."
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "PULSE_SERVER=$PULSE_SERVER"
