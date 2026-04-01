#!/bin/bash
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

DHCP_IF=eno1
STATIC_IF=enp65s0
STATIC_ADDRS=(
  "192.168.254.123/24"
  "192.168.178.122/24"
)
MODULES=(
  r8169
  igc
)

for module in "${MODULES[@]}"; do
  modprobe "$module"
done

ip link set dev "$DHCP_IF" up
ip link set dev "$STATIC_IF" up

if command -v pkill >/dev/null 2>&1; then
  pkill -f "dhclient.*${DHCP_IF}" || true
fi

dhclient -r "$DHCP_IF" || true
dhclient -v "$DHCP_IF"

ip -4 addr flush dev "$STATIC_IF" scope global || true
for addr in "${STATIC_ADDRS[@]}"; do
  ip addr add "$addr" dev "$STATIC_IF"
done

echo
ip -br addr show dev "$DHCP_IF"
ip -br addr show dev "$STATIC_IF"
