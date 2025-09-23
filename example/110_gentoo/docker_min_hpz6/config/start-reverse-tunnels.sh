#!/bin/sh
# Script executed inside initramfs to create reverse tunnels back to public servers.
# Expects /etc/dracut-crypt-ssh/tunnel_id_rsa to exist with private key and
# that autossh is available inside the initramfs.

export PATH=/usr/bin:/usr/sbin:/bin:/sbin

# small delay to let network come up
sleep 5

# ensure key permissions
if [ -f /etc/dracut-crypt-ssh/tunnel_id_rsa ]; then
  chmod 600 /etc/dracut-crypt-ssh/tunnel_id_rsa
fi

# helper to try starting a tunnel robustly
start_tunnel() {
  SERVER="$1"
  REMOTE_PORT="$2"
  # use autossh if available, fall back to ssh
  if command -v autossh >/dev/null 2>&1; then
    autossh -M 0 -f -N \
      -o "ExitOnForwardFailure=yes" \
      -o "ServerAliveInterval=30" \
      -o "ServerAliveCountMax=3" \
      -o "StrictHostKeyChecking=no" \
      -o "UserKnownHostsFile=/dev/null" \
      -i /etc/dracut-crypt-ssh/tunnel_id_rsa \
      -R ${REMOTE_PORT}:localhost:2222 "user@${SERVER}" >/dev/null 2>&1 || true
  else
    ssh -f -N \
      -o "ExitOnForwardFailure=yes" \
      -o "ServerAliveInterval=30" \
      -o "ServerAliveCountMax=3" \
      -o "StrictHostKeyChecking=no" \
      -o "UserKnownHostsFile=/dev/null" \
      -i /etc/dracut-crypt-ssh/tunnel_id_rsa \
      -R ${REMOTE_PORT}:localhost:2222 "user@${SERVER}" >/dev/null 2>&1 || true
  fi
}

# Start tunnels to the two public hosts. Adjust ports/users as needed.
start_tunnel "tinyeu" 2332
start_tunnel "tinyus" 2332

# keep script background-friendly; dracut-crypt-ssh expects it to return quickly
exit 0

