# Build the docker image based on a gentoo stage3 image

# Extract the public key line for kiel@localhost from the user's ~/.ssh/authorized_keys
AUTHKEY=""
if [ -f "${HOME}/.ssh/authorized_keys" ]; then
  AUTHKEY=$(grep "kiel@localhost" "${HOME}/.ssh/authorized_keys" | head -n1 || true)
fi

# The private key for the reverse tunnels should be provided in the environment as TUNNEL_KEY
# Example: export TUNNEL_KEY="$(cat ~/.ssh/tunnel_id_rsa)"
export TUNNEL_KEY="$(cat ~/.ssh/tinyus_key.pem)"
# It will be passed into the build and written into /etc/dracut-crypt-ssh/tunnel_id_rsa
DOCKER_BUILDKIT=1 docker build \
  --build-arg INITRAMFS_AUTH_KEY="$AUTHKEY" \
  --build-arg TUNNEL_KEY="${TUNNEL_KEY:-}" \
  -t gentoo-z6-min --progress=plain .
