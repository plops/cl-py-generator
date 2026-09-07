# Build the docker image for ThinkPad E14 minimal (no NVIDIA)

# Extract the public key line for kiel@localhost from the user's ~/.ssh/authorized_keys
AUTHKEY=""
if [ -f "${HOME}/.ssh/authorized_keys" ]; then
  AUTHKEY=$(grep "kiel@localhost" "${HOME}/.ssh/authorized_keys" | head -n1 || true)
fi

# The private key for the reverse tunnels should be provided in the environment as TUNNEL_KEY
# Example: export TUNNEL_KEY="$(cat ~/.ssh/tunnel_id_rsa)"
export TUNNEL_KEY="$(cat ~/.ssh/tinyus_key.pem)"
# It will be passed into the build and written into /etc/dracut-crypt-ssh/tunnel_id_rsa
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf '%s' unknown)"
REPOSITORY="$(git -C "$REPO_ROOT" config --get remote.origin.url 2>/dev/null || printf '%s' "$REPO_ROOT")"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DOCKER_BUILDKIT=1 docker build \
  --build-arg INITRAMFS_AUTH_KEY="$AUTHKEY" \
  --build-arg TUNNEL_KEY="${TUNNEL_KEY:-}" \
  --build-arg BUILD_GIT_COMMIT="$GIT_COMMIT" \
  --build-arg BUILD_REPOSITORY="$REPOSITORY" \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  -t gentoo-z6-min-openrc --progress=plain .
