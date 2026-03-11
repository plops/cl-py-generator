#!/bin/bash
set -e

# Log file
LOG_FILE="build_gcc_only.log"

# Find the line number where GCC build finishes
# We look for the end of the RUN command which contains @preserved-rebuild
# In the current Dockerfile, this is the last step of the gcc installation block
STOP_LINE=$(grep -n "@preserved-rebuild" Dockerfile | cut -d: -f1)

if [ -z "$STOP_LINE" ]; then
    echo "Could not find '@preserved-rebuild' in Dockerfile. Fallback to full builder stage."
    TARGET="builder"
    DOCKERFILE="Dockerfile"
else
    echo "Found GCC build end at line $STOP_LINE. Creating temporary Dockerfile."
    head -n "$STOP_LINE" Dockerfile > Dockerfile.gcc
    DOCKERFILE="Dockerfile.gcc"
    TARGET="" # No target needed as we build the whole (truncated) file
fi

# Arguments (copied from setup01_build_image.sh)
AUTHKEY=""
if [ -f "${HOME}/.ssh/authorized_keys" ]; then
  AUTHKEY=$(grep "kiel@localhost" "${HOME}/.ssh/authorized_keys" | head -n1 || true)
fi

TUNNEL_KEY_FILE="${HOME}/.ssh/tinyus_key.pem"
if [ -f "$TUNNEL_KEY_FILE" ]; then
    export TUNNEL_KEY="$(cat "$TUNNEL_KEY_FILE")"
else
    export TUNNEL_KEY=""
fi

echo "Starting build... Logging to $LOG_FILE"

# Run build with time measurement
START_TIME=$(date +%s)

# Using a subshell to capture time of the command
# We use 'time' shell keyword (or binary) to print stats to stderr, which is redirected to log
{ time DOCKER_BUILDKIT=1 docker build \
  --build-arg INITRAMFS_AUTH_KEY="$AUTHKEY" \
  --build-arg TUNNEL_KEY="${TUNNEL_KEY:-}" \
  ${TARGET:+--target "$TARGET"} \
  -f "$DOCKERFILE" \
  -t gentoo-gcc-only --progress=plain . ; } 2>&1 | tee "$LOG_FILE"

STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "Build finished with status $STATUS in $DURATION seconds." | tee -a "$LOG_FILE"

# Clean up temp file
if [ "$DOCKERFILE" == "Dockerfile.gcc" ]; then
    rm Dockerfile.gcc
fi

exit $STATUS
