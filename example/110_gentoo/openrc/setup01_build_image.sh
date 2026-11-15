#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "${GIT_COMMIT}" != "unknown" ] && ! git diff --quiet 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi
REPO_URL="$(git config --get remote.origin.url 2>/dev/null || echo local)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > logs/last_build_args.env <<EOF
BUILD_GIT_COMMIT=${GIT_COMMIT}
BUILD_REPOSITORY=${REPO_URL}
BUILD_DATE=${BUILD_DATE}
EOF

# Build debuggable target image AND full export graph
DOCKER_BUILDKIT=1 docker build \
  --target target \
  --build-arg BUILD_GIT_COMMIT="${GIT_COMMIT}" \
  --build-arg BUILD_REPOSITORY="${REPO_URL}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  -t gentoo-z6-min-openrc \
  --progress=plain \
  .

DOCKER_BUILDKIT=1 docker build \
  --target export \
  --build-arg BUILD_GIT_COMMIT="${GIT_COMMIT}" \
  --build-arg BUILD_REPOSITORY="${REPO_URL}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --progress=plain \
  .
