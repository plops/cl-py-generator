#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/setup01_run_${TS}.log"
META_FILE="logs/setup01_run_${TS}.meta"

START_ISO="$(date -Iseconds)"
START_EPOCH="$(date +%s)"
SECONDS=0

set +e
bash setup01_build_image.sh > >(tee "${LOG_FILE}") 2>&1
EXIT_CODE=$?
set -e

DURATION_SECONDS="${SECONDS}"
END_ISO="$(date -Iseconds)"
END_EPOCH="$(date +%s)"

cat > "${META_FILE}" <<EOF
start=${START_ISO}
end=${END_ISO}
start_epoch=${START_EPOCH}
end_epoch=${END_EPOCH}
duration_seconds=${DURATION_SECONDS}
exit_code=${EXIT_CODE}
cmd=bash setup01_build_image.sh
log=${LOG_FILE}
EOF

echo "LOG=${LOG_FILE}"
echo "META=${META_FILE}"
echo "EXIT_CODE=${EXIT_CODE}"
echo "DURATION_SECONDS=${DURATION_SECONDS}"

exit "${EXIT_CODE}"
