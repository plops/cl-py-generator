#!/bin/sh
set -eu

# Why this script exists:
# - On this HP Z6 setup, having /lib/modules/<kver>/video present during boot can
#   crash/hang the system.
# - To avoid that, the image stores those modules at /video<kver> instead.
# - Run this script before startx to activate the video modules from tmpfs.
# - The workflow is designed so runtime writes stay in /dev/shm and do not leave
#   persistent overlay changes after reboot.
#
# Safety note:
# - An older version used:
#     cp -ar /video<kver> /dev/shm
#   which, when run twice, could create /dev/shm/video<kver>/video<kver>.
#   That recursive nesting can make depmod -a loop.
# - This script is idempotent and prevents that.

KVER="$(uname -r)"
STORE_DIR="/video${KVER}"
SHM_DIR="/dev/shm/video${KVER}"
MODULES_VIDEO="/lib/modules/${KVER}/video"

if [ ! -d "${STORE_DIR}" ]; then
    echo "missing source directory: ${STORE_DIR}" >&2
    exit 1
fi

# Clean up the old buggy nested copy layout if it exists.
if [ -e "${SHM_DIR}/video${KVER}" ]; then
    rm -rf "${SHM_DIR}/video${KVER}"
fi

# Populate tmpfs once.
if [ ! -d "${SHM_DIR}" ]; then
    rm -rf "${SHM_DIR}"
    cp -a "${STORE_DIR}" "${SHM_DIR}"
fi

# Ensure /lib/modules/<kver>/video points at tmpfs.
if [ -L "${MODULES_VIDEO}" ]; then
    CURRENT_TARGET="$(readlink "${MODULES_VIDEO}" || true)"
    if [ "${CURRENT_TARGET}" != "${SHM_DIR}" ]; then
        rm -f "${MODULES_VIDEO}"
        ln -s "${SHM_DIR}" "${MODULES_VIDEO}"
    fi
elif [ -e "${MODULES_VIDEO}" ]; then
    echo "refusing to overwrite non-symlink path: ${MODULES_VIDEO}" >&2
    exit 1
else
    ln -s "${SHM_DIR}" "${MODULES_VIDEO}"
fi

depmod -a "${KVER}"
modprobe nvidia
