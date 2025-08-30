#!/bin/bash
# Install required tools for the hardware-only gather script (01_gather_info.sh).
#
# Purpose:
#  - Ensure the minimal set of utilities that probe hardware are present:
#    fastfetch, lshw, lspci, lsusb, smartctl, dmidecode, lscpu, lsmod, sensors, lsblk.
#  - Do not install or enable tools that gather network, user or process info.
#
# Usage: run as root or with sudo; the script will detect distro and install
# appropriate packages for the reduced toolset.

set -euo pipefail

# If not root, use sudo if available
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "Please run as root or install sudo." >&2
        exit 1
    fi
fi

# detect distro id
if [ -r /etc/os-release ]; then
    . /etc/os-release
    DISTRO_ID="${ID,,}"
    DISTRO_LIKE="${ID_LIKE:-}"
else
    echo "Cannot detect distro (/etc/os-release missing)" >&2
    exit 1
fi

# reduced set of required commands used by the hardware-only gather script
required_cmds=(fastfetch lshw lspci lsusb smartctl dmidecode lscpu lsmod sensors lsblk blkid df dmesg fdisk hostnamectl)

# per-distro mapping from command -> package (only entries for the reduced set)
declare -A pkg_debian=(
    [fastfetch]=fastfetch
    [lshw]=lshw
    [lspci]=pciutils
    [lsusb]=usbutils
    [smartctl]=smartmontools
    [dmidecode]=dmidecode
    [lscpu]=util-linux
    [lsmod]=kmod
    [sensors]=lm-sensors
    [lsblk]=util-linux
    [blkid]=util-linux
    [df]=util-linux
    [dmesg]=util-linux
    [fdisk]=util-linux
    [hostnamectl]=systemd
)
declare -A pkg_fedora=(
    [fastfetch]=fastfetch
    [lshw]=lshw
    [lspci]=pciutils
    [lsusb]=usbutils
    [smartctl]=smartmontools
    [dmidecode]=dmidecode
    [lscpu]=util-linux
    [lsmod]=kmod
    [sensors]=lm_sensors
    [lsblk]=util-linux
    [blkid]=util-linux
    [df]=util-linux
    [dmesg]=util-linux
    [fdisk]=util-linux
    [hostnamectl]=systemd
)
declare -A pkg_arch=(
    [fastfetch]=fastfetch
    [lshw]=lshw
    [lspci]=pciutils
    [lsusb]=usbutils
    [smartctl]=smartmontools
    [dmidecode]=dmidecode
    [lscpu]=util-linux
    [lsmod]=kmod
    [sensors]=lm_sensors
    [lsblk]=util-linux
    [blkid]=util-linux
    [df]=util-linux
    [dmesg]=util-linux
    [fdisk]=util-linux
    [hostnamectl]=systemd
)
declare -A pkg_gentoo=(
    [fastfetch]=app-misc/fastfetch
    [lshw]=sys-apps/lshw
    [lspci]=sys-apps/pciutils
    [lsusb]=sys-apps/usbutils
    [smartctl]=sys-apps/smartmontools
    [dmidecode]=sys-apps/dmidecode
    [lscpu]=sys-apps/util-linux
    [lsmod]=sys-apps/kmod
    [sensors]=sys-apps/lm-sensors
    [lsblk]=sys-apps/util-linux
    [blkid]=sys-apps/util-linux
    [df]=sys-apps/util-linux
    [dmesg]=sys-apps/util-linux
    [fdisk]=sys-apps/util-linux
    [hostnamectl]=sys-apps/systemd
)

pkgs_to_install=()

# helper to add package if the command is missing
add_pkg_if_missing() {
    local cmd=$1
    local pkg=$2
    [ -z "$pkg" ] && return
    if ! command -v "$cmd" >/dev/null 2>&1; then
        # avoid duplicates
        for p in "${pkgs_to_install[@]}"; do
            if [ "$p" = "$pkg" ]; then
                return
            fi
        done
        pkgs_to_install+=("$pkg")
    fi
}

case "$DISTRO_ID" in
    ubuntu|debian)
        for cmd in "${required_cmds[@]}"; do
            add_pkg_if_missing "$cmd" "${pkg_debian[$cmd]}"
        done
        if [ "${#pkgs_to_install[@]}" -eq 0 ]; then
            echo "All required commands already installed."
            exit 0
        fi
        echo "Updating package lists..."
        $SUDO apt-get update -y
        echo "Installing: ${pkgs_to_install[*]}"
        $SUDO apt-get install -y "${pkgs_to_install[@]}"
        ;;
    fedora)
        for cmd in "${required_cmds[@]}"; do
            add_pkg_if_missing "$cmd" "${pkg_fedora[$cmd]}"
        done
        if [ "${#pkgs_to_install[@]}" -eq 0 ]; then
            echo "All required commands already installed."
            exit 0
        fi
        echo "Installing: ${pkgs_to_install[*]}"
        $SUDO dnf install -y "${pkgs_to_install[@]}"
        ;;
    arch)
        for cmd in "${required_cmds[@]}"; do
            add_pkg_if_missing "$cmd" "${pkg_arch[$cmd]}"
        done
        if [ "${#pkgs_to_install[@]}" -eq 0 ]; then
            echo "All required commands already installed."
            exit 0
        fi
        echo "Refreshing pacman DB..."
        $SUDO pacman -Sy --noconfirm
        echo "Installing: ${pkgs_to_install[*]}"
        $SUDO pacman -S --noconfirm "${pkgs_to_install[@]}"
        ;;
    gentoo)
        for cmd in "${required_cmds[@]}"; do
            add_pkg_if_missing "$cmd" "${pkg_gentoo[$cmd]}"
        done
        if [ "${#pkgs_to_install[@]}" -eq 0 ]; then
            echo "All required commands already installed."
            exit 0
        fi
        echo "Installing (emerge): ${pkgs_to_install[*]}"
        $SUDO emerge --verbose "${pkgs_to_install[@]}"
        ;;
    *)
        echo "Unsupported distro: $DISTRO_ID. Supported: gentoo, ubuntu, debian, fedora, arch" >&2
        exit 2
        ;;
esac

echo "Installation finished."

# Post-install notes
if command -v sensors >/dev/null 2>&1; then
    echo "If lm-sensors was just installed, run 'sudo sensors-detect' and then reboot or load modules as needed to enable sensors."
fi
