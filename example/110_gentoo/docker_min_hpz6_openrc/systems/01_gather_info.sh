#!/bin/bash
# Gather hardware-only information for offline hardware configuration.
# Purpose:
#  - Collect details about CPU, memory, buses, block devices, firmware/BIOS,
#    PCI/USB devices, kernel modules, sensors and SMART data.
#  - Avoid collecting intrusive information: no network config or IPs,
#    no local users, no running process lists, and do not read per-system
#    config files like /etc/fstab or /etc/os-release.
#
# Usage:
#  - Create an empty directory, cd into it, then run:
#      sudo ../01_gather_info.sh
#
# Privacy: this script intentionally avoids collecting network, user or
# filesystem-specific confidential data. It focuses on hardware description
# only so the outputs are suitable for building a minimal OS/kernel/driver
# configuration.

# Helper: record missing tools and optionally run commands with warnings.
MISSING_TOOLS=()

check_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "WARNING: required tool '$tool' not found" >&2
        MISSING_TOOLS+=("$tool")
        return 1
    fi
    return 0
}

# run_or_warn <outfile> <cmd> [args...]
# if <cmd> exists run it and write stdout to <outfile>, otherwise write a note
run_or_warn() {
    local outfile="$1"; shift
    local cmd="$1"; shift
    if check_tool "$cmd"; then
        "$cmd" "$@" > "$outfile" 2>/dev/null || echo "$cmd failed" > "$outfile"
    else
        echo "$cmd not available" > "$outfile"
    fi
}

# try_alternatives <outfile> <cmd1> [args...] '::' <cmd2> [args...] ...
# use '::' as separator between alternative commands
try_alternatives() {
    local outfile="$1"; shift
    local i=1
    local sep_index
    local tried=0
    local cmd args
    while [ $# -gt 0 ]; do
        # gather one command until '::' or end
        cmd="$1"; shift
        args=()
        while [ $# -gt 0 ] && [ "$1" != "::" ]; do
            args+=("$1"); shift
        done
        if [ $# -gt 0 ] && [ "$1" = "::" ]; then
            shift
        fi
        if check_tool "$cmd"; then
            "$cmd" "${args[@]}" > "$outfile" 2>/dev/null && return 0
            echo "$cmd failed" > "$outfile" && return 0
        fi
        tried=$((tried+1))
    done
    echo "no alternative commands available" > "$outfile"
    return 1
}

report_missing_tools() {
    if [ "${#MISSING_TOOLS[@]}" -gt 0 ]; then
        echo ""
        echo "The following tools were not found and some outputs may be incomplete:"
        for t in "${MISSING_TOOLS[@]}"; do
            echo "  - $t"
        done
        echo "Install them (e.g. apt/yum/pacman) or run 00_setup_gather_dependencies.sh"
    fi
}

# Hardware probes (non-intrusive)
# CPU and kernel
cat /proc/cpuinfo > cpuinfo.txt
uname -a > uname.txt
run_or_warn lscpu.txt lscpu

# Kernel modules (helps identify loaded drivers)
run_or_warn lsmod.txt lsmod

# PCI/USB buses and system topology
run_or_warn lspci.txt lspci
run_or_warn lsusb.txt lsusb
run_or_warn lshw.txt lshw

# DMI / firmware / BIOS information (may require root)
if check_tool dmidecode; then
    dmidecode > dmidecode.txt 2>/dev/null || echo "dmidecode failed or requires root" > dmidecode.txt
else
    echo "dmidecode requires root or is not installed" > dmidecode.txt
fi

# Memory (hardware-related only)
cat /proc/meminfo > meminfo.txt
run_or_warn free.txt free -h
run_or_warn tlp-stat.txt tlp-stat
# Block devices: use non-invasive listing that focuses on device attributes
# (avoid printing mount paths or file contents)
if check_tool lsblk; then
    # -d = only devices, -n = no headings, -o = selected columns
    lsblk -dn -o NAME,TYPE,SIZE,MODEL,VENDOR > lsblk_devices.txt 2>/dev/null || echo "lsblk failed" > lsblk_devices.txt
else
    echo "lsblk not available" > lsblk_devices.txt
fi

# Also capture full lsblk output (devices + children) for topology reference
run_or_warn lsblk_all.txt lsblk -a

# SMART info for block devices
for disk in /dev/sd? /dev/sd?? /dev/nvme[0-9]n[0-9] /dev/hd?; do
    [ -b "$disk" ] || continue
    if check_tool smartctl; then
        smartctl -xa "$disk" > "smartctl_$(basename "$disk").txt" # 2>/dev/null || echo "smartctl failed for $disk" > "smartctl_$(basename "$disk").txt"
	# sleep 1
    else
        echo "smartctl not available" > "smartctl_$(basename "$disk").txt"
    fi
done

# Sensors / temps (may need lm-sensors)
if check_tool sensors; then
    sensors > sensors.txt 2>/dev/null || echo "sensors failed" > sensors.txt
else
    echo "sensors not available" > sensors.txt
fi

# Additional non-identifying system files requested by user:
# - blkid, df, dmesg: useful for storage/kernel diagnostics
# - /etc/fstab and /etc/os-release and hostnamectl: saved for offline review then sanitized
# - fdisk: add partition table listing (fdisk -l) for disk layout info
run_or_warn fdisk.txt fdisk -l
run_or_warn blkid.txt blkid
run_or_warn df.txt df -h

# dmesg may require root to include all messages; try if available
run_or_warn dmesg.txt dmesg

# Save /etc/fstab and /etc/os-release if readable (these may contain host/user references;
# they will be anonymized by 03_anonymize_files.sh)
if [ -r /etc/fstab ]; then
    cp /etc/fstab fstab.txt 2>/dev/null || { echo "failed to copy /etc/fstab" > fstab.txt; }
else
    echo "/etc/fstab not readable" > fstab.txt
fi

if [ -r /etc/os-release ]; then
    cp /etc/os-release os-release.txt 2>/dev/null || { echo "failed to copy /etc/os-release" > os-release.txt; }
else
    echo "/etc/os-release not readable" > os-release.txt
fi

# hostnamectl (may show hostname/pretty name; will be sanitized by anonymizer)
run_or_warn hostnamectl.txt hostnamectl

# System overview (non-identifying)
run_or_warn fastfetch.txt fastfetch

# Final summary of missing tools
report_missing_tools

# End of hardware-only gather script
