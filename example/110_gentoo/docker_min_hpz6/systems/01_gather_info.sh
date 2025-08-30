#!/bin/bash

# Gather comprehensive system information. This is helpful to find the correct
# drivers and settings for a specific hardware setup. If you have a new computer
# you can run a Fedora or Ubuntu live system and execute this script there.
# The output files can then be used to build a bespoke Gentoo configuration
# in the docker container.

# Use 00_setup_gather_dependencies.sh to install required tools (should work for Ubuntu,
# Debian, Fedora and Arch)

# Create a folder (e.g. mkdir fedora_hpz6) and then go into the folder and run
# this script `sudo ../01_gather_info.sh`.

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

cat /proc/cpuinfo > cpuinfo.txt
dmesg > dmesg.txt
run_or_warn fastfetch.txt fastfetch
run_or_warn lshw.txt lshw
run_or_warn lsmod.txt lsmod
run_or_warn lspci.txt lspci
run_or_warn lsusb.txt lsusb
uname -a > uname.txt

# Collect SMART info for all disks
for disk in /dev/sd? /dev/sd?? /dev/nvme[0-9]n[0-9] /dev/hd?; do
    [ -b "$disk" ] || continue
    if check_tool smartctl; then
        smartctl -a "$disk" > "smartctl_$(basename "$disk").txt" 2>/dev/null || echo "smartctl failed for $disk" > "smartctl_$(basename "$disk").txt"
    else
        echo "smartctl not available" > "smartctl_$(basename "$disk").txt"
    fi
done

# CPU & hardware
run_or_warn lscpu.txt lscpu
if check_tool dmidecode; then
    dmidecode > dmidecode.txt 2>/dev/null || echo "dmidecode failed or requires root" > dmidecode.txt
else
    echo "dmidecode requires root or is not installed" > dmidecode.txt
fi

# Memory & storage
cat /proc/meminfo > meminfo.txt
free -h > free.txt
df -h > df.txt
lsblk -a > lsblk.txt
if check_tool blkid; then
    blkid > blkid.txt 2>/dev/null || echo "blkid may require root" > blkid.txt
else
    echo "blkid not available" > blkid.txt
fi
if check_tool fdisk; then
    fdisk -l > fdisk.txt 2>/dev/null || echo "fdisk may require root" > fdisk.txt
else
    echo "fdisk not available" > fdisk.txt
fi

# Sensors / temps (may need lm-sensors installed)
if check_tool sensors; then
    sensors > sensors.txt 2>/dev/null || echo "sensors failed" > sensors.txt
else
    echo "sensors not available" > sensors.txt
fi

# OS & boot
cat /etc/os-release > os-release.txt 2>/dev/null || echo "no /etc/os-release" > os-release.txt
hostnamectl > hostnamectl.txt 2>/dev/null || hostname > hostnamectl.txt
cat /etc/fstab > fstab.txt 2>/dev/null || echo "no /etc/fstab" > fstab.txt
if check_tool journalctl; then
    journalctl -b > journalctl.txt 2>/dev/null || echo "journalctl may require root or systemd" > journalctl.txt
else
    echo "journalctl may require root or systemd or is not installed" > journalctl.txt
fi

# Processes & services
ps aux > ps.txt
# Try ss first, then netstat
try_alternatives ss.txt ss -tulpn :: netstat -tulpn || echo "ss/netstat not available or require root" > ss.txt
if check_tool systemctl; then
    systemctl list-units --type=service --state=running > running_services.txt 2>/dev/null || echo "systemctl not available or requires root" > running_services.txt
else
    echo "systemctl not available" > running_services.txt
fi

# summary of missing tools
report_missing_tools
