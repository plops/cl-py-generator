#!/bin/bash
# Anonymize gathered system files (IPs and user names).
#
# Purpose:
#  - Scan gathered text files (default: *.txt and common gather outputs) and
#    redact sensitive tokens while preserving remaining content and structure.
#  - Redacted items:
#      * IPv4 addresses -> [REDACTED_IP]
#      * IPv6-like addresses -> [REDACTED_IPV6]
#      * MAC addresses -> [REDACTED_MAC]
#      * Home paths (/home/USERNAME) -> /home/REDACTED
#      * UID(...) occurrences: uid=1000(USER) -> uid=1000(REDACTED)
#      * Common usernames as whole words -> [REDACTED_USER]
#  - The script makes a .bak copy of each processed file and overwrites the original.
#  - Prints a short diagnostic summary for each file (counts and sample lines).
#
# Usage:
#   ./03_anonymize_files.sh              # scan default files in current directory
#   ./03_anonymize_files.sh file1 file2  # scan only the listed files
#
set -euo pipefail

# Files to process by default (gather outputs)
default_files=(cpuinfo.txt uname.txt lscpu.txt lsmod.txt lspci.txt lsusb.txt lshw.txt dmidecode.txt meminfo.txt free.txt lsblk_devices.txt lsblk_all.txt blkid.txt df.txt dmesg.txt fdisk.txt fstab.txt os-release.txt hostnamectl.txt fastfetch.txt sensors.txt smartctl_*.txt)

# If arguments are provided, use them, otherwise use defaults that exist
if [ "$#" -gt 0 ]; then
    files=("$@")
else
    files=()
    for p in "${default_files[@]}"; do
        for f in $p; do
            [ -f "$f" ] && files+=("$f")
        done
    done
fi

if [ "${#files[@]}" -eq 0 ]; then
    echo "No files found to anonymize."
    exit 0
fi

# patterns and helpers
ipv4_regex='([0-9]{1,3}\.){3}[0-9]{1,3}'
ipv6_regex='([0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f:]{2,}'
mac_regex='([0-9A-Fa-f]{2}([:-])){5}[0-9A-Fa-f]{2}'
# common system usernames to redact (additional names will be caught by home-path and uid patterns)
common_users='root|admin|ubuntu|user|pi|debian|centos|ec2-user|azureuser|git'

# backup and process each file
for file in "${files[@]}"; do
    [ -f "$file" ] || continue
    bak="${file}.bak"
    cp -a -- "$file" "$bak"
    echo "Processing $file (backup -> $bak)..."

    # Count matches before
    ipv4_before=$(grep -Eo "$ipv4_regex" "$bak" | wc -l || true)
    ipv6_before=$(grep -E -o "$ipv6_regex" "$bak" | wc -l || true)
    mac_before=$(grep -Eo "$mac_regex" "$bak" | wc -l || true)
    home_before=$(grep -Eo '/home/[^/[:space:]]+' "$bak" | wc -l || true)
    uid_before=$(grep -Eo 'uid=[0-9]+\([[:alnum:]_@.-]+\)' "$bak" | wc -l || true)
    users_before=$(grep -Eo -i "\b($common_users)\b" "$bak" | wc -l || true)

    # Perform redactions in a pipeline, writing to a temp file
    tmp="$(mktemp)"
    awk '{
        print $0
    }' "$bak" > "$tmp"

    # IPv4
    sed -E -i "s/\\b$ipv4_regex\\b/[REDACTED_IP]/g" "$tmp" 2>/dev/null || true
    # IPv6 (simple heuristic)
    sed -E -i "s/$ipv6_regex/[REDACTED_IPV6]/g" "$tmp" 2>/dev/null || true
    # MAC addresses
    sed -E -i "s/$mac_regex/[REDACTED_MAC]/g" "$tmp" 2>/dev/null || true
    # home paths -> /home/REDACTED
    sed -E -i "s#/home/[^/[:space:]]+#/home/REDACTED#g" "$tmp" 2>/dev/null || true
    # uid=NNN(NAME) -> uid=NNN(REDACTED)
    sed -E -i "s/(uid=[0-9]+)\([[:alnum:]_@.-]+\)/\\1(REDACTED)/g" "$tmp" 2>/dev/null || true
    # common usernames as whole words -> [REDACTED_USER]
    sed -E -i "s/\\b($common_users)\\b/[REDACTED_USER]/gi" "$tmp" 2>/dev/null || true
    # username@host patterns -> [REDACTED_USER]@
    sed -E -i "s/([[:alnum:]_@.-]+)@([[:alnum:]_.-]+)/[REDACTED_USER]@\\2/g" "$tmp" 2>/dev/null || true

    # Move temp back to original file
    mv -f -- "$tmp" "$file"

    # Count matches after
    ipv4_after=$(grep -Eo "$ipv4_regex" "$file" | wc -l || true)
    ipv6_after=$(grep -E -o "$ipv6_regex" "$file" | wc -l || true)
    mac_after=$(grep -Eo "$mac_regex" "$file" | wc -l || true)
    home_after=$(grep -Eo '/home/[^/[:space:]]+' "$file" | wc -l || true)
    uid_after=$(grep -Eo 'uid=[0-9]+\([[:alnum:]_@.-]+\)' "$file" | wc -l || true)
    users_after=$(grep -Eo -i "\b($common_users)\b" "$file" | wc -l || true)

    # Diagnostic summary
    echo "  IPv4 addresses: removed $((ipv4_before - ipv4_after)) (before: $ipv4_before, after: $ipv4_after)"
    echo "  IPv6-like addresses: removed $((ipv6_before - ipv6_after)) (before: $ipv6_before, after: $ipv6_after)"
    echo "  MAC addresses: removed $((mac_before - mac_after)) (before: $mac_before, after: $mac_after)"
    echo "  /home/... paths: removed $((home_before - home_after)) (before: $home_before, after: $home_after)"
    echo "  uid(NAME) patterns: removed $((uid_before - uid_after)) (before: $uid_before, after: $uid_after)"
    echo "  common usernames: removed $((users_before - users_after)) (before: $users_before, after: $users_after)"

    # Show up to 5 example changed lines (diff-like)
    if command -v diff >/dev/null 2>&1; then
        echo "  Example changes (up to 5):"
        diff -u --label "orig: $bak" --label "anon: $file" "$bak" "$file" | sed -n '1,200p' | sed -n '1,200p' | head -n 50 || true
    fi

    echo "  (original saved as $bak)"
    echo ""
done

echo "Anonymization complete."
