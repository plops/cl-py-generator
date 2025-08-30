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

# ensure color variables are always defined (fixes "unbound variable" when set -u)
red="$(printf '\033[31m')"
reset="$(printf '\033[0m')"

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

    # IPv6: only redact specific IPv6 addresses present on THIS host (avoid broad regex)
    ipv6_addresses=()
    if command -v ip >/dev/null 2>&1; then
        # collect global (non-link-local) IPv6 addresses, strip prefix/zone
        while IFS= read -r a; do
            [ -n "$a" ] && ipv6_addresses+=("$a")
        done < <(ip -6 -o addr show scope global 2>/dev/null | awk '{print $4}' | sed 's#/.*##' | sed 's/%.*//g' | sort -u)
    elif command -v ifconfig >/dev/null 2>&1; then
        while IFS= read -r a; do
            [ -n "$a" ] && ipv6_addresses+=("$a")
        done < <(ifconfig 2>/dev/null | grep -oE '([0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F:]+' | sed 's/%.*//g' | sort -u)
    fi

    if [ "${#ipv6_addresses[@]}" -gt 0 ]; then
        for addr in "${ipv6_addresses[@]}"; do
            # use '#' delimiter to avoid escaping ':'; treat the found address as literal pattern
            sed -E -i "s#${addr}#[REDACTED_IPV6]#g" "$tmp" 2>/dev/null || true
        done
    else
        # No local IPv6 addresses detected — do not apply broad IPv6 redaction
        :
    fi

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

    # Concise diagnostic: show up to 5 anonymized changed lines (color placeholders red)
    if command -v diff >/dev/null 2>&1; then
        # collect added/changed lines from anonymized file (ignore diff headers like "+++")
        changes=$(diff -U0 --label "anon: $file" --label "orig: $bak" "$bak" "$file" \
            | awk '/^\+[^+]/ {sub(/^\+/, ""); print}' | sed -n '1,5p' || true)
        if [ -n "$changes" ]; then
            echo "  Changed lines (anonymized; red = redacted tokens):"
            printf '%s\n' "$changes" \
                | sed -E \
                    -e "s/(\[REDACTED_[A-Z0-9_]+\])/${red}\1${reset}/g" \
                    -e "s|/home/REDACTED|${red}/home/REDACTED${reset}|g" \
                    -e "s/\(REDACTED\)/${red}(REDACTED)${reset}/g" \
                    -e "s/(\[REDACTED_USER\]@)/${red}\1${reset}/g" \
                    -e "s/(\[REDACTED_USER\])/${red}\1${reset}/g" \
                | sed 's/^/    /'
        else
            echo "  (no changed lines detected)"
        fi
    else
        echo "  (diff not available to show changes)"
    fi

    echo "  (original saved as $bak)"
    echo ""
done

echo "Anonymization complete."
