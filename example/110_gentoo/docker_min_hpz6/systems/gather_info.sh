#!/bin/bash

cat /proc/cpuinfo > cpuinfo.txt
dmesg > dmesg.txt
fastfetch > fastfetch.txt
lshw > lshw.txt
lsmod > lsmod.txt
lspci > lspci.txt
lsusb > lsusb.txt
uname -a > uname.txt

# Collect SMART info for all disks
for disk in /dev/sd? /dev/sd?? /dev/nvme[0-9]n[0-9] /dev/hd?; do
    [ -b "$disk" ] || continue
    smartctl -a "$disk" > "smartctl_$(basename "$disk").txt" 2>/dev/null || echo "smartctl failed for $disk" > "smartctl_$(basename "$disk").txt"
done

# Additional useful info to collect:

# CPU & hardware
lscpu > lscpu.txt
dmidecode > dmidecode.txt 2>/dev/null || echo "dmidecode requires root" > dmidecode.txt

# Memory & storage
cat /proc/meminfo > meminfo.txt
free -h > free.txt
df -h > df.txt
lsblk -a > lsblk.txt
blkid > blkid.txt 2>/dev/null || echo "blkid may require root" > blkid.txt
fdisk -l > fdisk.txt 2>/dev/null || echo "fdisk may require root" > fdisk.txt

# Sensors / temps (may need lm-sensors installed)
sensors > sensors.txt 2>/dev/null || echo "sensors not available" > sensors.txt

# OS & boot
cat /etc/os-release > os-release.txt 2>/dev/null || echo "no /etc/os-release" > os-release.txt
hostnamectl > hostnamectl.txt 2>/dev/null || hostname > hostnamectl.txt
cat /etc/fstab > fstab.txt 2>/dev/null || echo "no /etc/fstab" > fstab.txt
journalctl -b > journalctl.txt 2>/dev/null || echo "journalctl may require root or systemd" > journalctl.txt

# Processes & services
ps aux > ps.txt
ss -tulpn > ss.txt 2>/dev/null || netstat -tulpn > ss.txt 2>/dev/null || echo "ss/netstat not available or require root" > ss.txt
systemctl list-units --type=service --state=running > running_services.txt 2>/dev/null || echo "systemctl not available" > running_services.txt

# Network
ip addr show > ip_addr.txt
ip route show > ip_route.txt

# User environment & cron
crontab -l > crontab.txt 2>/dev/null || echo "no crontab or permission denied" > crontab.txt
env > env.txt

# Package list (try common package managers)
if command -v dpkg-query >/dev/null 2>&1; then
    dpkg -l > packages.txt 2>/dev/null || echo "dpkg-query failed" > packages.txt
elif command -v rpm >/dev/null 2>&1; then
    rpm -qa > packages.txt 2>/dev/null || echo "rpm failed" > packages.txt
elif command -v pacman >/dev/null 2>&1; then
    pacman -Q > packages.txt 2>/dev/null || echo "pacman failed" > packages.txt
else
    echo "no known package manager found" > packages.txt
fi

# permissions note (optional): some outputs above require root to be complete.
# End of additions
