#!/bin/sh
# This script runs during the dracut 'mount' hook

# 1. Wait for the encrypted device to appear (handled by dracut-crypt)
# /dev/mapper/enc should already be decrypted if rd.luks.uuid is used
if [ -e /dev/mapper/enc ]; then
    # 2. Ensure the mount point exists
    mkdir -p /run/overlayfs
    
    # 3. Mount your persistent partition
    # Use -o noatime,rw for better performance/longevity
    mount -t ext4 /dev/mapper/enc /run/overlayfs
    
    # 4. Prepare required subdirectories for OverlayFS
    # OverlayFS requires 'upper' and 'work' to be on the same filesystem
    mkdir -p /run/overlayfs/upper /run/overlayfs/work
    
    info "Mounted persistent overlay from /dev/mapper/enc"
else
    warn "Persistent device /dev/mapper/enc not found; falling back to tmpfs defaults"
fi
