#!/bin/sh
# This script runs during the dracut 'mount' hook

# 1. Wait for the encrypted device to appear (handled by dracut-crypt)
# /dev/mapper/enc should already be decrypted if rd.luks.uuid is used
if [ -e /dev/mapper/enc ]; then
    # 2. Ensure the mount point exists
    mkdir -p /mntenc
    
    # 3. Mount your persistent partition
    # Use -o noatime,rw for better performance/longevity
    mount -t ext4 /dev/mapper/enc /mntenc
    ln -s /mntenc/upper /run/overlayfs
    ln -s /mntenc/work /run/ovlwork
        
    info "Mounted persistent overlay from /dev/mapper/enc"
else
    warn "Persistent device /dev/mapper/enc not found; falling back to tmpfs defaults"
fi
