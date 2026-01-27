#!/bin/bash
check() {
    # Always include this module
    return 0
}
depends() {
    # Ensure crypt and dm are available to handle /dev/mapper/enc
    echo crypt dm
    return 0
}
install() {
    # Install the hook script into the 'mount' stage
    inst_hook mount 01 "$moddir/mount-persistent.sh"
}
