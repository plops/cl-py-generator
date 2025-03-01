#!/bin/bash

# This script copies the resulting gentoo files (squashfs image and kernel) to the host

cp /gentoo.squashfs /tmp/outside/
cp /boot/vmlinuz /tmp/outside/
cp /boot/initramfs_squash_crypt-x86_64.img /tmp/outside/