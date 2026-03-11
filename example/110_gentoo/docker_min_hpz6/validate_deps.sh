#!/bin/bash
set -e

# Setup portage config
cp /tmp/config/make.conf /etc/portage/make.conf
cp /tmp/config/package.license /etc/portage/package.license
cp /tmp/config/package.use /etc/portage/package.use
# Handle package.env and env directory if they exist in source
if [ -f /tmp/config/package.env ]; then
    cp /tmp/config/package.env /etc/portage/package.env
fi
if [ -d /tmp/config/env ]; then
    cp -r /tmp/config/env /etc/portage/env
fi

# Ensure package.accept_keywords directory exists
mkdir -p /etc/portage/package.accept_keywords
cp /tmp/config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords

# Copy world file
mkdir -p /var/lib/portage
cp /tmp/config/world /var/lib/portage/world

# Sync essential repos if needed (skip for now as we have portage snapshot)
# emerge --ask=n app-eselect/eselect-repository dev-vcs/git

echo "Validating dependencies for @world..."
# Check for llvm, mesa, rust (source)
emerge -pD @world | grep -iE 'llvm|mesa|rust' || echo "No matches found (which might be good or bad depending on what we expect)"
echo "Checking for explicit source LLVM/Clang builds..."
emerge -pD @world | grep -E '^\[.+\] llvm-core/(llvm|clang)' || echo "No explicit LLVM/Clang source builds found."

