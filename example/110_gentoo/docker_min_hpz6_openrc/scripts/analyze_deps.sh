#!/bin/bash
set -e

# Run a temporary container from the gcc-only image
# Mount the config directory to apply settings
# Run emerge with --pretend --tree to show dependencies

echo "Starting dependency analysis container..."
docker run --rm -v "$(pwd)/config":/config -w /gentoo-gcc-only gentoo-gcc-only /bin/bash -c '
    set -e
    
    echo "Applying configuration..."
    # Replicate Dockerfile COPY steps
    cp /config/make.conf /etc/portage/make.conf
    
    # Handle package.accept_keywords (it is a directory in the image)
    mkdir -p /etc/portage/package.accept_keywords
    if [ -f /config/package.accept_keywords ]; then
        cp /config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
    fi
    
    cp /config/package.license /etc/portage/package.license
    cp /config/package.use /etc/portage/package.use
    cp /config/package.env /etc/portage/package.env
    
    # env is a directory in config/ but destination is /etc/portage/env
    if [ -d /config/env ]; then
        mkdir -p /etc/portage/env
        cp -r /config/env/* /etc/portage/env/
    fi
    
    cp /config/world /var/lib/portage/world
    
    echo "Updating env..."
    env-update >/dev/null
    source /etc/profile
    
    echo "Calculating dependency tree for @world..."
    # We use --pretend (-p), --verbose (-v), --tree (-t), --deep (-D)
    # We redirect output to a file inside the container, but since we want to see it,
    # we can pipe it to less or just cat it. 
    # Since we want to save it to host, we should have mounted a bind mount for output.
    # But for now, let"s just output to stdout and capture it.
    
    emerge --color=y --pretend --verbose --tree --deep @world
' | tee dependency_tree.log

echo "Analysis complete. Output saved to dependency_tree.log"
echo "You can view the tree with: less -R dependency_tree.log"
