#!/bin/bash
# Build Docker image and stop after GCC is built

# Find the line number of the last RUN/ADD/COPY command that installs gcc
gcc_layer=$(grep -n -i 'install.*gcc' Dockerfile | tail -1 | cut -d: -f1)

if [ -z "$gcc_layer" ]; then
  echo "GCC install step not found in Dockerfile."
  exit 1
fi

# Add 1 to include the next line (if needed)
let stop_line=gcc_layer+1

# Build up to the GCC layer using a temporary Dockerfile
head -n "$stop_line" Dockerfile > Dockerfile.gcc
echo -e "\n# Stopping after GCC build\n" >> Dockerfile.gcc

docker build -f Dockerfile.gcc -t gentoo-gcc-stop .

echo "Docker build stopped after GCC was built."
