#!/usr/bin/env bash
set -euo pipefail

rotation="${1:-3}"
rotate_path="/sys/class/graphics/fbcon/rotate_all"

case "$rotation" in
  0|1|2|3)
    ;;
  *)
    echo "Usage: $0 [0|1|2|3]" >&2
    exit 1
    ;;
esac

if [[ ! -w "$rotate_path" ]]; then
  exec sudo /bin/sh -c "echo $rotation > $rotate_path"
fi

echo "$rotation" > "$rotate_path"
