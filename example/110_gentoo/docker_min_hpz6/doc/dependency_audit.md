# Dependency Size Audit

Date: 2026-03-05
Host context: local Gentoo system (`/var/lib/portage/world`)

## Largest Installed World Packages (MiB)

1. `sys-kernel/linux-firmware` ~1724 MiB
2. `net-im/signal-desktop-bin` ~442 MiB
3. `sys-devel/gcc` ~357 MiB
4. `www-client/google-chrome` ~339 MiB
5. `app-editors/emacs` ~318 MiB
6. `dev-lang/rust` ~245 MiB
7. `net-analyzer/wireshark` ~114 MiB
8. `app-containers/docker` ~79 MiB
9. `app-containers/docker-buildx` ~65 MiB
10. `dev-python/uv` ~62 MiB

## Recreate

Run the script:

```bash
./scripts/audit_world_sizes.sh
```

Optional:

```bash
TOP_N=20 ./scripts/audit_world_sizes.sh
./scripts/audit_world_sizes.sh /path/to/world
```

The script reads a world file, runs `qsize -m` per atom, sorts descending by
installed size, and prints a ranked table.
