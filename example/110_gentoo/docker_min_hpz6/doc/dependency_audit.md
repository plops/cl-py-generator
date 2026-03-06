# Dependency Size Audit

Date: 2026-03-06
Host context: `gentoo-z6-min` Docker image (`/var/lib/portage/world`)

## Largest Installed World Packages (MiB)

1. `sys-kernel/linux-firmware` ~1725 MiB
2. `sys-kernel/gentoo-sources` ~1410 MiB
3. `net-im/signal-desktop-bin` ~442 MiB
4. `sys-devel/gcc` ~357 MiB
5. `www-client/google-chrome` ~339 MiB
6. `app-editors/emacs` ~318 MiB
7. `dev-lang/rust` ~266 MiB
8. `www-client/firefox` ~243 MiB
9. `net-analyzer/wireshark` ~115 MiB
10. `dev-python/uv` ~62 MiB
11. `dev-db/postgresql` ~53 MiB
12. `app-text/mupdf` ~52 MiB
13. `dev-lisp/sbcl` ~48 MiB
14. `dev-vcs/git` ~39 MiB
15. `net-analyzer/nmap` ~24 MiB

## Recreate

Run the script on a Gentoo host/container:

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

For this image specifically, you can run directly in Docker:

```bash
docker run --rm --privileged -v "$PWD/scripts:/tmp/scripts:ro" \
  gentoo-z6-min bash -lc 'TOP_N=15 /tmp/scripts/audit_world_sizes.sh /var/lib/portage/world'
```
