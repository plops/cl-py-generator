# openrc2 — Minimales Gentoo-OpenRC-Testsystem für QEMU

Konsolen-Testimage ohne X11, ohne Hardware-Treiber (kein NVIDIA/AMDGPU/Mesa),
ohne Emacs/Firefox/Chrome, ohne WLAN-/Audio-/Drucker-Stacks. Zweck: schneller
Neubau (statt ~2h Voll-Build) und Boot in QEMU. Ohne ASan gebaut; Sanitizer
kommen später als Verifikationsbuilds dazu.

Das Vollsystem für e14/hpz6-Hardware liegt in `../openrc/` und bleibt unverändert.

## Layout

| Datei | Zweck |
|---|---|
| `Dockerfile` | Minimal-Image: gleiche Stage3-Basis und gleicher `KVER_PURE`-Pin wie `../openrc/`, aber nur EIN QEMU-Squashfs, kein MELPA-Emacs, kein NVIDIA-Image |
| `config/world` | 15 Pakete (statt 112): Boot-, Netzwerk- und Test-Werkzeuge |
| `config/make.conf` | Wie `../openrc/`, aber `USE` ohne X/Wayland, `VIDEO_CARDS="fbdev vesa"` |
| `config/package.use` | Minimal (lvm, elogind/dbus, squashfs-tools) |
| `config/package.mask` | Gleicher `gentoo-sources`-Pin wie Voll-Image |
| `config/package.env` | Nur `low-mem` für gcc |
| `config/env/sanitizers` | `-fsanitize=address,undefined` für Verifikationsbuilds (ASan inkl. LSan); per `package.env` aktiv für `net-dialup/picocom` |
| `config/config-base` | 1:1-Oracle-Kopie von `../openrc/config/config6.18.48` |
| `config/kernel-qemu.fragment` | Abweichungen davon: Virtio + serielle Konsole rein, Grafik/WLAN/Sound raus, VFAT drin |
| `config/smoke-test.sh` | Gast-seitiger Smoke-Test (im QEMU-Gast als root laufen lassen) |
| `config/mount-overlayfs.sh` | Kopie von `../openrc/` (dracut-Modul) |
| `setup01_build_image.sh` | Baut `gentoo-qemu-min-openrc` |
| `setup01_run_with_log.sh` | Wie `../openrc/`: Build mit Zeitstempel-Log unter `logs/` |
| `setup03_copy_from_container.sh` | Exportiert Artefakte nach `/dev/shm/gentoo-qemu-min-openrc_<datum>/` |
| `copy_files.sh` | Läuft im Container (s. oben) |
| `setup04_create_qemu.sh` | Erzeugt `qemu/sda1.img` (vfat, GRUB, Kernel+Initramfs+Squashfs) |
| `setup05_run_qemu.sh` | Startet QEMU (`-nographic`, KVM wenn vorhanden, sonst TCG); `--direct <dir>` ohne Disk-Image |
| `tools/check-world-diff.sh` | Prüft: world-mini vs. world-voll, keine verbotenen Schwergewichte |

## Ablauf

```sh
cd openrc2
./setup01_run_with_log.sh          # Image bauen (ohne ASan)
./setup03_copy_from_container.sh   # Artefakte nach /dev/shm holen
./setup04_create_qemu.sh           # Boot-Disk erzeugen (braucht loop-Geräte)
./setup05_run_qemu.sh              # Starten; Login als kiel (passwortlos)
# im Gast:
bash /root/smoke-test.sh  # (Datei aus config/ in den Gast kopieren)
```

## Weg zurück auf Hardware (e14/hpz6)

- Release-Builds: weiter aus `../openrc/` (alle Treiber/Firmware/Desktop).
- Verifikationsbuilds: `config/env/sanitizers` ist per `package.env`-Eintrag
  auf `net-dialup/picocom` angewendet (kleinstes C-Paket im Tree; ASan enthält
  auf Linux LSan, UBSan via `undefined`; wirksam ab nächstem Build). Im Gast
  verifiziert: LSan meldet Test-Leak (`123 byte(s) leaked`), UBSan meldet
  `signed integer overflow`. Weitere Pakete nach demselben Muster dazu.
