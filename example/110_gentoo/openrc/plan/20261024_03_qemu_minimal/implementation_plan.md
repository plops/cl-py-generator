# Implementierungsplan: Minimales QEMU-Testimage (openrc2, ohne ASan)

Datum: 2026-10-24 · Autor: wolpumba (wolpumba@gmail.com)
Plan-Verzeichnis: `openrc/plan/20261024_03_qemu_minimal/` (`prompt.txt`, dieser Plan,
später `walkthrough.md`, `task.md`)
Arbeitsverzeichnis: `/workspace/src/cl-py-generator/example/110_gentoo/openrc2` (neu)

## Ziel

Neubau-Zeit von ~2h (Voll-Image `openrc/`, e14 + hpz6) auf einen Bruchteil drücken
und Boot in QEMU ermöglichen. Mittel: 112 → 15 Welt-Pakete, ein einziges Squashfs,
kein NVIDIA-/E14-Sonderimage, kein MELPA-Emacs, QEMU-Kernel per Fragment.
`openrc/` bleibt unverändert; der Hardware-Pfad (Release + ASan/LSan/UBSan-
Verifikation) bleibt offen.

## Kontext-Dateien

- `../openrc/Dockerfile` (605 Zeilen) — Oracle für alle Build-Stufen.
- `../openrc/config/world` (112 Einträge), `make.conf`, `package.use`,
  `package.mask`, `package.env`, `env/{low-mem,lto-gcc}` — Oracle-Konfiguration.
- `../openrc/config/config6.18.48` (232 kB) — Oracle-Kernel-`.config`, liegt als
  `openrc2/config/config-base` in 1:1-Kopie.
- `../openrc/gentoo-z6-min-openrc_20261024/packages.tsv` (753 Pakete mit Größen) —
  Beleg für jede Streichung (s. Tabelle unten).
- Frühere QEMU-Versuche: `../docker_min/setup04_create_qemu.sh`,
  `../docker_min/setup05_run_qemu.sh`, `../docker_min_hpz6_openrc/setup06_create_qemu_dual_lower.sh`.
- `../openrc/plan/next.md` — Enthält u. a. „i really need vfat support (to develop
  pi pico 2)": VFAT bleibt deshalb im QEMU-Kernel und `sys-fs/dosfstools` im Image.

## Streich-Plan (mit Größenbeleg aus packages.tsv, Summe Voll-Image ~8,4 GiB)

| Gestrichen | Größe | Grund |
|---|---|---|
| `sys-kernel/linux-firmware` | ~1841 MiB | QEMU/virtio braucht keine HW-Firmware |
| `x11-drivers/nvidia-drivers` | ~1083 MiB | Kein NVIDIA in QEMU |
| `llvm-core/llvm` | ~395 MiB | Fällt implizit weg (kein mesa/nvidia/rust im Tree); `LLVM_TARGETS="X86"`, Build-Check per `qlist` |
| `net-im/signal-desktop-bin` | ~364 MiB | Desktop-Chat, kein Test-Bedarf |
| `www-client/firefox-bin` | ~325 MiB | Kein Browser im Konsolen-Image |
| `app-editors/emacs` (+ MELPA-Run) | ~175 MiB | Kein Editor im Testimage |
| `net-analyzer/wireshark` + Rest | ~119 MiB+ | Kein Netzwerk-Forensik-Bedarf |
| `media-libs/mesa`, `media-video/ffmpeg`, `dev-qt/qtbase`, `x11-libs/gtk+` | ~150 MiB | Kein Grafik-Stack ohne X11 |
| `app-text/mupdf`, `sys-firmware/sof-firmware`, `media-video/pipewire/wireplumber` | ~110 MiB | Kein Desktop/Audio |
| `dev-debug/gdb`, `dev-debug/strace` | klein | Auf Wunsch gestrichen; Debugging ggf. später via Sanitizer-Logs |
| `app-shells/zsh`, `app-misc/tmux`, `sys-process/htop`, `sys-devel/mold`, … | klein | Komfort-Tools, vorerst draußen |

Welt nachher (15 Einträge): `sudo`, `sysklogd`, `gentoolkit` (für `qlist`-Report),
`git`, `picocom` (Pico-2-Seriell), `dhcpcd`, `pciutils`, `usbutils`,
`cryptsetup` + `lvm2` (dracut-Module, HW-Pfad-kompatibel), `dosfstools`,
`e2fsprogs`, `squashfs-tools` (wird im Build gebraucht), `dracut`,
`gentoo-sources`-Pin. Neuzugänge ggü. Voll-Image (in `tools/check-world-diff.sh`
dokumentiert): nur `dosfstools`, `e2fsprogs`.

## Architektur-Entscheidungen

1. **Neuer Ordner `openrc2/` statt `openrc/` zu ändern:** Das Voll-Image ist die
   produktive Basis für e14/hpz6; Experimente dürfen es nicht anfassen.
2. **Gleicher Stage3-/Portage-Stand und gleicher `KVER_PURE`-Pin:** Damit bleibt der
   Weg zurück auf Hardware ein reiner Paket-Delta, kein Versions-Sprung.
3. **Kernel per Fragment, nicht per `defconfig`:** `config-base` ist Oracle;
   `kernel-qemu.fragment` dokumentiert jede Abweichung (Virtio/Seriell rein,
   DRM_AMDGPU/NOUVEAU/I915/SND/WLAN/BT raus, VFAT drin).
4. **Kein LUKS/LVM im QEMU-Image:** `setup04` baut eine simple vfat-Disk mit GRUB;
   `setup05 --direct` bootet Kernel/Initramfs sogar ganz ohne Disk-Image.
5. **Kein ASan in diesem Schritt:** Erst beweisen, dass minimal baut und bootet;
   Sanitizer (`config/env/sanitizers`) kommen als zweiter Schritt nur für kleine
   Testpakete (mehr RAM/Zeit pro Build).
6. **Build-Zeit-Garantie im Dockerfile:** Eigene `RUN`-Stufe scheitert, falls
   LLVM/NVIDIA/Mesa/Emacs/Firefox/Firmware doch im Tree landen.

## Git-Strategie (Conventional Commits)

- `feat(openrc2): add minimal world without X, firmware and desktop`
- `feat(openrc2): add qemu kernel fragment and keep vfat`
- `feat(openrc2): add qemu create/run scripts and guest smoke test`
- `docs(plan): add qemu-minimal prompt, implementation plan and walkthrough`

## Risiken

- `emerge -uDN @world` könnte über System-Abhängigkeiten doch LLVM ziehen
  (Mitigation: `LLVM_TARGETS="X86"` + Build-Check-Stufe).
- `emerge =gentoo-sources-6.18.57` vs. `config-base` (Name 6.18.48): gleiche
  Übernahme wie `openrc/` (sed-Patch + `olddefconfig`); Drift wird im Build-Log
  sichtbar.
- QEMU ohne `/dev/kvm` fällt auf TCG zurück (langsam, aber bootfähig).

## Verifikation

1. `bash -n` auf allen Skripten — erledigt, grün.
2. `tools/check-world-diff.sh` — grün (112 → 15, nur `dosfstools`/`e2fsprogs` neu).
3. Docker-Build `gentoo-qemu-min-openrc` (ohne ASan) — nächster Schritt, auf
   Host mit genug Platte (~2h-Budget des Voll-Builds entfällt großteils, da fast
   alles aus Binärpaketen kommt; Kernel `-j32` bleibt).
4. QEMU-Boot + `config/smoke-test.sh` im Gast (Ergebnis in `walkthrough.md`).
