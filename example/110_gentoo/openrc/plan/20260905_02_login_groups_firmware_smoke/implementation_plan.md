# Implementierungsplan: Login, Gruppen, Firmware-Trim, Cleanup und Smoke-Tests

Datum: 2026-09-05
Autor: wolpumba (wolpumba@gmail.com)
Ziel-Pfad: `/workspace/src/cl-py-generator/example/110_gentoo/openrc`
Basis: `plan/next` (Rohnotiz) + `logs/setup01_run_20260905_095049.log` (Stand `#38 1886.0`)
Vorgaenger: `plan/20260905_01_packages_and_kernel_config/` (bereits umgesetzt:
`MAKEOPTS="-j32"` in `config/make.conf` Zeile 17f, `gdb`/`dpkg`/`fontconfig`/
`gentoo-zsh-completions` in `config/world`, `freerdp[X]`/`mupdf[X opengl]` in
`config/package.use`, `>=gentoo-sources-6.19`-Maske)

## Goal

Das Gentoo-OpenRC-Image wird anhand der Rohnotiz `plan/next` gehaertet und
verkleinert, ohne Boot-, X11- oder WLAN-Funktionalitaet zu brechen: passwortloser
`kiel`-Login als Fallback, korrekte Gruppen, vollstaendig abgearbeitete
`Messages for package`-Hinweise aus dem Build-Log `095049`, Firmware-Trim auf
wifi/nvidia/amdgpu, reproduzierbarer Kernel-Pin, robuste Go-/Rust-Bereinigung und
Smoke-Tests fuer alle Kernprogramme. Ergebnis wird in `walkthrough.md` belegt.

## Success Criteria

- `kiel` kann sich ohne Passwort anmelden (leeres Shadow-Feld), dokumentiert mit
  Sicherheitshinweis.
- Build-Log enthaelt keine `Couldn't find 'virtual/rust' / 'dev-lang/go-bootstrap' /
  'dev-util/cargo-c'`-Zeilen mehr; danach laufen Smoke-Tests erfolgreich.
- Alle `Messages for package`-Abschnitte des Logs `095049` sind je als
  uebernommen, bereits-erledigt oder begruendetes Non-Goal entschieden und im
  `walkthrough.md` tabellarisch abgehakt.
- Firmware-Reduktion ist messbar (Groesse vorher/nachher) und beide Squashfs
  (NV + E14) booten mit WLAN/Grafik.
- Kein `gentoo-sources-6.18.49`-Eintrag mehr im Log, obwohl `KVER_PURE=6.18.48`
  gebaut wird.
- Smoke-Test-Skript existiert und prueft mindestens `Xorg`, `emacs`, `gcc`,
  `git`, `firefox`, `pipewire`, `iwd`, `docker`, `xfreerdp`, `mupdf`, `picocom`,
  `gdb` (plus `rustc`/`cargo` nur falls im Image).

## Context And Current Facts

- `Dockerfile` Zeile 173ff: `useradd -m -G users,wheel,audio,video,input kiel`
  plus `libvirt,kvm,qemu`-Nachtraege. `plugdev`, `pcap`, `docker` fehlen; ohne
  Passwort-Entschaerfung (`passwd -d`) kein passwortloser Login.
- `Dockerfile` Zeile 288: `emerge -C dev-lang/rust-bin virtual/rust dev-lang/go
  dev-lang/go-bootstrap dev-util/cargo-c || true` produziert im Log die drei
  `Couldn't find`-Warnungen aus `plan/next` Zeile 6ff (Atome existieren nicht).
- Squashfs-Stufen Zeile 447ff (NV) und 519ff (E14) laufen mit `-one-file-system-x`,
  melden trotzdem `//sys //proc //etc/hosts //etc/resolv.conf //dev ... ignored`
  (`plan/next` Zeile 15ff).
- `config/world` (112 Zeilen) enthaelt bereits `cryptsetup` (Z. 85), `lvm2` (Z. 87),
  `strace` (Z. 27), `fontconfig` (Z. 36); es fehlen `x11-apps/xauth`,
  `sys-auth/rtkit`, `media-plugins/alsa-plugins`.
- `config/package.mask` (2 Zeilen) maskiert nur `>=6.19`; `config/world` Zeile 90
  pinnt `<6.19`. Deshalb zog `emerge -uDN @world` `gentoo-sources-6.18.49`, obwohl
  `KVER_PURE=6.18.48` (Dockerfile Z. 30) gebaut wird (Log ~13783, `plan/next` Z. 231ff).
- `Dockerfile` Zeile 279 (`rc-update add elogind boot`) und Zeile 280ff
  (`user-runtime`, `user.kiel`, `dbus`, `iwd`) sind bereits gesetzt; die
  elogind/dbus-Meldungen (Log ~13900/~13910) sind damit Verifikations-, keine
  Bauaufgabe.
- `Dockerfile` Zeile 210ff installiert MELPA-Pakete (`magit paredit slime`) per
  `emacs --batch`; `app-emacs/emacs-common`-Hinweis (Log ~13689) betrifft nur
  site-Init, keinen Ersatz der MELPA-Schicht.
- `Dockerfile` Zeile 229 (`fc-cache -fv`) und E14-Firmware-Sammlung Zeile 492ff
  (`yellow_carp*`, `rembrandt*`, `mediatek`, `rtl_bt`, `rtl_nic`, `regulatory.db*`)
  sowie NV-Firmware per `modinfo -F firmware nvidia.ko` (Z. 435) sind die Anker
  fuer Fontconfig- und Firmware-Arbeiten.
- `plan/next` Zeile 277ff: kein PPP-Bedarf (Kernel-PPP bleibt aus). Zeile 117/144/
  152/176: Gruppen `video` (ok), `pcap`, `plugdev` fehlen. Zeile 152: DoH bleibt
  aus (`network.trr.mode=5`).

## Constraints And Non-goals

- Kein Wechsel des Gentoo-Profils und kein Display-Manager; `dwm` + `startx`/
  `.xinitrc` bleiben. Nur `XSESSION`-Default setzen.
- Kein PPP-Stack im Kernel (`CONFIG_PPP*` aus, als Non-Goal dokumentieren).
- Dracut-Optionen ausserhalb von `cryptsetup`/`lvm2` (cifs, iscsi, nfs, tpm2,
  bluetooth, rng-tools, plymouth usw., Log ~14005ff) werden nicht installiert.
- Askpass-Pakete fuer `sudo` entfallen (`wheel`-NOPASSWD besteht, Z. 140).
- `evince` (gtk-print-preview), `libevdev`-Tools, `mc.sh`-Integration, DoH-Flip und
  optionale Firefox-Helfer bleiben Non-Goals, werden aber je einzeilig begruendet.
- Firmware-Trim nur ueber `USE=savedconfig` + belegte Groessenmessung; kein
  ungetestetes Loeschen von Blobs.
- Whitelist-Squashfs nur als Untersuchung; Pflicht ist das Smoke-Test-Skript, kein
  riskantes Voll-Refactoring des Images in diesem Schritt.

## Key Decisions

1. Shadow statt PAM-Hack: `passwd -d kiel` nach `useradd` (einfach, sichtbar,
   ruecksetzbar). Alternative `passwd -u`/`usermod -p '*'` verworfen (nicht leer).
2. Cleanup per Bestand: erst `qlist -I | grep -Ei 'rust|go|cargo'` im Container,
   dann nur reale Atome deinstallieren. Alternative: Warnungen ignorieren —
   verworfen, weil sie echte Drift (falsche Atomnamen) signalisieren.
3. Squashfs-Warnungen als benign einstufen, falls `-e`-Ausschluesse nichts bringen;
   kein `-one-file-system-x` entfernen. Pseudo-`/dev`-`/proc`-`/sys`-Eintraege
   (Z. 457ff/530ff) bleiben.
4. Firmware via `linux-firmware[ savedconfig ]` + savedconfig-Datei statt
   weiterem Dockerfile-`cp`-Filters; Dockerfile-`cp`-Logik (Z. 492ff) bleibt als
   zweite Stufe fuer das E14-Image bestehen.
5. Kernel-Pin exakt `=sys-kernel/gentoo-sources-6.18.48` in `world` und
   `package.mask` (statt `<6.19`); Reproduzierbarkeit schlaegt Neuigkeit.
   Ob `6.18.49` stabil ist, wird nicht entschieden.
6. Neue Welt-Pakete minimal: `x11-apps/xauth`, `sys-auth/rtkit`,
   `media-plugins/alsa-plugins` (alle direkt vom Log gefordert). Rest via
   `package.use`/Config statt neuer Pakete.
7. Kernel-`.config` minimal erweitern: `CRYPTO_MD4=m`,
   `PKCS8_PRIVATE_KEY_PARSER=y` (iwd, Log ~13998), `I2C_CHARDEV=m` (lm-sensors,
   ~13797), `USB_PRINTER=y` oder `cups[usb]` (cups, ~14047). PPP aus.

## Recommended Approach

Zuerst Gruppen + Shadow + Runlevel-Verifikation (klein, gut testbar), dann
`Messages`-Abarbeitung in drei Bloecken (Pakete/Config, Kernel-`.config`,
eselect/sysctl/Runlevel), dann Firmware-savedconfig + Kernel-Pin, dann
Cleanup-Fix + Smoke-Skript, dann ein Build-Lauf mit Log-Scan, dann
`walkthrough.md` mit Non-Goal-Tabelle und Z6/E14-Anleitung. Keine
MELPA-Entfernung, kein Profilwechsel, kein Whitelist-Umbau in diesem Schritt.

## Work Plan

1. Login und Gruppen (`Dockerfile` ~173ff, `config/activate`, `config/start2`)
   - `passwd -d kiel` nach `useradd`; `usermod -aG plugdev,pcap,docker kiel`
     nur fuer existierende Gruppen (guard mit `getent group`, Muster wie Z. 175ff).
   - Verifikation: `grep kiel /etc/shadow`, `id kiel` im Build-Log/Smoke-Test.
2. Paket- und USE-Aenderungen (`config/world`, `config/package.use`)
   - `world` += `x11-apps/xauth`, `sys-auth/rtkit`, `media-plugins/alsa-plugins`.
   - `package.use`: `media-plugins/alsa-plugins pulseaudio` besteht (Z. 4);
     ggf. `sys-kernel/linux-firmware savedconfig` ergaenzen.
   - `dev-debug/strace` (htop-Wunsch) und `cryptsetup`/`lvm2` (dracut-Wunsch) nur
     verifizieren, nicht doppeln.
3. Kernel-Pin und `.config` (`config/world` Z. 90, `config/package.mask`,
   `config/config6.18.35`, `Dockerfile` Z. 30ff)
   - Pin auf `=sys-kernel/gentoo-sources-6.18.48` in `world` und `package.mask`
     (ersetzt `<6.19` / `>=6.19`).
   - `.config`: `CRYPTO_MD4=m`, `PKCS8_PRIVATE_KEY_PARSER=y`, `I2C_CHARDEV=m`,
     `USB_PRINTER=y` (oder `cups[usb]`-Entscheid); `make olddefconfig`-Check.
   - PPP-Symbole bewusst aus lassen.
4. eselect/sysctl/Runlevel/XSESSION (`Dockerfile` ~229/~274ff, `config/resolv.conf`,
   `~/.zshrc`-Ausstattung)
   - `eselect fontconfig enable 44-wqy-zenhei.conf` (+ `60-liberation.conf`);
     `XSESSION` via `/etc/env.d/90xsession` + `env-update`.
   - `kernel.task_delayacct = 1` in `/etc/sysctl.d/`; `~/.zshrc` (`compinit`,
     `promptinit`, `prompt gentoo`, Cache-`zstyle`) fuer `kiel` und root.
   - `elogind boot` (Z. 279) verifizieren; `nftables`/`lvm`-Runlevel bewusst
     auslassen und dokumentieren; `nullmailer --config`-Entscheidung festhalten.
5. Firmware-Trim (`Dockerfile` ~435/~492ff)
   - `linux-firmware` auf savedconfig umstellen, nur wifi/nvidia/amdgpu-Blobs
     behalten; NV- (`modinfo`-Liste) und E14-Listen (Z. 492ff) als Massstab.
   - Image-/Firmware-Groessen vorher/nachher messen.
6. Cleanup-Fix und Smoke-Tests (`Dockerfile` ~288, neu `config/smoke-test.sh`)
   - Atomliste per `qlist`-Befund korrigieren.
   - Skript mit Versions-/Startchecks (Xorg, emacs, gcc, git, firefox, pipewire,
     iwd, docker, xfreerdp, mupdf, picocom, gdb) + `id kiel` + Shadow-Check;
     als `RUN`-Stufe und/oder gegen exportierte Squashfs-Images laufen lassen.
7. Build und `walkthrough.md`
   - `./setup01_run_with_log.sh`; Log-Scan auf `Couldn't find`, `Messages for`,
     `CONFIG_*`, `6.18.49`, Gruppen/Shadow/XSESSION/fontconfig/Firmware/Groessen.
   - `walkthrough.md`: geaenderte Dateien + Begruendung, Log-Auszuege,
     Non-Goal-Tabelle, Z6/E14-Testanleitung (`@module-rebuild`-Hinweis fuer
     nvidia, WLAN-/Grafikchecks).

## Validation Plan

- `grep -E "kiel|plugdev|pcap|docker"`: `id kiel` zeigt `video plugdev pcap docker`
  (nur existierende); Shadow-Feld leer.
- `grep -c "Couldn't find 'virtual/rust|go-bootstrap|cargo-c'"` im neuen Log = 0.
- `grep "Messages for package"`-Liste des neuen Logs gegen Non-Goal-Tabelle
  abgeglichen; jede Zeile hat einen Status.
- `grep "gentoo-sources-6.18.49"` im neuen Log = 0; `eselect kernel list` zeigt
  `6.18.48`.
- `grep -E "CRYPTO_MD4|PKCS8_PRIVATE_KEY_PARSER|I2C_CHARDEV|USB_PRINTER"` in
  `.config`/Build-Log ohne Warnung; keine PPP-Symbole aktiviert.
- `eselect fontconfig list` enthaelt aktivierte `44-wqy-zenhei` (+ Liberation);
  `XSESSION` gesetzt; `sysctl kernel.task_delayacct` = 1.
- Firmware-Verzeichnisgroesse und Squashfs-Groessen vorher/nachher dokumentiert;
  E14-Image enthaelt `yellow_carp*`/`rembrandt*`/`mediatek`/`rtl_*`/
  `regulatory.db*`, NV-Image die `modinfo`-Blobs.
- Smoke-Skript laeuft gruen; Ausgabe im `walkthrough.md` zitiert.
- Hoechstes Risiko: Firmware-Trim (zu aggressiv = kein WLAN/Grafik).
  Mitigation: nur savedconfig-Whitelist aus belegten Listen, Groessenmessung und
  Boot-/Grafik-/WLAN-Check auf Z6 und E14 vor Abschluss.

## Risks / Rollback

- Zu viel Firmware entfernt: WLAN/Grafik tot. Rollback: savedconfig erweitern,
  E14-`cp`-Liste (Z. 492ff) als Netz.
- Exakter Kernel-Pin blockiert Security-Updates: bewusst akzeptiert, im Plan
  dokumentiert; Anhebung nur als eigener Folgeschritt.
- Passwortloser Login senkt physische Sicherheit: dokumentiert; Rueckbau via
  `passwd kiel` einzeilig.
- `task_delayacct`, `USB_PRINTER`, `I2C_CHARDEV` sind risikoarm (nur Monitoring/
  Drucker/Sensoren).
- Squashfs-Whitelist wird nicht umgesetzt, nur untersucht: kein Boot-Risiko aus
  diesem Schritt.

## Open Questions

- Keine. Alle Entscheidungen sind aus `plan/next` + Log `095049` ableitbar;
  DoH bleibt aus, PPP bleibt aus, Askpass/evince/libevdev-Tools bleiben Non-Goals.
