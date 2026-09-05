# Implementierungsplan: Gentoo OpenRC Pakete, Kernel-Konfiguration und Image-Optimierungen

Datum: 2026-09-05  
Autor: wolpumba  (wolpumba@gmail.com) 
Ziel-Pfad: `/workspace/src/cl-py-generator/example/110_gentoo/openrc`

---

## 1. Übersicht und Zielsetzung

Das bestehende Gentoo OpenRC-Image (`example/110_gentoo/openrc`) erzeugt Live-Squashfs-Images
für zwei Zielsysteme:
1. **HP Z6 Workstation** (mit NVIDIA/CUDA-Modulen und -Firmware: `gentoo.squashfs_nv`)
2. **ThinkPad E14** (AMD Rembrandt / Yellow Carp APU ohne NVIDIA: `gentoo.squashfs_e14`)

Ziel dieses Plans ist es:
1. Fehlende Netzwerk-, Netfilter-, Bridge- und IPVS-Kernel-Optionen in `config/config6.18.35`
   zu aktivieren, um Laufzeitwarnungen von Docker/runc und nftables zu beheben.
2. Die Portage-Parallelität in `config/make.conf` auf 32 Threads (`MAKEOPTS="-j32"`) zu setzen.
3. Fehlende Pakete (`picocom`, `gdb`, `dpkg`, `app-shells/gentoo-zsh-completions`) in `config/world` aufzunehmen.
4. `xfreerdp` (`net-misc/freerdp`) und `mupdf` (`app-text/mupdf`) mit GUI-Support (`USE="X opengl"`)
   nutzbar zu machen.
5. Die D-Bus- und elogind-Session-Infrastruktur unter OpenRC zu konsolidieren.
6. Die AMDGPU- und Hardware-Unterstützung für das ThinkPad E14 zu verifizieren.
7. Hinweise aus dem Build-Log (`logs/setup01_run_20260905_072631.log`) sowie Best Practices
   aus `/workspace/src/cl-cl-generator/example/05_dockerfile_meta/source01/examples/01_gentoo/` zu übernehmen.

---

## 2. Ist-Zustand und Analyse

### 2.1 Kernel-Konfigurations-Warnungen (`logs/setup01_run_20260905_072631.log`)
Im aktuellen Build warnten `app-containers/runc-1.4.3` (Zeile 463) und `net-libs/libnftnl-1.3.1` (Zeile 759)
über fehlende Kernel-Symbole:
```text
CONFIG_BRIDGE: is not set when it should be.
CONFIG_BRIDGE_NETFILTER: is not set when it should be.
CONFIG_IP_NF_FILTER: is not set when it should be.
CONFIG_IP_NF_TARGET_MASQUERADE: is not set when it should be.
CONFIG_NETFILTER_XT_MATCH_ADDRTYPE: is not set when it should be.
CONFIG_NETFILTER_XT_MATCH_COMMENT: is not set when it should be.
CONFIG_NETFILTER_XT_MATCH_CONNTRACK: is not set when it should be.
CONFIG_NETFILTER_XT_MATCH_IPVS: is not set when it should be.
CONFIG_IP_NF_NAT: is not set when it should be.
CONFIG_NF_NAT: is not set when it should be.
CONFIG_RT_GROUP_SCHED: is not set when it should be.
CONFIG_IP_NF_TARGET_REDIRECT: is not set when it should be.
CONFIG_IP_VS: is not set when it should be.
CONFIG_IP_VS_NFCT: is not set when it should be.
CONFIG_IP_VS_PROTO_TCP: is not set when it should be.
CONFIG_IP_VS_PROTO_UDP: is not set when it should be.
CONFIG_IP_VS_RR: is not set when it should be.
CONFIG_NF_TABLES: is not set when it should be.
```
In `config/config6.18.35` sind diese Optionen aktuell deaktiviert (`# CONFIG_BRIDGE is not set`, etc.).

### 2.2 Fehlende Thread-Parallelität in `config/make.conf`
In `config/make.conf` ist aktuell weder `MAKEOPTS` noch `EMERGE_DEFAULT_OPTS` gesetzt.
Portage baut Ebuilds daher mit nur einem CPU-Core, obwohl auf dem Build-Host 32 Threads bereitstehen.

### 2.3 Fehlende / unvollständige Pakete
- **`picocom`**: fehlt in `config/world` (`net-dialout/picocom`).
- **`gdb`**: fehlt in `config/world` (`dev-debug/gdb`).
- **`dpkg`**: fehlt in `config/world` (`app-arch/dpkg`).
- **`xfreerdp`**: `net-misc/freerdp` ist in `config/world`, wurde aber im Build mit
  `USE="client ffmpeg fuse icu -X ..."` gebaut. Das X11-Programm `xfreerdp` wird nur mit `USE="X"` gebaut.
- **`mupdf`**: `app-text/mupdf` ist in `config/world`, wurde aber mit `USE="... -X ... -opengl"` gebaut.
  Dadurch fehlt das interaktive Viewer-Binary (`mupdf` / `mupdf-gl`).
- **`amdgpu`**: Im Kernel ist `CONFIG_DRM_AMDGPU=m` bereits gesetzt. In `Dockerfile` werden
  Firmware-Dateien für E14 (`yellow_carp`, `rembrandt`, `mediatek`, `rtl_bt`, `rtl_nic`) kopiert.
  Es muss sichergestellt werden, dass keine Lücken bei Firmware oder X11-Mesa-Treibern (`radeonsi`) bestehen. Ich habe ein image bereits getestet. Xorg hat gestartet, Graphik sollte also schon gehen.
- **`elogind mit D-Bus`**:
  `USE="... dbus elogind policykit udev ..."` ist in `make.conf` vorhanden. In `Dockerfile` wird
  `user-runtime` und `user.kiel` eingerichtet, aber der Standard-D-Bus-Session-Bus und elogind-Sitzungen
  sind noch nicht vollständig nahtlos verzahnt (siehe `plan/20260610_01_elogin/dbus-elogind-openrc-plan.md`).
  Der Wechsel auf das Gentoo-Profil `default/linux/amd64/23.0/no-multilib/desktop` bzw. die konsistente
  Einbindung von `sys-auth/elogind` in den Runlevel (`rc-update add elogind boot`) muss evaluiert werden.

Es gibt mit Xorg probleme mit dem touchpad siehe Xorg.0.log 

### 2.4 Log-Hinweise und Versions-Drift
1. **`sys-kernel/gentoo-sources` Drift**:
   In `Dockerfile` (Zeile 34) wird `=sys-kernel/gentoo-sources-6.18.48` installiert und kompiliert.
   Später bei `emerge -uDN @world` zieht `config/world` (Zeile 85: `sys-kernel/gentoo-sources`)
   automatisch die neuere Version `7.2.3` nach. Dies verschwendet Build-Zeit und Speicherplatz.
   Lösung: In `config/world` auf `=sys-kernel/gentoo-sources-6.18*` pinnen oder in `package.mask` maskieren.
2. **`media-libs/fontconfig`**:
   Das Log meldet `Skipping fontcache update (media-libs/fontconfig not installed)`.
   In `Dockerfile` steht `RUN fc-cache -fv`. `media-libs/fontconfig` sollte explizit in `config/world`
   geführt werden.
3. **`app-shells/gentoo-zsh-completions`**:
   Das Log empfiehlt das Paket für Portage-Completions in Zsh.
4. **`docker-buildx`**:
   Ist in `config/world` vorhanden. Sicherstellen, dass das CLI-Plugin in `/usr/libexec/docker/cli-plugins`
   von Docker gefunden wird.

---

## 3. Geplante Änderungen

### 3.1 `config/config6.18.35`
Aktivieren der benötigten Netzwerk-, Bridge- und Netfilter-Module:
```ini
CONFIG_BRIDGE=m
CONFIG_BRIDGE_NETFILTER=m
CONFIG_NETFILTER_FAMILY_BRIDGE=y
CONFIG_NF_CONNTRACK=m
CONFIG_NF_NAT=m
CONFIG_NF_TABLES=m
CONFIG_NF_TABLES_INET=y
CONFIG_NFT_CT=m
CONFIG_NFT_MASQ=m
CONFIG_NFT_NAT=m
CONFIG_IP_NF_IPTABLES=m
CONFIG_IP_NF_FILTER=m
CONFIG_IP_NF_TARGET_MASQUERADE=m
CONFIG_IP_NF_TARGET_REDIRECT=m
CONFIG_IP_NF_NAT=m
CONFIG_NETFILTER_XT_MATCH_ADDRTYPE=m
CONFIG_NETFILTER_XT_MATCH_COMMENT=m
CONFIG_NETFILTER_XT_MATCH_CONNTRACK=m
CONFIG_NETFILTER_XT_MATCH_IPVS=m
CONFIG_IP_VS=m
CONFIG_IP_VS_NFCT=y
CONFIG_IP_VS_PROTO_TCP=y
CONFIG_IP_VS_PROTO_UDP=y
CONFIG_IP_VS_RR=m
CONFIG_RT_GROUP_SCHED=y
```

### 3.2 `config/make.conf`
Hinzufügen von Parallelitäts-Optionen und Optimierungen:
```ini
MAKEOPTS="-j32"
EMERGE_DEFAULT_OPTS="--jobs=32 --load-average=32"
```
USE-Flags überprüfen:
Globales `X` ergänzen oder gezielt in `package.use` freischalten.

### 3.3 `config/world`
Ergänzen der neuen Pakete:
```text
net-dialout/picocom
dev-debug/gdb
app-arch/dpkg
app-shells/gentoo-zsh-completions
media-libs/fontconfig
```
Pin für gentoo-sources:
```text
<sys-kernel/gentoo-sources-6.19
```
bzw. `=sys-kernel/gentoo-sources-6.18.48`

### 3.4 `config/package.use`
Ergänzen von GUI- und X11-Support:
```ini
net-misc/freerdp X alsa pulseaudio ffmpeg
app-text/mupdf X opengl javascript
sys-auth/elogind pam policykit
sys-auth/polkit elogind
sys-apps/dbus elogind
```

### 3.5 `config/package.env` und `config/env`
Übernahme von Best Practices aus `cl-cl-generator/example/05_dockerfile_meta/source01/examples/01_gentoo/`:
- `low-mem` (`MAKEOPTS="-j8"`) für speicherhungrige Pakete (Rust, GCC, LLVM).
- `lto-gcc` für ausgewählte rechenintensive CLI-Tools.

### 3.6 `Dockerfile`
- Profil-Prüfung (`RUN eselect profile set ...`).
- Elogind OpenRC-Dienst einrichten:
  `rc-update add elogind boot`
- nftables OpenRC-Dienst vorbereiten:
  `rc-update add nftables default` (optional / falls gewünscht)
- Überprüfung der E14-Firmware-Kopierbefehle für amdgpu und WLAN/Bluetooth. (Ich denke graphik geht schon)

---

## 4. Test- und Verifikationsschritte

1. **Konfigurationsprüfung**:
   - Syntax von `make.conf`, `package.use`, `world` validieren.
   - Kernel `.config` mit `make olddefconfig` oder im Test-Container prüfen.
2. **Build-Test**:
   - Ausführen von `./setup01_run_with_log.sh`.
   - Überprüfung des neuen Logs:
     - Werden alle 32 Threads genutzt?
     - Sind die `CONFIG_*`-Warnungen bei `runc` und `libnftnl` verschwunden?
     - Wurden `picocom`, `gdb`, `dpkg`, `xfreerdp`, `mupdf` erfolgreich gebaut?
     - Wurde die unerwünschte Installation von `gentoo-sources-7.2.3` verhindert?
3. **Abschluss**:
   - Erstellung des `walkthrough.md` mit konkreten Log-Auszügen, Artefaktgrößen
     und Handlungsanweisungen für E14 und HP Z6.
