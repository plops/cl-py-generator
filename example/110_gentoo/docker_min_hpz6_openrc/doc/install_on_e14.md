# Install Gentoo Image on ThinkPad E14 (using existing GRUB)

This guide describes how to install a new live Gentoo image on the ThinkPad E14 workstation by copying new artifacts and updating the GRUB entry on the secondary disk.


## 0. Identifiers for this Machine (E14)

- **Artifacts Partition (Kernel/Squashfs)**: `/dev/nvme0n1p2` (UUID `df544e10-90c0-4315-860c-92a58ec8499e`)
- **GRUB Config Partition**: `/dev/nvme1n1p2` (currently mounted at `/1p2`)
- **Persistent Partition**: `/dev/nvme0n1p4` (LUKS UUID `bbac9bb8-39d9-42fa-8d04-94610ced9839`)
- **Build Date Suffix**: `0321` (aktuell, OpenRC), `0310` (Fallback, Systemd)

> [!NOTE]
> Ab 0320 wird OpenRC anstelle von Systemd genutzt. Das squashfs-Image für das E14 heißt lokal `gentoo.squashfs_e14` und wird als `gentoo.squashfs_0321` abgelegt. Du kannst sehr alte Versionen löschen, um Platz auf der Partition zu machen, aber behalte die Version `0310` als funktionierendes Fallback.


## 1. Copy New Build Artifacts

Wir kopieren die neuen Artefakte auf die Storage-Partition (aktuell `/run/initramfs/live`).
Remount als `rw` ist ggf. nötig:

```bash
sudo mount -o remount,rw /run/initramfs/live

# Optional: Alte Images löschen (z.B. von *vor* 0310), falls der Platz knapp wird:
# sudo rm /run/initramfs/live/*_0308* 

sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260321/gentoo.squashfs_e14 /run/initramfs/live/gentoo.squashfs_0321
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260321/vmlinuz /run/initramfs/live/vmlinuz_0321
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260321/initramfs_squash_sda1-x86_64.img /run/initramfs/live/initramfs_squash_sda1-x86_64_0321.img
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260321/packages.txt /run/initramfs/live/packages_0321.txt
```


## 2. Update GRUB Configuration

Die GRUB-Konfiguration liegt auf `/dev/nvme1n1p2` (gemountet als `/1p2`).
Füge einen neuen Eintrag für die 0321-Version (OpenRC) in `/1p2/boot/grub/custom.cfg` hinzu. Lass den alten 0310-Eintrag (Systemd) dort als Fallback stehen:

```bash
cat <<'EOF' | sudo tee -a /1p2/boot/grub/custom.cfg >/dev/null

menuentry 'Gentoo Dracut (E14 persist OpenRC nvme0n1p4 0321)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    # Search for the partition containing the artifacts (nvme0n1p2)
    search --no-floppy --fs-uuid --set=root df544e10-90c0-4315-860c-92a58ec8499e

    linux /vmlinuz_0321 \
      root=live:UUID=df544e10-90c0-4315-860c-92a58ec8499e \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0321 \
      rd.live.ram=1 \
      rd.luks.uuid=bbac9bb8-39d9-42fa-8d04-94610ced9839 \
      rd.luks.name=bbac9bb8-39d9-42fa-8d04-94610ced9839=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      nvme_core.default_ps_max_latency_us=0

    initrd /initramfs_squash_sda1-x86_64_0321.img
}
EOF
```

> [!NOTE]
> The `nvme_core.default_ps_max_latency_us=0` parameter is added to prevent "Unsafe Shutdown" logs on the DRAM-less NVMe drives (ADATA/Hynix) used in this machine. It ensures the controller is ready for the shutdown signal from OpenRC/systemd.


## 4. Audio Configuration (PipeWire)

This image uses **PipeWire** as the primary sound server, replacing PulseAudio.

### OpenRC Session Bringup
On the OpenRC image, PipeWire, `pipewire-pulse`, and WirePlumber are started from the user X session. After `startx`, verify the user-session audio stack first:

```bash
ps -ef | grep -E 'pipewire|wireplumber' | grep -v grep
wpctl status
```

If those processes are missing, restart X or start them manually as user `kiel`:

```bash
pipewire >/tmp/pipewire.log 2>&1 &
pipewire-pulse >/tmp/pipewire-pulse.log 2>&1 &
wireplumber >/tmp/wireplumber.log 2>&1 &
wpctl status
```

### Diagnostic Tools
*   **Check Status**: `wpctl status` lists all active Sinks (outputs) and Sources (inputs).
*   **Control Volume (CLI)**:
    *   `wpctl set-volume <ID> <VALUE>` (e.g., `wpctl set-volume 50 0.4` for 40% volume).
    *   `wpctl set-mute <ID> 1` (mute) or `0` (unmute).
*   **Control Volume (TUI)**: `pulsemixer` provides an interactive terminal interface.
*   **Control Volume (X11)**: `pavucontrol` is the standard GUI for audio routing and levels.

### Audio Output Test
On the running system, first inspect the available sinks:

```bash
wpctl status
wpctl get-volume @DEFAULT_AUDIO_SINK@
```

Then play any known-good local media file through the default sink, for example:

```bash
mpv --no-video /path/to/test-audio-file
```

Useful fallback commands if channel routing looks wrong:

```bash
wpctl set-mute @DEFAULT_AUDIO_SINK@ 0
wpctl set-volume @DEFAULT_AUDIO_SINK@ 0.5
pulsemixer
```

### Current E14 Audio Status
Current bringup status on the OpenRC E14:

- Analog microphone input works.
- The microphone array is not exposed yet as a separate ALSA/PipeWire capture device.
- Analog speaker/headphone playback works on the Conexant `CX8070` path when accessed directly via ALSA.
- The generic `default` playback path is currently broken on the live system because ALSA is configured to hand `default` to PipeWire, while no working PipeWire user session is running yet.

Useful low-level checks:

```bash
cat /proc/asound/cards
cat /proc/asound/pcm
```

The current E14 shows:

- `card0`: HDMI/DP outputs
- `card1`: `CX8070 Analog`
- `01-00: CX8070 Analog : playback 1 : capture 1`

### Working mpv Audio Output
At the moment, audio output works reliably when `mpv` is pointed directly at the Conexant ALSA device instead of the broken `default`/PipeWire path:

```bash
mpv --no-video --ao=alsa --audio-device=alsa/sysdefault:CARD=Generic_1 /home/kiel/stage/olisun_psych/download.wav
```

This also works:

```bash
mpv --no-video --ao=alsa --audio-device=alsa/front:CARD=Generic_1,DEV=0 /home/kiel/stage/olisun_psych/download.wav
```

To list the currently visible `mpv` audio devices:

```bash
mpv --audio-device=help /home/kiel/stage/olisun_psych/download.wav
```

Interpretation of the current failure mode:

- `mpv --ao=pipewire ...` fails because no PipeWire user session is available.
- `mpv --ao=pulse ...` fails because `pipewire-pulse` is not available as a running user service.
- `mpv --ao=alsa ...` without an explicit device fails because ALSA `default` is redirected to PipeWire by `/etc/alsa/conf.d/50-pipewire.conf`.
- `mpv --ao=alsa --audio-device=alsa/sysdefault:CARD=Generic_1 ...` works because it bypasses the broken default routing and opens the analog Conexant device directly.

### Microphone Input Test
First inspect the capture devices and default source:

```bash
wpctl status
wpctl get-volume @DEFAULT_AUDIO_SOURCE@
```

Then record a short sample with PipeWire and play it back:

```bash
pw-record --target @DEFAULT_AUDIO_SOURCE@ /tmp/e14-mic.wav
mpv --no-video /tmp/e14-mic.wav
```

If the capture level is too low or muted:

```bash
wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0
wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 1.0
pulsemixer
```

### Microphone Array (ThinkPad E14 Gen 6)
The E14 Gen 6 uses an **AMD Pink Sardine (ACP 6.3)** coprocessor for its digital microphone array. 

**Troubleshooting if Mic is missing:**
1.  **Kernel Version**: Ensure you are on Kernel **6.12.7** or newer. This version includes DMI quirks specifically for the E14 Gen 6 microphone array.
2.  **Driver Check**: Run `lspci -nnk -s 05:00.5`. It should show the `snd_pci_ps` driver in use.
3.  **WirePlumber UCM Fix**: If the mic is still missing, try disabling ALSA UCM in WirePlumber.
    *   Create a file `/etc/wireplumber/wireplumber.conf.d/50-alsa-config.conf`:
        ```lua
        monitor.alsa.rules = [
          {
            matches = [
              {
                device.name = "~alsa_card.*"
              }
            ]
            actions = {
              update-props = {
                api.alsa.use-ucm = false
              }
            }
          }
        ]
        ```
    *   Restart services: 
        *   Mit Systemd: `systemctl --user restart wireplumber pipewire`
        *   Mit OpenRC: Die Prozesse (z.B. `killall wireplumber pipewire`) killen, damit sie durch den X-Session-Autostart (`gentoo-pipewire-launcher`) neu gestartet werden.

### Microphone Array Test Sequence
Use this order on the E14 so it is obvious where the failure is:

```bash
sudo /home/kiel/activate
lspci -nnk -s 05:00.5
dmesg | grep -Ei 'snd|sof|acp|pink|audio'
wpctl status
pw-cli ls Node | grep -Ei 'input|source|mic|acp|dmic'
pw-record --target @DEFAULT_AUDIO_SOURCE@ /tmp/e14-array.wav
mpv --no-video /tmp/e14-array.wav
```

Interpretation:

- If `lspci -nnk -s 05:00.5` does not show `snd_pci_ps`, the kernel/module side is still broken.
- If `snd_pci_ps` is present but `wpctl status` shows no source, the user-session PipeWire/WirePlumber stack is the next thing to fix.
- If the source exists but the recording is silent, inspect mute/gain first and then test the WirePlumber UCM workaround above.

## 5. WiFi Configuration

The system uses `iwd` (iNet wireless daemon) for backend management.

### Console / TUI (iwctl)
This is the preferred method for first-time setup in a console.
```bash
iwctl
# Inside iwctl:
station wlan0 scan
station wlan0 get-networks
station wlan0 connect "My-SSID"
# (Enter password when prompted)
exit
```

### Low-level Scanning (iw)
To just scan for nearby networks without connecting:
```bash
sudo iw dev wlan0 scan | grep SSID

# Detailed scan
sudo iw dev wlan0 scan
```

### X11 / GUI (iwgtk)
For a visual interface in `dwm`:
1.  Open a terminal or use the dmenu shortcut.
2.  Run `iwgtk`.
3.  Select your network and enter the password. This tool uses `iwd` in the background and supports password prompting in the GUI.

---
> [!TIP]
> To verify network connectivity, use `ping 8.8.8.8` or `ip addr show wlan0`. IWD stores passwords in `/var/lib/iwd/`.

## 6. OpenRC First-Boot Recovery

For the OpenRC-based `0321` image, some systems may boot without the expected keyboard layout, kernel module dependency map, GPU module, or WiFi module being activated automatically yet. If `startx` opens a screen but keyboard and mouse do not work, or if WiFi is missing until manual `modprobe`, run:

```bash
sudo /home/kiel/activate
```

Keep `/home/kiel/activate` in sync with the repo copy when iterating on bringup from the running E14:

```bash
cp /home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6_openrc/config/activate /home/kiel/activate
chmod 0755 /home/kiel/activate
```

If desktop applications still fail with loader/runtime errors after that, refresh the dynamic linker cache as well:

```bash
sudo su
. /etc/profile
ldconfig
```

This helper currently performs:

```bash
sync repo-provided OpenRC service overrides into /etc/init.d
rc-update -u
on E14: remove reverse-ssh-eu and reverse-ssh-us from default
on E14: set rc_autostart_user="NO" in /etc/rc.conf
rc-service udev start
rc-service udev-trigger start
rc-service udev-settle start
rc-service hostname start
rc-service dbus start
loadkeys /usr/share/keymaps/.../colemak/en.latin*.gz
depmod -a
modprobe amdgpu
modprobe mt7921e
modprobe snd_hda_intel
modprobe snd_pci_ps
modprobe snd_soc_ps_mach
modprobe snd_sof_amd_acp63
```

> [!NOTE]
> The keymap path is discovered dynamically from `/usr/share/keymaps`, so the exact filename may differ slightly between images.

> [!NOTE]
> This `ldconfig` step is important when booting with a persistent overlay partition. An older `/etc/ld.so.cache` can survive there and still point to runtime libraries from a previous image build.

After that, retry `startx` without changing permissions on `/dev/tty*`.

### OpenRC Services on E14

The E14 image keeps the reverse tunnel scripts installed, but does not start them automatically. This avoids unwanted outbound tunnels from the laptop while keeping the services available for manual testing:

```bash
sudo rc-service reverse-ssh-eu start
sudo rc-service reverse-ssh-us start
```

The OpenRC service overrides in this repo also replace package defaults for `tailscale` and `openvpn` so they no longer depend on the non-existent `net` aggregate service on this image.

### Login Failure: `user kiel`

If login prints `Starting user kiel ... failed to start user kiel`, the important thing to check is not stale persistent OpenRC user state first, but the login environment. On this OpenRC image there is no `elogind`, so `pam_openrc` can try to start user services without a valid `XDG_RUNTIME_DIR`. In that case `openrc-user` exits immediately.

Current workaround on the E14:

```bash
grep -n 'rc_autostart_user' /etc/rc.conf
env | grep XDG_RUNTIME_DIR
```

Expected current state:

- `rc_autostart_user="NO"` is present in `/etc/rc.conf`.
- `XDG_RUNTIME_DIR` may be unset on console login.
- PipeWire and WirePlumber are started from `~/.xinitrc` instead of OpenRC user services.

### Bringup notes from the current `0321` test

Die folgenden Punkte sind der aktuelle Zwischenstand auf dem laufenden OpenRC-E14-System und noch keine final bereinigte Anleitung:

- `hostname` war zur Laufzeit zunächst `(none)`, obwohl `/etc/hostname` bereits `e14` enthält. Ursache war, dass der OpenRC-Dienst `hostname` noch `stopped` war. `rc-service hostname start` setzt den laufenden Hostname korrekt auf `e14`.
- `emacs` und sogar `rg` konnten mit `libgcc_s.so.1 must be installed for pthread_exit to work` bzw. `error while loading shared libraries: libgcc_s.so.1` abstürzen, obwohl die Bibliothek im Image vorhanden war.
- Ursache war kein reines Docker-Build-Problem, sondern ein veralteter `ld.so.cache` auf der persistenten Overlay-Partition. Der Cache zeigte noch auf den alten GCC-16-Pfad, während das aktuelle OpenRC-Image bereits GCC-15-Runtime-Dateien ausliefert.
- Der konkrete Fix auf dem laufenden System war:
  ```bash
  sudo su
  . /etc/profile
  ldconfig
  ldconfig -p | grep libgcc_s
  emacs --version
  rg --version
  ```
- Deshalb muss dieser Schritt in der Install-/Recovery-Dokumentation stehen. Nur ein Fix im Dockerfile reicht nicht aus, wenn beim Booten ein persistenter Cache aus einem älteren Image wieder eingeblendet wird.
- `~/Downloads/Antigravity/antigravity` scheiterte zunächst mit `libnspr4.so: cannot open shared object file`. Der verifizierte Befund auf dem laufenden System war aber nicht ein kaputter Loader-Cache, sondern tatsächlich fehlende Runtime-Bibliotheken: `libnspr4.so`, `libnss3.so`, `libnssutil3.so`, `libsmime3.so` und `libcups.so.2`.
- Der minimale Live-Fix war ein gezieltes `emerge` von `dev-libs/nss` und `net-print/cups`; `dev-libs/nspr` kam dabei als Abhängigkeit mit. Danach war `ldd ~/Downloads/Antigravity/antigravity | grep 'not found'` leer.
- Für den Docker-Build wurde deshalb `dev-libs/nss` und `net-print/cups` ins World-Set aufgenommen. Um `cups` klein zu halten, reicht hier:
  ```bash
  net-print/cups -X -acl -pam -dbus -zeroconf -usb -xinetd -openssl -kerberos
  ```
- `Downloads/Handy_0.7.7_amd64.AppImage` scheiterte zuerst an fehlendem `fusermount` im `PATH`, obwohl das Kernelmodul `fuse` geladen war. Für AppImages fehlt also derzeit die Userspace-Seite von FUSE.
- Der Start per `--appimage-extract` umgeht FUSE, scheitert auf dem aktuellen System aber danach an `libstdc++.so.6: cannot open shared object file`.
- Auf dem laufenden System wurden weder `libstdc++.so.6` noch andere `libstdc++`-Dateien unter `/usr/lib` oder `/usr/lib64` gefunden. Das deutet auf eine fehlende C++-Runtime im Live-System hin.
- Das aus dem Container exportierte `packages.txt` enthält jedoch `sys-devel/gcc-15.2.1_p20260214`. Der Docker-Build hat GCC also offenbar gebaut; die fehlende `libstdc++.so.6` wirkt eher wie ein Problem des gebooteten/ausgelieferten Images als wie ein bewusst nicht gebautes Paket.

Nützliche Kommandos für die weitere Diagnose:

```bash
hostname
rc-service hostname status
ldd ~/Downloads/Antigravity/antigravity | grep -E 'nspr|not found'
ldconfig -p | grep -E 'libnspr4|libnss3|libnssutil3|libsmime3|libcups\\.so\\.2'
find /usr/lib /usr/lib64 -name 'libstdc++.so*' 2>/dev/null
ls -l /run/initramfs/live/packages_0321.txt
grep 'sys-devel/gcc' /run/initramfs/live/packages_0321.txt
```

### Temporary X11 workaround on the live system

On the currently running `0321` live system, plain `startx` can still fail with:

```text
parse_vt_settings: Cannot open /dev/tty0 (Permission denied)
```

A working temporary workaround is:

```bash
/home/kiel/startx-workaround
```

This starts the X server with elevated privileges but runs the X session itself as user `kiel`, so applications like `xterm` are no longer owned by `root`.

> [!NOTE]
> This is only a live-system workaround. The correct long-term fix is to rebuild the image with `x11-base/xorg-server` using the `suid` USE flag.
