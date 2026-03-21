# Install Gentoo Image on ThinkPad E14 (using existing GRUB)

This guide describes how to install a new live Gentoo image on the ThinkPad E14 workstation by copying new artifacts and updating the GRUB entry on the secondary disk.


## 0. Identifiers for this Machine (E14)

- **Artifacts Partition (Kernel/Squashfs)**: `/dev/nvme0n1p2` (UUID `df544e10-90c0-4315-860c-92a58ec8499e`)
- **GRUB Config Partition**: `/dev/nvme1n1p2` (currently mounted at `/1p2`)
- **Persistent Partition**: `/dev/nvme0n1p4` (LUKS UUID `bbac9bb8-39d9-42fa-8d04-94610ced9839`)
- **Build Date Suffix**: `0320` (aktuell, OpenRC), `0310` (Fallback, Systemd)

> [!NOTE]
> Ab 0320 wird OpenRC anstelle von Systemd genutzt. Das squashfs-Image für das E14 heißt lokal `gentoo.squashfs_e14` und wird als `gentoo.squashfs_0320` abgelegt. Du kannst sehr alte Versionen löschen, um Platz auf der Partition zu machen, aber behalte die Version `0310` als funktionierendes Fallback.


## 1. Copy New Build Artifacts

Wir kopieren die neuen Artefakte auf die Storage-Partition (aktuell `/run/initramfs/live`).
Remount als `rw` ist ggf. nötig:

```bash
sudo mount -o remount,rw /run/initramfs/live

# Optional: Alte Images löschen (z.B. von *vor* 0310), falls der Platz knapp wird:
# sudo rm /run/initramfs/live/*_0308* 

sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260320/gentoo.squashfs_e14 /run/initramfs/live/gentoo.squashfs_0320
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260320/vmlinuz /run/initramfs/live/vmlinuz_0320
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260320/initramfs_squash_sda1-x86_64.img /run/initramfs/live/initramfs_squash_sda1-x86_64_0320.img
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260320/packages.txt /run/initramfs/live/packages_0320.txt
```


## 2. Update GRUB Configuration

Die GRUB-Konfiguration liegt auf `/dev/nvme1n1p2` (gemountet als `/1p2`).
Füge einen neuen Eintrag für die 0320-Version (OpenRC) in `/1p2/boot/grub/custom.cfg` hinzu. Lass den alten 0310-Eintrag (Systemd) dort als Fallback stehen:

```bash
cat <<'EOF' | sudo tee -a /1p2/boot/grub/custom.cfg >/dev/null

menuentry 'Gentoo Dracut (E14 persist OpenRC nvme0n1p4 0320)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    # Search for the partition containing the artifacts (nvme0n1p2)
    search --no-floppy --fs-uuid --set=root df544e10-90c0-4315-860c-92a58ec8499e

    linux /vmlinuz_0320 \
      root=live:UUID=df544e10-90c0-4315-860c-92a58ec8499e \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0320 \
      rd.live.ram=1 \
      rd.luks.uuid=bbac9bb8-39d9-42fa-8d04-94610ced9839 \
      rd.luks.name=bbac9bb8-39d9-42fa-8d04-94610ced9839=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      nvme_core.default_ps_max_latency_us=0

    initrd /initramfs_squash_sda1-x86_64_0320.img
}
EOF
```

> [!NOTE]
> The `nvme_core.default_ps_max_latency_us=0` parameter is added to prevent "Unsafe Shutdown" logs on the DRAM-less NVMe drives (ADATA/Hynix) used in this machine. It ensures the controller is ready for the shutdown signal from OpenRC/systemd.


## 4. Audio Configuration (PipeWire)

This image uses **PipeWire** as the primary sound server, replacing PulseAudio.

### Diagnostic Tools
*   **Check Status**: `wpctl status` lists all active Sinks (outputs) and Sources (inputs).
*   **Control Volume (CLI)**:
    *   `wpctl set-volume <ID> <VALUE>` (e.g., `wpctl set-volume 50 0.4` for 40% volume).
    *   `wpctl set-mute <ID> 1` (mute) or `0` (unmute).
*   **Control Volume (TUI)**: `pulsemixer` provides an interactive terminal interface.
*   **Control Volume (X11)**: `pavucontrol` is the standard GUI for audio routing and levels.

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
