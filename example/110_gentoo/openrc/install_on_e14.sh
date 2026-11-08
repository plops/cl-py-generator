#!/bin/bash
#
# Installiert einen neuen Gentoo-OpenRC-Build auf dem ThinkPad E14.
#
# Vorgehen (vgl. doc/install_on_e14.md):
#   1. Artefakt-Partition (per UUID) mounten, Platz pruefen.
#   2. Neuen Build nach /<MMDD>/ kopieren (squashfs, Kernel, Initramfs, Pakete, Microcode).
#   3. GRUB-Eintrag in grub/custom.cfg auf der GRUB-Partition (per UUID) ergaenzen.
#
# Das Script loescht NICHTS. Reicht der Platz nicht, schlaegt es exakte
# rm-Kommandos fuer alte Installationsverzeichnisse vor und bricht ab.
#
# Aufruf auf dem Gentoo-Host (als root):
#   sudo bash install_on_e14.sh
#   sudo bash install_on_e14.sh --src /pfad/zu/gentoo-z6-min-openrc_20261107 --dry-run
#
set -euo pipefail

# --- Feste Kennungen dieser Maschine (E14) ---
ARTIFACTS_UUID="df544e10-90c0-4315-860c-92a58ec8499e"
GRUB_UUID="cfb582a8-a460-40bc-bdf8-c760c5ecebb6"
LUKS_UUID="bbac9bb8-39d9-42fa-8d04-94610ced9839"

# Sicherheitspuffer zusaetzlich zur reinen Dateigroesse (btrfs-Metadaten).
SAFETY_BYTES=$((64 * 1024 * 1024))

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR=""
SUFFIX=""
DRY_RUN=0
ARTIFACTS_MNT="/1p2"
GRUB_MNT="/mnt"

info() { printf '%s\n' "$*"; }
ok()   { printf '[OK] %s\n' "$*"; }
fail() { printf '[FEHLER] %s\n' "$*" >&2; exit 1; }

usage() {
    info "Installiert einen neuen Gentoo-OpenRC-Build auf dem ThinkPad E14."
    info ""
    info "Aufruf auf dem Gentoo-Host (als root):"
    info "  sudo bash install_on_e14.sh"
    info "  sudo bash install_on_e14.sh --src /pfad/zu/gentoo-z6-min-openrc_20261107 --dry-run"
    info ""
    info "Optionen:"
    info "  --src DIR           Build-Verzeichnis (Default: <script-dir>/gentoo-z6-min-openrc_*)"
    info "  --suffix MMDD       Zielverzeichnis, z.B. 1107 (Default: aus --src abgeleitet)"
    info "  --artifacts-mnt DIR Mountpunkt der Artefakt-Partition (Default: /1p2)"
    info "  --grub-mnt DIR      Mountpunkt der GRUB-Partition (Default: /mnt)"
    info "  --dry-run           Nur Pruefungen, keine Aenderungen"
    info "  -h, --help          Diese Hilfe"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --src)           SRC_DIR="$2"; shift 2 ;;
        --suffix)        SUFFIX="$2"; shift 2 ;;
        --artifacts-mnt) ARTIFACTS_MNT="$2"; shift 2 ;;
        --grub-mnt)      GRUB_MNT="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               fail "Unbekannte Option: $1 (siehe --help)" ;;
    esac
done

# --- Precondition: root ---
[ "$(id -u)" -eq 0 ] || fail "Bitte als root ausfuehren: sudo bash $0"

# --- Precondition: Werkzeuge ---
for tool in blkid findmnt mount df du cp awk grep; do
    command -v "$tool" >/dev/null 2>&1 || fail "Benoetigtes Werkzeug fehlt: $tool"
done

# --- Quelle bestimmen ---
if [ -z "$SRC_DIR" ]; then
    # Neuestes Build-Verzeichnis neben dem Script nehmen.
    SRC_DIR="$(ls -d "$SCRIPT_DIR"/gentoo-z6-min-openrc_* 2>/dev/null | sort | tail -n 1 || true)"
    [ -n "$SRC_DIR" ] || fail "Kein Build-Verzeichnis neben dem Script gefunden. --src angeben."
fi
[ -d "$SRC_DIR" ] || fail "Quellverzeichnis fehlt: $SRC_DIR"

# --- Suffix bestimmen (MMDD aus ..._YYYYMMDD) ---
if [ -z "$SUFFIX" ]; then
    STAMP="$(basename "$SRC_DIR" | grep -o '[0-9]\{8\}$' || true)"
    [ -n "$STAMP" ] || fail "Kein Datum im Verzeichnisnamen, --suffix MMDD angeben."
    SUFFIX="${STAMP:4:4}"
fi
[[ "$SUFFIX" =~ ^[0-9]{4}$ ]] || fail "Ungueltiges Suffix (MMDD erwartet): $SUFFIX"

# --- Precondition: Quelldateien ---
SQUASHFS_SRC="$SRC_DIR/gentoo.squashfs_e14"
VMLINUZ_SRC="$SRC_DIR/vmlinuz"
INITRAMFS_SRC="$SRC_DIR/initramfs_squash_sda1-x86_64.img"
for f in "$SQUASHFS_SRC" "$VMLINUZ_SRC" "$INITRAMFS_SRC" \
         "$SRC_DIR/packages.txt" "$SRC_DIR/packages.tsv"; do
    [ -s "$f" ] || fail "Quelldatei fehlt oder ist leer: $f"
done
ok "Quelle: $SRC_DIR -> Suffix $SUFFIX"

# Aktuell gebooteer Build (darf nie zum Loeschen vorgeschlagen werden).
BOOTED="$(grep -o 'rd\.live\.dir=/[0-9]\{4\}/' /proc/cmdline 2>/dev/null | grep -o '[0-9]\{4\}' || true)"
[ -n "$BOOTED" ] && info "Aktuell gebootet: $BOOTED (bleibt unangetastet)"

ensure_mounted_rw() {
    local uuid="$1" mnt="$2" label="$3"
    local dev existing opts
    dev="$(blkid -U "$uuid" 2>/dev/null || true)"
    [ -n "$dev" ] || fail "$label: keine Partition mit UUID $uuid gefunden (blkid)."
    existing="$(findmnt -rn -S "UUID=$uuid" -o TARGET 2>/dev/null | head -n 1 || true)"
    if [ -z "$existing" ]; then
        [ "$DRY_RUN" -eq 1 ] && fail "$label: dry-run, $dev ist nicht gemountet (wuerde nach $mnt mounten)."
        mkdir -p "$mnt"
        mount "$dev" "$mnt" || fail "$label: mount $dev $mnt schlug fehl."
        existing="$mnt"
    fi
    opts="$(findmnt -rn -M "$existing" -o OPTIONS 2>/dev/null || true)"
    if [[ ",$opts," == *",ro,"* ]]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            # stderr: stdout wird als Mountpunkt per $(...) aufgefangen.
            info "[dry-run] $existing ist read-only, wuerde remounten: mount -o remount,rw $existing" >&2
        else
            mount -o remount,rw "$existing" \
                || fail "$label: $existing ist read-only. Manuell: mount -o remount,rw $existing"
        fi
    fi
    if [ "$DRY_RUN" -eq 0 ]; then
        touch "$existing/.writetest.$$" 2>/dev/null \
            || fail "$label: $existing ist nicht beschreibbar."
        rm -f "$existing/.writetest.$$"
    fi
    printf '%s' "$existing"
}

# --- Artefakt-Partition ---
A_MNT="$(ensure_mounted_rw "$ARTIFACTS_UUID" "$ARTIFACTS_MNT" "Artefakt-Partition")"
ok "Artefakt-Partition: $A_MNT"
TARGET="$A_MNT/$SUFFIX"

# --- Microcode-Quelle: neuer Build oder neuester vorhandener Satz ---
UCODE_SRC=""
if [ -s "$SRC_DIR/amd-uc.img" ]; then
    UCODE_SRC="$SRC_DIR/amd-uc.img"
else
    for d in $(ls -d "$A_MNT"/[0-9][0-9][0-9][0-9] 2>/dev/null | sort -r); do
        [ "$(basename "$d")" = "$SUFFIX" ] && continue
        if [ -s "$d/amd-uc.img" ]; then UCODE_SRC="$d/amd-uc.img"; break; fi
    done
fi
[ -n "$UCODE_SRC" ] || fail "Kein amd-uc.img gefunden (weder in $SRC_DIR noch in $A_MNT/<MMDD>/)."
ok "Microcode-Quelle: $UCODE_SRC"

# --- Precondition: Platz ---
need_total=0
for f in "$SQUASHFS_SRC" "$VMLINUZ_SRC" "$INITRAMFS_SRC" \
         "$SRC_DIR/packages.txt" "$SRC_DIR/packages.tsv" "$UCODE_SRC"; do
    need_total=$((need_total + $(stat -c %s "$f")))
done
# Bei Re-Install belegt das Ziel bereits Platz, der wiederverwendet wird.
existing_target=0
[ -d "$TARGET" ] && existing_target="$(du -sb "$TARGET" | awk '{print $1}')"
need=$((need_total - existing_target + SAFETY_BYTES))
[ "$need" -lt 0 ] && need=0
avail="$(df -B1 --output=avail "$A_MNT" | tail -n 1 | tr -d ' ')"
info "Benoetigt: $need_total Bytes (+ $SAFETY_BYTES Puffer), frei: $avail Bytes"

if [ "$avail" -lt "$need" ]; then
    missing=$((need - avail))
    info ""
    info "[FEHLER] Zu wenig Platz auf $A_MNT (fehlen: $missing Bytes)."
    info "Vorhandene Installationsverzeichnisse (alt zuerst):"
    freed=0
    printf '  %-10s %12s %12s\n' "Verzeichnis" "Groesse" "Kumuliert"
    for d in $(ls -d "$A_MNT"/[0-9][0-9][0-9][0-9] 2>/dev/null | sort); do
        name="$(basename "$d")"
        [ "$name" = "$SUFFIX" ] && continue
        size="$(du -sb "$d" | awk '{print $1}')"
        if [ "$name" = "$BOOTED" ]; then
            printf '  %-10s %12s   (aktuell gebootet, NICHT loeschen)\n' "$name" "$size"
            continue
        fi
        freed=$((freed + size))
        printf '  %-10s %12s %12s\n' "$name" "$size" "$freed"
    done
    info ""
    info "Bitte manuell Platz schaffen, danach Script erneut starten. Exakte Kommandos:"
    freed=0
    for d in $(ls -d "$A_MNT"/[0-9][0-9][0-9][0-9] 2>/dev/null | sort); do
        name="$(basename "$d")"
        [ "$name" = "$SUFFIX" ] && continue
        [ "$name" = "$BOOTED" ] && continue
        size="$(du -sb "$d" | awk '{print $1}')"
        freed=$((freed + size))
        if [ "$freed" -ge "$missing" ]; then
            info "  rm -rf \"$d\"   # <-- ab hier reicht der Platz wieder"
            break
        else
            info "  rm -rf \"$d\""
        fi
    done
    info "Danach ggf. auch den zugehoerigen GRUB-Eintrag aus grub/custom.cfg entfernen."
    exit 1
fi
ok "Platz reicht aus"

# --- GRUB-Partition (Precondition, auch fuer dry-run relevant) ---
G_MNT="$(ensure_mounted_rw "$GRUB_UUID" "$GRUB_MNT" "GRUB-Partition")"
CUSTOM="$G_MNT/grub/custom.cfg"
[ -f "$CUSTOM" ] || fail "GRUB-Konfiguration fehlt: $CUSTOM (falsche Partition? Inhalt: $(ls "$G_MNT" 2>/dev/null | tr '\n' ' '))"
ok "GRUB-Konfiguration: $CUSTOM"

if [ "$DRY_RUN" -eq 1 ]; then
    info "dry-run: alle Pruefungen bestanden, keine Aenderungen vorgenommen."
    exit 0
fi

# --- Kopieren ---
mkdir -p "$TARGET"
cp -v "$SQUASHFS_SRC" "$TARGET/gentoo.squashfs"
cp -v "$VMLINUZ_SRC" "$TARGET/vmlinuz"
cp -v "$INITRAMFS_SRC" "$TARGET/initramfs.img"
cp -v "$SRC_DIR/packages.txt" "$TARGET/packages.txt"
cp -v "$SRC_DIR/packages.tsv" "$TARGET/packages.tsv"
cp -v "$UCODE_SRC" "$TARGET/amd-uc.img"

# --- Kopien verifizieren ---
verify_one() {
    local src="$1" dst="$2"
    [ -s "$dst" ] || fail "Kopie fehlt: $dst"
    [ "$(stat -c %s "$src")" = "$(stat -c %s "$dst")" ] \
        || fail "Groessenabweichung: $dst"
    if command -v sha256sum >/dev/null 2>&1; then
        [ "$(sha256sum <"$src" | awk '{print $1}')" = "$(sha256sum <"$dst" | awk '{print $1}')" ] \
            || fail "Pruefsummenabweichung: $dst"
    else
        cmp -s "$src" "$dst" || fail "Inhaltsabweichung: $dst"
    fi
}
verify_one "$SQUASHFS_SRC" "$TARGET/gentoo.squashfs"
verify_one "$VMLINUZ_SRC" "$TARGET/vmlinuz"
verify_one "$INITRAMFS_SRC" "$TARGET/initramfs.img"
verify_one "$UCODE_SRC" "$TARGET/amd-uc.img"
ok "Kopien verifiziert: $TARGET"

# --- GRUB-Eintrag schreiben (Partition wurde oben bereits geprueft) ---
TITLE="Gentoo Dracut (E14 persist OpenRC nvme0n1p4 $SUFFIX)"
BACKUP="$CUSTOM.bak-$(date +%Y%m%d-%H%M%S)"
cp -v "$CUSTOM" "$BACKUP"

# Vorhandenen Eintrag gleichen Suffix entfernen (idempotenter Re-Run), Rest behalten.
awk -v title="$TITLE" '
    /^menuentry / { skip = (index($0, title) > 0) }
    !skip { print }
    skip && /^\}/ { skip = 0 }
' "$CUSTOM" >"$CUSTOM.new"
mv "$CUSTOM.new" "$CUSTOM"

cat >>"$CUSTOM" <<EOF
menuentry '$TITLE' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    # Search for the partition containing the artifacts
    search --no-floppy --fs-uuid --set=root $ARTIFACTS_UUID

    linux /$SUFFIX/vmlinuz \\
      root=live:UUID=$ARTIFACTS_UUID \\
      rd.live.dir=/$SUFFIX/ \\
      rd.live.squashimg=gentoo.squashfs \\
      rd.live.ram=1 \\
      rd.luks.uuid=$LUKS_UUID \\
      rd.luks.name=$LUKS_UUID=enc \\
      rd.overlay=/dev/mapper/enc:persistent \\
      rd.live.overlay.overlayfs=1 \\
      nvme_core.default_ps_max_latency_us=0

    initrd /$SUFFIX/amd-uc.img /$SUFFIX/initramfs.img
}
EOF
ok "GRUB-Eintrag '$TITLE' geschrieben (Backup: $BACKUP)"

# --- Abschlussbericht ---
info ""
info "=== Fertig ==="
ls -lh "$TARGET/"
df -h "$A_MNT" "$G_MNT"
grep -n "^menuentry" "$CUSTOM"
