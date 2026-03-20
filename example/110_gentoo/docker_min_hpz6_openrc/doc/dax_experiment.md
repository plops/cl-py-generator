# Extended Fearfile System: DAX & Execute-In-Place (XIP)

Dieses Dokument fasst das Konzept für die Nutzung von DAX (Direct Access) in unserem Gento-basierten Minimal-Container zusammen.

## Konzept: Execute-In-Place (XIP)
Normalerweise werden ausführbare Dateien (Binaries) und Bibliotheken vom Dateisystem erst in den RAM (Linux Page Cache) kopiert, bevor die CPU sie ausführt. Dadurch liegen die Programmdateien doppelt im Speicher (1x im Dateisystem-Image im RAM, 1x im Cache). 
**DAX** eliminiert diesen Page Cache komplett. Über `mmap()` greift die CPU direkt und ohne Umwege auf die Speicherblocks der ausführbaren Datei zu.

## Die Hybrid-Root Architektur
Um Speicherplatz zu sparen, aber gleichzeitig XIP für Binaries zu nutzen, spalten wir unser Image in zwei Dateisysteme auf, die zur Laufzeit per **OverlayFS** übereinandergelegt werden:

1. **SquashFS (`gentoo.squashfs`)**: Ein hochkomprimiertes Nur-Lese-Dateisystem. Hierin liegen alle Konfigurationsdateien, Logs, Assets und Dokumentationen. Die Ordner mit Binaries (`/bin`, `/sbin`, `/usr/bin`, `/usr/lib`, etc.) sind absichtlich **ausgeschlossen**.
2. **Ext4 DAX RAM-Disk (`gentoo.ext4`)**: Ein unkomprimiertes Ext4-Dateisystem, welches exakt diese exkludierten Binärdateien und Bibliotheken enthält. Es wird strikt mit einer *4KB-Blockgröße (`mke2fs -b 4096`)* formatiert, was eine zwingende Voraussetzung für DAX ist.

## Kernel-Voraussetzungen
DAX funktioniert nur auf PMEM (Persistent Memory) oder expliziten Block-RAM-Geräten. In unserem Kernel sind deshalb folgende Module aktiviert:
- `CONFIG_LIBNVDIMM`, `CONFIG_BLK_DEV_PMEM` (Basis für `/dev/pmem0`)
- `CONFIG_FS_DAX`, `CONFIG_DAX` (Filesystem-Level Direct Access)
- `CONFIG_ZONE_DEVICE` (notwendig für PMEM Memory Mapping)

## Boot-Ablauf & Aktivierung (Experimentell)

### 1. PMEM via Grub reservieren
Damit der Kernel einen Teil des normalen RAMs als "Persistent Memory" (/dev/pmem0) erkennt, muss die Kernel-Cmdline (in GRUB) angepasst werden:
```bash
memmap=4G!12G  # Reserviert 4GB ab der 12-GB-Marke
```

### 2. Module laden und PMEM für DAX vorbereiten
```bash
modprobe libnvdimm pmem dax_pmem
```
*(Optional kann das Device mit `ndctl create-namespace -f -e namespace0.0 --mode=fsdax --map=dev` reinitialisiert werden, um Metadaten zu optimieren).*

### 3. Dateisysteme mounten und ineinanderfügen (Overlay)
Das in RAM geladene `gentoo.ext4` wird blockweise per `dd` über das `/dev/pmem0` Image gelegt (bzw. formatiert und per `rsync` befüllt).
Der entscheidende Mount-Befehl ist:
```bash
mount -t ext4 -o dax /dev/pmem0 /mnt/xip_storage
mount -t squashfs /rootfs.squashfs /mnt/ro_base

# Kombinieren
mount -t overlay overlay -o lowerdir=/mnt/xip_storage:/mnt/ro_base /newroot
```

### 4. Besonderheit: Der OverlayFS "Trap"
Es ist bekannt, dass einige Kernel-Versionen das DAX-Flag "verlieren", wenn Dateien über das OverlayFS aufgerufen werden. Dadurch landen sie wieder ungewollt im Page-Cache. 
Um das zu umgehen (oder zu überprüfen), empfiehlt sich:
1. Den Aufruf über den absoluten Pfad des DAX-Mounts zu testen: `/mnt/xip_storage/usr/bin/bash`
2. Den Dynamic Linker mit `LD_LIBRARY_PATH=/mnt/xip_storage/lib64:/mnt/xip_storage/usr/lib64` direkt auf das DAX-Gerät zeigen zu lassen.

## Verifizierung
Nach dem Booten lässt sich prüfen, ob das Setup erfolgreich ist:
1. **ndctl**: `ndctl list -R` sollte den Reservierten PMEM auflisten.
2. **Mount**: `cat /proc/mounts | grep dax` sollte `/mnt/xip_storage` anzeigen.
3. **Filefrag**: `filefrag -v /usr/bin/bash` listet die Extents der Festplatte auf. Unter `flags` muss hier idealerweise `ext4_dax` stehen. Wenn `delalloc` steht, funktioniert XIP nicht.
4. **RAM-Test**: Wenn ein großes Programm ausgeführt wird, darf der "buff/cache"-Wert bei "free -m" nicht exponentiell ansteigen.
