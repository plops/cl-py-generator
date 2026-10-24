#!/bin/sh
# openrc2 ASan-Rollout: Sanitizer-Policy fuer die Initramfs-Bootphase.
# Wird per `dracut --include` nach /var/lib/dracut/hooks/pre-udev/ gelegt und
# von dracuts init VOR udev+mount-Hooks gesourct (source_hook, belegt).
#
# 1) LSAN exitcode=0: dmsquash-live-Skripte (set -e!) rufen sanitisierte
#    Werkzeuge (mount/grep/sed, Exit-Time-Leaks). Ohne das: "Could not boot".
# 2) LD_PRELOAD=libasan: Batch-01 (util-linux) liefert auch SHARED LIBS mit
#    (libblkid u.a.). systemd-udevd laedt libblkid per dlopen, ohne selbst mit
#    ASan gelinkt zu sein -> "ASan runtime does not come first" -> Daemon
#    startet nicht (rc=1, Docker- und Gast-Beweis) -> keine /dev-Nodes ->
#    189s warten -> "Could not boot". Mit Preload steht die Runtime an erster
#    Stelle, der Daemon startet sauber (rc=0). Pfad per Shell-Glob (nur
#    Builtins, frueh verfuegbar), damit kein GCC-Versions-Update bricht.
# ASan/UBSan bleiben lethal; nur Exit-Leaks melden statt abbrechen.
# Gilt dauerhaft fuer alle Batches. Wird vor switch-root wieder entfernt
# (pre-pivot-Hook 99-unset-preload.sh), damit das echte System plain bootet.
export LSAN_OPTIONS="exitcode=0"
for _f in /usr/lib/gcc/*/*/libasan.so; do
    [ -f "$_f" ] && { LD_PRELOAD="$_f"; break; }
done
export LD_PRELOAD
unset _f
