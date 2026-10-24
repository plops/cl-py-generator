#!/bin/sh
# openrc2 ASan-Rollout: LD_PRELOAD vor switch-root entfernen.
# Wird per `dracut --include` nach /var/lib/dracut/hooks/pre-pivot/ gelegt
# und VOR dem Wechsel ins echte System gesourct. Der Preload ist nur fuer
# die Initramfs-Phase (udevd + dlopen auf Batch-01-Libs, s. 00-lsan.sh).
# Das echte System (OpenRC, PID 1, Dienste) bootet bewusst plain; Batch-
# Werkzeuge bleiben ueber ihre eigene Linkung sanitisiert.
unset LD_PRELOAD
