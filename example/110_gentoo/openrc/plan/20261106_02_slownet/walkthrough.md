# Walkthrough: wlan0 absichtlich drosseln (6 kB/s down, 1 kB/s up)

Stand: 2026-11-07. Du hast einen Handy-Hotspot, SSH bricht nach
ca. 30 s bis wenigen Minuten ab, Ping läuft weiter. Wireshark zeigt
Retransmissions und Duplicate ACKs. Dieser Text nimmt dich von genau
dort mit bis zum nächsten Build.

Grundlagen stehen in [report.md](/workspace/src/cl-py-generator/example/110_gentoo/openrc/plan/20261106_02_slownet/report.md),
Rohnotizen in [wifirecon](/workspace/src/cl-py-generator/example/110_gentoo/openrc/plan/20261106_02_slownet/wifirecon).

## 1. Was du vorhattest

Die Provider-Drossel lokal nachstellen. Empfang auf 6 kB/s,
Senden auf 1 kB/s deckeln. So wird der Fehler reproduzierbar,
statt vom Handynetz abhängig zu sein.

Achtung bei den Einheiten. `tc` rechnet in kbit/s, nicht kB/s:

- 6 kB/s Empfang = ca. 48 kbit/s
- 1 kB/s Senden = ca. 8 kbit/s

Senden ist einfach, das ist die normale `root`-qdisc auf `wlan0`.
Empfang braucht einen Ingress-Policer. Genau dort klemmte es.

## 2. Der Fehler

Du hast ausgeführt:

```sh
sudo tc qdisc add dev wlan0 handle ffff: ingress
sudo tc filter add dev wlan0 parent ffff: protocol ip u32 match u32 0 0 police rate 48kbit burst 32kbit drop
```

Antwort:

```text
Error: Specified qdisc kind is unknown.
Error: Parent Qdisc doesn't exists.
We have an error talking to the kernel
```

Lies das von oben nach unten. Der erste Befehl scheitert:
Der Kernel kennt `ingress` nicht. Der zweite scheitert als Folge:
Es gibt keinen Parent `ffff:`, an den der Filter könnte.
Das ist kein Tippfehler und kein `tc`-Versionsproblem.
Dem Kernel fehlt die Unterstützung.

## 3. Warum: die Kernel-Config

Vorher stand in [config6.18.48](/workspace/src/cl-py-generator/example/110_gentoo/openrc/config/config6.18.48):

```text
# CONFIG_NET_SCH_INGRESS is not set
# CONFIG_NET_CLS_U32 is not set
# CONFIG_NET_CLS_MATCHALL is not set
# CONFIG_NET_ACT_POLICE is not set
# CONFIG_NET_ACT_MIRRED is not set
```

Nur `CONFIG_NET_CLS_ACT=y` war schon da. `BBR` und `FQ`
aus dem Report waren also drin, Ingress aber nicht.
Darum schlug dein erster Befehl fehl.

## 4. Was geändert wurde

In [config6.18.48](/workspace/src/cl-py-generator/example/110_gentoo/openrc/config/config6.18.48)
sind jetzt gesetzt, fest einkompiliert (`=y`, keine Module,
wie schon `BBR` und `FQ`):

| Option | Wofür |
|---|---|
| `CONFIG_NET_SCH_INGRESS=y` | kennt `ingress` / `clsact` als qdisc |
| `CONFIG_NET_CLS_U32=y` | dein `u32 match u32 0 0`-Filter |
| `CONFIG_NET_ACT_POLICE=y` | dein `police rate ... drop` |
| `CONFIG_NET_CLS_MATCHALL=y` | moderne Filter-Alternative zu `u32` |
| `CONFIG_NET_CLS_BASIC=y` | simple `basic`-Filter-Alternative |
| `CONFIG_NET_ACT_GACT=y` | Basis-Aktion, braucht `mirred` |
| `CONFIG_NET_ACT_MIRRED=y` | Umleitung auf `ifb` für sauberes Shaping |
| `CONFIG_NET_IFB=y` | virtuelles `ifb`-Device für Ingress-Shaping |

`CONFIG_NET_IFB` stand bisher gar nicht in der Datei und wurde
im Netzgeräteteil hinter `CONFIG_VETH` ergänzt.
`make olddefconfig` im Build sortiert und prüft das.

## 5. Dockerfile: ein Schritt statt Chunks

Der letzte Build mit Chunks war erfolgreich:
`logs/setup01_run_20261106_182536.log`, Exit-Code 0.
Die Chunk-Variante war nur Debugging, damit ein Fehler
leichter neu probiert werden kann. Das ist nicht mehr nötig.

In der [Dockerfile](/workspace/src/cl-py-generator/example/110_gentoo/openrc/Dockerfile)
steht jetzt aktiv:

```dockerfile
RUN emerge -uDN @world
```

Die 10 Chunk-Zeilen (`grep` / `split` / `xargs ... stage_00`
bis `stage_09`) sind darunter auskommentiert erhalten.
Reaktivieren geht so: die Ein-Schritt-Zeile auskommentieren,
die Chunk-Zeilen einkommentieren. Sonst nichts anfassen.

## 6. Nächster Build

Der Build nutzt `KVER_PURE=6.18.60` mit der `.config` aus
`config6.18.48` (plus `olddefconfig` und `LOCALVERSION`-Anpassung).
Bauen wie bisher:

```sh
bash setup01_run_with_log.sh
```

Das schreibt wie immer ein neues Log plus `.meta`-Datei
unter `logs/`. Erwarte roughly die Dauer des letzten Laufs
(ca. 47 Minuten). Danach Image booten. Sonst misst du
den alten Kernel.

## 7. Prüfen nach dem Reboot

Neuer Kernel aktiv?

```sh
uname -r
```

Neue Optionen drin?

```sh
zcat /proc/config.gz | grep -E 'NET_SCH_INGRESS|NET_CLS_U32|NET_ACT_POLICE|NET_IFB'
```

Erwartet: überall `=y`. Dann erst drosseln.
Senden (egress):

```sh
sudo tc qdisc replace dev wlan0 root tbf rate 8kbit burst 32kbit latency 400ms
```

Empfang (ingress):

```sh
sudo tc qdisc add dev wlan0 handle ffff: ingress
sudo tc filter add dev wlan0 parent ffff: protocol ip u32 match u32 0 0 police rate 48kbit burst 32kbit drop
```

Moderne Alternative, falls `u32` zickt:

```sh
sudo tc qdisc add dev wlan0 clsact
sudo tc filter add dev wlan0 ingress protocol ip matchall action police rate 48kbit burst 32kbit drop
```

Anzeigen und aufräumen:

```sh
tc -s qdisc show dev wlan0
tc -s filter show dev wlan0 ingress
sudo tc qdisc del dev wlan0 root 2>/dev/null
sudo tc qdisc del dev wlan0 ingress 2>/dev/null
```

## 8. Was das bringt, was nicht

Mit 8 kbit/s up und 48 kbit/s down wird SSH zäh.
Das ist normal und genau der Sinn der Übung:
Du siehst dann reproduzierbar, was der Provider-Policer
mit `cubics` Sägezahn und den `TCPLostRetransmit`-Zählern
aus dem Report anrichtet. `BBR` + `fq`, SSH-Keepalive
oder `mosh` und notfalls USB-Tethering bleiben die
eigentlichen Gegenmittel. Die Drossel heilt nichts,
sie macht den Fehler nur stellbar.
