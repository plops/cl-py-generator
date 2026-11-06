# Slownet-Report: SSH-Abbrüche über Handy-Hotspot

Stand: 2026-11-06, Gentoo, Kernel 6.18.60-dist

## 1. Worum geht es

Die Internetverbindung läuft über einen Handy-Hotspot (WLAN-Tethering).
Das WLAN selbst steht stabil, trotzdem brechen SSH-Sitzungen zum Server
nach ca. 30 s bis wenigen Minuten ab. Ping läuft dabei weiter, nur TCP
wirkt zäh. Wireshark zeigt viele Retransmissions und Duplicate ACKs.

Dieser Report fasst zusammen, was geändert, wie geprüft und was gefunden
wurde. Alle Beispiele sind zensiert: `<CLIENT-IP>`, `<SERVER-IP>`,
`<AP-MAC>`, `<SSID>`, `<USER>`.

## 2. Was geändert wurde

Zwei Kernel-Optionen wurden in `config/config6.18.48` eingeschaltet
(beide fest einkompiliert, `=y`, keine Module):

```text
CONFIG_TCP_CONG_BBR=y
CONFIG_NET_SCH_FQ=y
```

Hintergrund aus der Analyse:

- CUBIC fährt bei Verlust immer wieder zu schnell hoch, der
  Provider-Policer verwirft dann Bursts. Ergebnis: Sägezahn mit Retransmits.
- BBR schätzt die Bottleneck-Rate statt auf Verlust zu warten und
  benimmt sich auf gedrosselten Leitungen ruhiger.
- `fq` (Fair Queueing) ist die passende Warteschlange dazu, damit
  Pacing sauber arbeitet.

Hinweis: `CONFIG_DEFAULT_TCP_CONG` steht weiter auf `cubic`.
BBR ist also verfügbar, wird aber erst nach explizitem Umschalten aktiv.
Das ist Absicht: Altsysteme starten unverändert.

## 3. Wie geprüft wurde (Kommandos)

### 3.1 Läuft der neue Kernel?

```sh
uname -r
```

Muss zur neu gebauten Version passen. Sonst misst man den alten Kernel.

### 3.2 Ist BBR / fq einkompiliert?

```sh
zcat /proc/config.gz | grep -E 'TCP_CONG_BBR|NET_SCH_FQ|DEFAULT_TCP_CONG'
```

Erwartet:

```text
CONFIG_TCP_CONG_BBR=y
CONFIG_NET_SCH_FQ=y
CONFIG_NET_SCH_FQ_CODEL=y
CONFIG_DEFAULT_TCP_CONG="cubic"
```

Da beides `=y` ist, zeigt `lsmod` nichts. Das ist normal, kein Fehler.

### 3.3 Ist BBR global aktiv und verfügbar?

```sh
sysctl net.ipv4.tcp_available_congestion_control
sysctl net.ipv4.tcp_congestion_control
```

Beispiel nach Umschaltung:

```text
net.ipv4.tcp_available_congestion_control = reno bbr cubic
net.ipv4.tcp_congestion_control = bbr
```

Umschalten (gilt bis Reboot, nur für neue Verbindungen):

```sh
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

Wichtig: Bestehende Sockets behalten den alten Algorithmus.
SSH-Sitzungen daher schließen und neu aufbauen.

### 3.4 Welche Warteschlange hängt am Interface?

```sh
tc qdisc show dev wlan0
tc -s qdisc show dev wlan0
```

Vorher: `noqueue`. Nachher z. B.:

```sh
sudo tc qdisc replace dev wlan0 root fq
```

`Sent`, `throttled`, `flows` und `backlog` in der `-s`-Ausgabe zeigen,
ob Verkehr durchläuft und Pacing greift.

### 3.5 Welchen Algorithmus nutzt eine Verbindung wirklich?

```sh
ss -tin
ss -tin dst <SERVER-IP>
```

Alte Sitzungen zeigen `cubic`, neue nach der Umschaltung:

```text
bbr:(bw:...,mrtt:...,pacing_gain:...,cwnd_gain:...)
```

Dazu `rtt`, `cwnd`, `pacing_rate`, `delivery_rate`, `retrans:x/y`,
`bytes_retrans` vs. `bytes_sent`, `rcv_ooopack`, `minrtt`.

### 3.6 Wie viel Verlust sieht der Stack?

Kumulierte Zähler, immer als Differenz lesen:

```sh
nstat | grep -E 'TcpOutSegs|TcpRetransSegs|TcpInSegs'
nstat | grep -Ei 'retrans|loss' > /tmp/vor.txt
# 60 s normal arbeiten
sleep 60; nstat | grep -Ei 'retrans|loss' > /tmp/nach.txt
diff /tmp/vor.txt /tmp/nach.txt
```

Wichtige Felder:

- `TcpRetransSegs` vs. `TcpOutSegs`: Anteil der Wiederholungen.
- `TCPLostRetransmit`: auch Wiederholungen gehen verloren.
- `TCPLossFailures`: Timeout-Recovery gescheitert (Verbindungsabriss).
- `TCPFastRetrans`, `TCPSlowStartRetrans`: woher die Wiederholungen kommen.
- `TCPLossProbes` / `TCPLossProbeRecovery`: Tail-Loss-Probes, meist harmlos.

Detailansicht:

```sh
cat /proc/net/netstat | tr ' ' '\n' | grep -i retrans
```

### 3.7 Funkstrecke prüfen

```sh
iw dev wlan0 link
sudo dmesg | tail -30
ping -M do -s 1472 -c 4 8.8.8.8
ping -M do -s 1400 -c 4 8.8.8.8
```

Beispiel (zensiert):

```text
Connected to <AP-MAC> (on wlan0)
    SSID: <SSID>
    freq: 2417.0
    signal: -50 dBm
    rx bitrate: 104.0 MBit/s MCS 13
    tx bitrate: 130.0 MBit/s MCS 15
```

MTU-Test: Wenn 1472 klemmt und 1400 geht, MTU auf 1400 senken.

## 4. Was gefunden wurde

1. **Kernel-Seite ok.** `zcat /proc/config.gz` zeigt `BBR=y` und `FQ=y`.
   `sysctl` meldet `reno bbr cubic`, aktiv ist `bbr`.

2. **BBR greift auf neuen Verbindungen.** `ss -tin` zeigt
   `bbr:(bw:...)` statt `cubic`. Geschätzte Bottleneck-Rate je nach
   Sitzung ca. 30–1000 kbit/s, typisch um 35–55 kbit/s. `pacing_rate`
   liegt in gleicher Höhe. `minrtt` stabil bei ca. 37 ms. Kein
   Bufferbloat, Pacing arbeitet.

3. **Der Verlust bleibt.** WAN-Sockets zeigen 30–60 %
   `bytes_retrans` an `bytes_sent`, viele `rcv_ooopack`,
   `retrans:0/164` und ähnliche Werte. `nstat`-Intervalle zeigen
   mehrere hundert `TcpRetransSegs`, davon ein großer Teil
   `TCPLostRetransmit`. `TCPLossFailures` > 0 erklärt die
   SSH-Abbrüche: Timeout-Recovery scheitert.

4. **WLAN ist unschuldig.** `-50 dBm` bei über 100 MBit/s
   negotiated rate ist sehr gut. `dmesg` zeigt keine
   Firmware-, Deauth- oder Disassoc-Fehler. Loopback-Sockets laufen
   mit voller Rate ohne Retransmits. Der Host-Stack ist gesund.

5. **Schlussfolgerung:** Der Verlust entsteht hinter dem Handy
   (Mobilfunk-Uplink oder Provider-Policer), nicht auf der
   WLAN-Strecke und nicht im Congestion Control. BBR + `fq` machen
   das Verhalten ruhiger, zaubern aber keine Bandbreite herbei.

## 5. Empfehlungen

- `fq` + `bbr` lassen. Das ist die richtige Kombination für diese Leitung.
- Alte `cubic`-Sitzungen schließen, nur neue Sockets vergleichen.
- Handy per USB-Tethering statt WLAN-Hotspot testen. Das nimmt
  WLAN-Jitter und 2,4-GHz-Störungen raus.
- Falls WLAN bleiben muss: Hotspot auf 5 GHz stellen.
- SSH stabilisieren (unabhängig von BBR):

```text
Host *
    ServerAliveInterval 15
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

- Bei viel Loss: `mosh` statt `ssh` erwägen.
- Weiter beobachten mit `ss -tin dst <SERVER-IP>` und `nstat`-Differenzen.
  Entscheidend sind `bbr:(bw:...)`, `retrans` und `delivery_rate`
  auf frischen Verbindungen.

## 6. Anhang: `fq`-Ausgabe für Einsteiger

Dieser Abschnitt holt von vorne ab. Kein Vorwissen nötig.

### 6.1 Was ist ein qdisc?

Jedes Netzwerkinterface hat eine Warteschlange für ausgehende Pakete.
`qdisc` heißt „queueing discipline“. Also: die Regel, wer wann senden
darf. `fq` heißt Fair Queueing. Jeder Flow bekommt fair seinen Anteil.

### 6.2 Was ist ein Paket, was heißt `p`?

Daten werden in Stücken verschickt. Ein Stück heißt Paket.
`p` heißt Pakete. `10000p` heißt: maximal 10.000 Pakete puffern.

Warum Pakete und nicht Bytes? Pakete sind verschieden groß. Ein Limit
in Bytes könnte man mit vielen Minipaketen umgehen. Die Paketanzahl
begrenzt den Verwaltungsaufwand. Die Bytegröße steht daneben,
z. B. `backlog 0b 0p`: 0 Bytes, 0 Pakete.

### 6.3 Was ist ein Flow?

Ein Flow ist eine Verbindung. Erkannt an fünf Merkmalen: Quell-IP,
Ziel-IP, Quell-Port, Ziel-Port, Protokoll. Eine SSH-Sitzung zum Server
ist ein Flow. Jede Loopback-Verbindung ein weiterer.

### 6.4 Wozu die Hash-Tabelle?

Der Kernel will pro Flow einen eigenen Zähler und Puffer führen. Aber
viele Flows linear zu durchsuchen ist langsam.

Trick: Hash-Funktion. Sie rechnet aus den fünf Flow-Merkmalen eine
Zahl. Diese Zahl heißt Bucket. Es gibt 1024 Buckets (`buckets 1024`).
Jeder Flow landet in genau einem Bucket. Nachschlagen geht dann sofort,
ohne Suchen. Wie ein Postfachregal mit 1024 Fächern: Der Name auf dem
Brief bestimmt per Rechenregel das Fach. Der Postbote rechnet statt
zu suchen.

1024 Fächer, aber beliebig viele Flows. Zwei Flows können dasselbe
Fach treffen. Das heißt Kollision. Dann teilen sie sich Zähler und
Puffer. Ungenau, aber selten ein Problem.

Manche Pakete gehören zu keinem bekannten Socket, etwa
weitergeleitete Pakete. Sie heißen Orphans (Waisen). Sie bekommen
keine eigene Flow-Verwaltung, sondern teilen sich Buckets.
`orphan_mask 1023` steuert, wie diese Buckets ausgewählt werden.
1023 ist 1024 minus 1. Das begrenzt das Hash-Ergebnis auf 0 bis 1023.

### 6.5 Die `fq`-Felder im Einzelnen

Beispiel (zensiert, Werte gerundet):

```text
qdisc fq 8001: root refcnt 2 limit 10000p flow_limit 100p
buckets 1024 orphan_mask 1023 bands 3
quantum 3028b initial_quantum 15140b low_rate_threshold 550Kbit
refill_delay 40ms timer_slack 10us horizon 10s
Sent 5845801 bytes 12692 pkt (dropped 0, overlimits 0 requeues 0)
backlog 0b 0p requeues 0
flows 17 (inactive 17 throttled 0)
gc 0 highprio 0 fastpath 9440 throttled 2300 latency 71.9us
```

- `limit 10000p`: Gesamtpuffer. Erst darüber wird verworfen.
- `flow_limit 100p`: Puffer pro Flow. Ein Flow kann die Queue nicht
  allein volllaufen lassen.
- `bands 3`: drei Prioritätsbänder nach TOS, mit Gewichten grob 9:3:1
  im Round Robin. Band 0 ist die Überholspur.
- `quantum 3028b`: Guthaben pro Runde, ca. 2× MTU.
- `initial_quantum 15140b`: Startguthaben neuer Flows, ca. 10× MTU.
  Kurze Verbindungen kommen sofort los.
- `low_rate_threshold 550Kbit`: Flows unter 550 kbit/s gelten als
  „langsam“ und werden bevorzugt sofort geschickt (`fastpath`).
  Flows darüber werden per Pacing verzögert (`throttled`). Darum läuft
  Kleinkram sofort durch, nur Bursts werden geglättet.
- `refill_delay 40ms`: Takt der Guthaben-Nachfüllung.
- `timer_slack 10us`: Timer-Toleranz, spart Wakeups.
- `horizon 10s`: Pakete dürfen höchstens 10 s in die Zukunft terminiert
  werden. `horizon_drop` ist aus, also wird gewartet statt verworfen.
- `Sent ..., dropped 0`: nichts verworfen, Queue lief nie über.
- `backlog 0b 0p`: gerade nichts gepuffert.
- `flows 17 (inactive 17 throttled 0)`: 17 bekannte Flows, alle idle,
  keiner gedrosselt.
- `fastpath 9440, throttled 2300`: ca. 80 % ging sofort raus,
  ca. 20 % wurde per Pacing geglättet. Genau dafür ist `fq` da.
- `latency 71.9us`: maximale beobachtete Verweildauer in der Queue.
  Winzig. Kein Bufferbloat auf dieser Seite.
