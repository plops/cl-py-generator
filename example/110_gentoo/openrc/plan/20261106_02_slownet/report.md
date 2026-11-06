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
