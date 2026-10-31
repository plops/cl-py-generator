# Warum bricht meine Verbindung über den Handy-Hotspot ab?

Analyse einer Wireshark-Aufzeichnung (`disconnect_.txt`, nicht im Repository) auf dem ThinkPad E14.

Die Konsequenzen sind für das E14-Image ins `Dockerfile` eingeflossen (Abschnitt 10.6). Die IP-Adresse des SSH-Servers ist in diesem Dokument durch `<ssh-server>` ersetzt.

## 1. Ausgangslage

Ein Gentoo-Laptop (ThinkPad, WLAN-Interface `wlan0`, gesteuert von `iwd` und `dhclient`) hängt am WLAN-Hotspot eines Android-Telefons. Im Telefon steckt eine kostenlose Digital-Republic-SIM, die ungebucht nur etwa 6 kB/s liefert. Über diese Leitung läuft praktisch nur SSH.

Seit gestern bricht die Verbindung oft ab. Auffällig ist, dass vor jedem Abbruch mDNS- und IGMPv3-Pakete auftauchen. Danach hilft nur das Skript `/nettether` (WLAN neu verbinden, `dhclient` neu starten, siehe Anhang), und das dauert 14 Sekunden oder länger.

Dieser Bericht beantwortet drei Fragen:

1. Wer redet in der Aufzeichnung miteinander?
2. Was passiert beim Abbruch, und lösen die mDNS/IGMP-Pakete ihn aus?
3. Warum dauert das Wiederverbinden so lange, und woher kommt die HTTP-Anfrage?

Eine Einschränkung vorab: Einige SSH-Pakete wurden aus der Datei entfernt. Lücken im SSH-Verkehr sind deshalb kein Beweis für Funkstille.

## 2. Lesehilfe: Wie liest man so eine Zeile?

Jedes Paket beginnt in der Datei mit einer Übersichtszeile:

```
No.   Time          Source         Destination     Protocol Length Info
1425  279.796113134 10.28.83.143   224.0.0.251     MDNS     79     Standard query 0x0001 PTR _hotspot._tcp.local, "QU" question
```

- `No.` ist die laufende Paketnummer.
- `Time` zählt die Sekunden seit Beginn der Aufzeichnung. Alle Zeitangaben in diesem Bericht beziehen sich darauf.
- `Source`/`Destination` sind IP-Adressen. Bei reinen Ethernet-Protokollen wie ARP stehen dort MAC-Adressen.
- `Info` ist Wiresharks Kurzfassung des Inhalts.

Darunter folgt die Detailansicht, Schicht für Schicht: Ethernet (MAC-Adressen), IP, UDP/TCP, Anwendung.

Wichtig: Aufgezeichnet wurde auf `wlan0` aus Sicht des Laptops. Man sieht also nur, was der Laptop gesendet oder empfangen hat. WLAN-Verwaltungsrahmen wie *Deauthentication* sind in dieser Sicht unsichtbar.

## 3. Die Teilnehmer

| Wer | MAC-Adresse | IP-Adresse | Rolle |
|---|---|---|---|
| Laptop, 1. Sitzung | `d2:c1:98:f3:b2:93` | 10.28.83.99 | Client |
| Laptop, 2. Sitzung | `5e:a7:bc:2f:f2:75` | 10.28.83.87 | derselbe Client, aber mit neuer MAC |
| Telefon (Hotspot) | `b6:06:e0:78:05:21` | 10.28.83.143 und `fe80::b406:e0ff:fe78:521` | Access Point, DHCP-Server, Gateway, DNS-Server |
| SSH-Server | (hinter dem Gateway) | `<ssh-server>` | `OpenSSH_9.6p1 Ubuntu-3ubuntu13.20`, Port 22 |
| Fastly-CDN | (hinter dem Gateway) | 146.75.117.91 | liefert `nmcheck.gnome.org` aus |

Dass das Telefon gleichzeitig DHCP-Server, Router und DNS ist, steht im DHCP-Angebot (Frame 1438): *Server Identifier*, *Router* und *Domain Name Server* sind alle 10.28.83.143. Die Lease-Zeit beträgt 1 Stunde.

### Exkurs: Woran erkennt man zufällige MAC-Adressen?

Im ersten Byte einer MAC-Adresse markiert das zweitniedrigste Bit (Wert `0x02`) eine *lokal administrierte*, also selbst ausgedachte Adresse. Vom Hersteller fest vergebene Adressen haben dieses Bit nicht gesetzt.

| MAC | erstes Byte binär | lokal? |
|---|---|---|
| `d2:…` | 1101 00**1**0 | ja |
| `5e:…` | 0101 11**1**0 | ja |
| `b6:…` | 1011 01**1**0 | ja |

Alle drei Adressen sind also zufällig erzeugt. Beim Telefon ist das normal, Android randomisiert die Hotspot-MAC. Beim Laptop ist das wichtig: Er hatte nach dem Neuverbinden eine andere MAC als vorher. Für das Telefon ist er damit ein völlig neues Gerät. Das wird in Abschnitt 6 eine Rolle spielen.

Nebenbei: Die IPv6-Adresse des Telefons ist aus seiner MAC abgeleitet (EUI-64). Aus `b6:06:e0` + `ff:fe` + `78:05:21` mit gekipptem Bit wird `b406:e0ff:fe78:0521`.

## 4. Die Zeitleiste

```
t [s]     Was passiert                                                 Frames
───────── ──────────────────────────────────────────────────────────── ─────────
  0,0     Laptop (MAC d2:…) fragt nach alter IP .33  -> keine Antwort  1
  6,5     zweiter Versuch                            -> keine Antwort  2
 14,7     Laptop gibt auf, fragt allgemein (Discover)                  3
 14,8     Telefon bietet .99 an, bestätigt                             4–6
 14,9     SSH-Verbindung zu <ssh-server> steht                         9–31
   …      SSH-Verkehr (teilweise herausgeschnitten)
268,3     letztes SSH-Paket der 1. Sitzung                             1424
279,8     TELEFON: IGMP-Join, MLD-Report, mDNS 0x0001  <- Neustart     1425–1429
   …      Stille (Laptop vermutlich nicht mehr verbunden)
367,7     Telefon: mDNS 0x0004 (Laptop hört wieder mit)                1430–1431
372,6     Laptop (jetzt MAC 5e:…) fragt nach alter IP .99 -> Stille    1432–1433
389,9     vier Discover-Versuche                     -> lange Stille   1434–1437
427,3     Telefon bietet .87 an, bestätigt (55 s nach dem 1. Versuch)  1438–1440
427,3     neue SSH-Verbindung                                          1443 ff.
428,6     DNS + HTTP: nmcheck.gnome.org                                1479–1492
   …      SSH-Verkehr
681,9     letztes SSH-Paket der 2. Sitzung                             2658
695,5     TELEFON: IGMP-Join, MLD-Report, mDNS 0x0001  <- Neustart     2659–2668
          (Ende der Aufzeichnung)
```

Die Aufzeichnung enthält also zwei Abbrüche, und beide haben dasselbe Muster.

## 5. Was die mDNS/IGMP-Pakete wirklich bedeuten

### Was sind das für Pakete?

- **IGMPv3 Membership Report** (IPv4) und **MLDv2 Listener Report** (IPv6) bedeuten „Ich möchte Multicast-Gruppe X empfangen“. Hier ist das `224.0.0.251` bzw. `ff02::fb`, die Gruppe für mDNS.
- **mDNS** (Multicast-DNS, Port 5353) ist Namens- und Dienstsuche im lokalen Netz ohne DNS-Server. Das Telefon fragt nach `_hotspot._tcp.local`, also ob es im Netz einen Dienst namens „hotspot“ gibt.

Alle diese Pakete kommen vom Telefon (Quell-MAC `b6:06:e0:78:05:21`), nicht vom Laptop.

### Der entscheidende Hinweis: die Transaction-ID

Jede mDNS-Anfrage trägt eine Nummer, die der Absender hochzählt:

| Frame | Zeit | ID |
|---|---|---|
| 1425 | 279,8 | **0x0001** |
| 1430 | 367,7 | 0x0004 |
| 1501 | 466,0 | 0x0005 |
| 1728 | 503,6 | 0x0006 |
| 2661 | 695,5 | **0x0001** |
| 2665 | 696,6 | 0x0002 |
| 2667 | 697,6 | 0x0003 |

Der Zähler springt zweimal zurück auf 1. Ein Programm fängt nur dann wieder bei 1 an, wenn es neu gestartet wurde. Dazu passt, dass im selben Moment IGMP- und MLD-Joins kommen. Diese Joins sendet ein Gerät genau dann, wenn eine Netzwerkschnittstelle frisch hochfährt.

Die Schlussfolgerung: Das Telefon baut seinen Hotspot neu auf. Die mDNS/IGMP-Pakete sind nicht die Ursache des Abbruchs, sondern das erste sichtbare Lebenszeichen des neu gestarteten Hotspots. Sie sind ein Symptom.

Auch das Fehlen von 0x0002 und 0x0003 nach dem ersten Neustart passt dazu. Beim zweiten Neustart folgen sie im Sekundentakt (Frames 2665, 2667). Beim ersten fehlen sie in der Aufzeichnung, und das nächste Paket ist 0x0004 bei t=367,7. Wahrscheinlich war der Laptop da schon nicht mehr mit dem WLAN verbunden und hat sie nicht mehr empfangen. Erst nach dem `iwctl connect` aus `/nettether` hörte er wieder mit.

### Warum startet das Telefon den Hotspot neu?

Das lässt sich aus dieser Aufzeichnung nicht beantworten, weil der Grund auf dem Telefon liegt. Mögliche Kandidaten (ungeprüft):

- Die Mobilfunk-Datenverbindung wird neu aufgebaut, zum Beispiel weil der Netzbetreiber die gedrosselte Sitzung trennt. Android richtet das Tethering dann neu ein.
- Die Funktion „Hotspot automatisch ausschalten“ oder der Akkusparmodus greift.
- Ein System-Update von gestern hat das Verhalten geändert („seit gestern“).

## 6. Warum das Wiederverbinden so lange dauert

### Exkurs: So holt sich ein Gerät per DHCP eine IP-Adresse

Normalfall bei einem neuen Gerät, vier Schritte („DORA“):

```
Client                         Server
  │ ── DISCOVER ("wer gibt mir eine IP?") ──▶ │
  │ ◀── OFFER   ("nimm .87")             ──── │
  │ ── REQUEST  ("ich nehme .87")        ──▶ │
  │ ◀── ACK     ("ok, 1 Stunde")         ──── │
```

Abkürzung für ein bekanntes Gerät: `dhclient` merkt sich die letzte Adresse in seiner Lease-Datei. Beim Neustart probiert er zuerst direkt `REQUEST ("ich möchte wieder .99")`. Ein höflicher Server antwortet mit ACK oder mit NAK („nein, frag neu“). Ein unhöflicher Server schweigt. Dann wartet `dhclient` mehrere Sekunden und versucht es erneut, bevor er auf DISCOVER zurückfällt.

### Was in der Aufzeichnung passiert

**1. Sitzung (Frames 1–6):**

```
 0,0 s  REQUEST .33   -> Schweigen
 6,5 s  REQUEST .33   -> Schweigen
14,7 s  DISCOVER      -> sofort OFFER .99, ACK
```

Das Telefon schickt nie ein NAK. Allein dadurch gehen etwa 14 Sekunden verloren, genau die Zeit, die du beobachtest.

**2. Sitzung (Frames 1432–1440):**

```
372,6 s  REQUEST .99   (neue MAC 5e:…!)   -> Schweigen
377,4 s  REQUEST .99                      -> Schweigen
389,9 s  DISCOVER                         -> Schweigen
397,4 s  DISCOVER                         -> Schweigen
412,7 s  DISCOVER                         -> Schweigen
427,2 s  DISCOVER                         -> OFFER .87, ACK
```

Hier kommt die MAC-Randomisierung ins Spiel. Der Laptop fragt mit der neuen MAC `5e:…` nach `.99`, der Adresse, die das Telefon der alten MAC `d2:…` zugeteilt hatte. Aus Sicht des Telefons will ein fremdes Gerät eine vergebene Adresse haben. Es schweigt erst und vergibt schließlich eine andere (`.87`). Das Feld *Seconds elapsed: 37* im Request (Frame 1439) zeigt, wie lange `dhclient` schon im DISCOVER-Modus wartete. Insgesamt dauerte es 55 Sekunden.

Warum das Telefon auch die DISCOVER-Pakete eine Weile ignoriert, lässt sich aus der Aufzeichnung nicht sicher sagen. Der Zusammenhang mit der gewechselten MAC und der angefragten alten Adresse liegt nahe, ist aber nicht bewiesen.

### Weitere Stolpersteine im Skript `/nettether`

- `killall dhclient` steht ohne `sudo`, während `dhclient` mit `sudo` gestartet wird. Als Benutzer `kiel` kann `killall` einen root-Prozess nicht beenden, alte `dhclient`-Instanzen laufen also weiter.
- `iw dev wlan0 set txpower limit 300` setzt die Sendeleistung auf 300 mBm = 3 dBm, also sehr wenig. Zusammen mit `power_save on` macht das die Funkverbindung anfälliger.

## 7. Woher kommt die HTTP-Anfrage?

In der ganzen Aufzeichnung gibt es genau eine HTTP-Anfrage (Frames 1479–1492), etwa 1,3 Sekunden nach dem DHCP-ACK:

```
428,58  DNS  A nmcheck.gnome.org  -> CNAME …fastly.net, A 146.75.117.91
428,66  HTTP GET /check_network_status.txt
428,71  HTTP/1.1 200 OK (text/plain, eine Zeile)
```

`http://nmcheck.gnome.org/check_network_status.txt` ist die voreingestellte Konnektivitätsprüfung von **NetworkManager**. NetworkManager prüft damit, ob „echtes Internet“ da ist oder ein Captive Portal (Hotel-WLAN-Anmeldeseite) dazwischenhängt. Ausgelöst wurde sie durch die neue Default-Route, die `dhclient` gerade gesetzt hatte.

NetworkManager steht nicht in der `world`-Datei des Gentoo-Builds, ist also als Abhängigkeit eines anderen Pakets hereingekommen. Auf dem laufenden System ist er bestätigt aktiv:

```
localhost /etc/iwd # pgrep -a NetworkManager
2660 /usr/sbin/NetworkManager --pid-file /run/NetworkManager/NetworkManager.pid
localhost /etc/iwd # nmcli general status
STATE         CONNECTIVITY  WIFI-HW  WIFI     WWAN-HW  WWAN     METERED
disconnected  limited       enabled  enabled  missing  enabled  unknown
```

`STATE disconnected` passt zu `unmanaged-devices=interface-name:wlan0`: NetworkManager betrachtet `wlan0` nicht als seine Verbindung. Trotzdem führt er Konnektivitätsprüfungen durch (`CONNECTIVITY limited`). Er läuft also nebenher, ohne etwas Nützliches zu tun. Wie man das sauber auflöst, steht in Abschnitt 10.

Abschalten, falls nicht gebraucht:

```ini
# /etc/NetworkManager/conf.d/no-connectivity.conf
[connectivity]
enabled=false
```

Oder ganz entfernen: `rc-update del NetworkManager default`. Bei 6 kB/s kosten die rund 1 kB pro Prüfung nicht viel, aber jedes unnötige Paket zählt.

## 8. Was diese Aufzeichnung nicht zeigen kann

- WLAN-Ebene (Assoziation, Deauthentication, Reason-Codes): Dafür `iwmon` mitlaufen lassen oder die iwd-Logs ansehen.
- Den Grund für den Hotspot-Neustart: Der liegt auf dem Telefon.
- Vollständigen SSH-Verkehr: Pakete wurden herausgeschnitten.

## 9. Was man tun kann

In der Reihenfolge des Nutzens:

1. **iwd die IP-Konfiguration selbst machen lassen.** iwd hat einen eingebauten, schnellen DHCP-Client und verbindet sich nach einem Hotspot-Neustart automatisch wieder. Damit werden `dhclient` und meist auch `/nettether` überflüssig.
   ```ini
   # /etc/iwd/main.conf
   [General]
   EnableNetworkConfiguration=true

   [Network]
   NameResolvingService=resolvconf
   ```
2. **Stabile MAC-Adresse verwenden.** Herausfinden, wer randomisiert: In `/var/lib/iwd/mi.psk` nach `AlwaysRandomizeAddress=true` suchen und prüfen, ob `macchanger` läuft. Die Hardware-MAC zeigt `ethtool -P wlan0`.
3. **Falls `dhclient` bleibt:** `sudo killall dhclient` schreiben und mit frischer Lease-Datei starten, damit er keine veraltete Adresse anfragt:
   ```bash
   sudo dhclient -lf /tmp/dhclient.wlan0.leases wlan0
   ```
4. **Sendeleistung normal lassen:** `sudo iw dev wlan0 set txpower auto`.
5. **Am Telefon:** „Hotspot automatisch ausschalten“ deaktivieren und den Hotspot vom Akkusparmodus ausnehmen. Beobachten, ob die Neustarts mit einem Mobilfunk-Wechsel zusammenfallen, etwa wenn die Netzanzeige kurz verschwindet.
6. **Für SSH selbst:** Bei häufigen Netzwechseln überlebt `mosh` (laut `world` installiert) IP- und MAC-Wechsel, während eine TCP-Verbindung dabei abreißt.

## 10. Netzwerk richtig einrichten: Wer macht was?

### 10.1 Die Schichten einer WLAN-Verbindung

Damit SSH über das Handy läuft, müssen vier Aufgaben erledigt werden. Für jede gibt es mehrere Werkzeuge:

| Schicht | Aufgabe | Werkzeuge auf diesem System |
|---|---|---|
| 1. Funk | Netz suchen, anmelden (WPA2), Verbindung halten, Sendeleistung, Stromsparen | Kernel-Treiber `mt7921e`, `iw` (Diagnose und Einstellungen), **`iwd`** (alternativ `wpa_supplicant`) |
| 2. IP-Adresse | Per DHCP Adresse, Route und DNS-Server holen und die Lease verlängern | `dhclient` (ISC, seit 2022 nicht mehr gepflegt), `dhcpcd`, **iwd intern**, **NetworkManager intern** |
| 3. Namensauflösung | `/etc/resolv.conf` schreiben | `dhclient`, `dhcpcd`, iwd (über `resolvconf`), NetworkManager, oder statisch |
| 4. Steuerung | Entscheiden, wann was passiert; automatisch wiederverbinden; Prioritäten | **NetworkManager**, iwd (für sich selbst), netifrc (Gentoo-OpenRC), dein Skript `/nettether` |

Die wichtigste Regel gegen Konflikte lautet: Pro Interface und pro Schicht darf genau ein Werkzeug zuständig sein. Kämpfen zwei DHCP-Clients um `wlan0`, entstehen doppelte Adressen, wechselnde Default-Routen und ein ständig überschriebenes `/etc/resolv.conf`. Melden sich zwei Funk-Dienste (iwd und `wpa_supplicant`) gleichzeitig an, fliegt die Verbindung im Sekundentakt raus.

### 10.2 Drei saubere Varianten

| | A: nur iwd | B: NetworkManager mit iwd-Backend | C: manuelles Skript |
|---|---|---|---|
| Funk (Schicht 1) | iwd | iwd (von NM gesteuert) | iwd |
| DHCP (Schicht 2) | iwd intern | NM intern | `dhclient` |
| `resolv.conf` | iwd über `resolvconf` | NM | `dhclient` |
| Steuerung | iwd | NM | du, von Hand |
| NetworkManager | aus | an | aus oder `wlan0` unmanaged |
| geeignet für | schlankes System (dwm, SSH), ein bis zwei bekannte Netze | Desktop mit vielen Netzen, VPN, Ethernet/WLAN-Wechsel, GUI | Fehlersuche |

Für dieses Setup (dwm, SSH, ein Hotspot) ist **Variante A** die einfachste und robusteste.

#### Variante A: iwd allein

```ini
# /etc/iwd/main.conf
[General]
# iwd macht DHCP selbst
EnableNetworkConfiguration=true
# beim Verbinden die echte MAC verwenden
AddressRandomization=disabled

[Network]
# resolvconf braucht net-dns/openresolv.
# Ohne openresolv: none, dann bleibt /etc/resolv.conf unangetastet.
NameResolvingService=none
```

> **Achtung: keine Kommentare hinter Werten.** iwd (Bibliothek ell) und NetworkManager (GKeyFile) kennen Kommentare nur am Zeilenanfang. Aus `EnableNetworkConfiguration=true   # Kommentar` wird der Wert `true   # Kommentar`. Das ist kein gültiger Wahrheitswert, also gilt still der Standard `false`, und iwd sendet kein DHCP.

```ini
# /var/lib/iwd/mi.psk   (wird beim ersten "iwctl ... connect" angelegt)
[Security]
Passphrase=...

[Settings]
AutoConnect=true
# AlwaysRandomizeAddress=true  <- falls vorhanden: entfernen
```

Die übrigen Dienste abschalten, damit niemand dazwischenfunkt:

```bash
sudo rc-service NetworkManager stop
sudo rc-update del NetworkManager default    # falls dort eingetragen
sudo rc-update del dhcpcd default 2>/dev/null
sudo killall dhclient
sudo rc-service iwd restart
```

Danach genügt einmalig `iwctl station wlan0 connect mi`. Wenn das Telefon den Hotspot neu startet, verbindet sich iwd von selbst wieder.

Ein Hinweis zu `/etc/resolv.conf`: Der Docker-Build kopiert eine statische Datei mit `194.11.90.70` (Firmen-DNS) und `8.8.8.8`. Am Hotspot ist das Firmen-DNS nicht erreichbar, und jede Anfrage dorthin läuft erst in einen Timeout. Mit `NameResolvingService=none` fasst iwd die Datei nicht an. Unterwegs also am besten nur `nameserver 8.8.8.8` (oder `1.1.1.1`) eintragen. Alternativ `net-dns/openresolv` installieren und `resolvconf` verwenden; dann trägt iwd den DNS des Telefons ein.

#### Variante B: NetworkManager mit iwd als Funk-Backend

NetworkManager nutzt standardmäßig `wpa_supplicant`. Wenn iwd daneben läuft, gibt es Streit. Deshalb muss NM ausdrücklich auf iwd umgestellt werden. Unter Gentoo braucht `net-misc/networkmanager` dafür das USE-Flag `iwd`; prüfen mit `equery uses networkmanager`.

```ini
# /etc/NetworkManager/NetworkManager.conf
[main]
# NM macht DHCP selbst, kein dhclient/dhcpcd
dhcp=internal

[device]
wifi.backend=iwd

# die Zeile unmanaged-devices=interface-name:wlan0 entfernen

[connectivity]
# keine nmcheck.gnome.org-Abfragen
enabled=false
```

```ini
# /etc/iwd/main.conf
[General]
# IP-Konfiguration macht NM, nicht iwd
EnableNetworkConfiguration=false
```

```bash
sudo rc-service iwd restart
sudo rc-service NetworkManager restart
nmcli device wifi connect mi password '...'
nmcli connection modify mi \
    connection.autoconnect yes \
    connection.metered yes \
    802-11-wireless.cloned-mac-address permanent
```

`connection.metered yes` teilt Programmen mit, die NM abfragen, dass die Leitung teuer ist, damit sie Updates zurückhalten.

#### Variante C: Skript (der jetzige Zustand)

Das funktioniert nur konfliktfrei, wenn

- `EnableNetworkConfiguration=false` in iwd gesetzt ist (Standard),
- NetworkManager `wlan0` als unmanaged führt (ist so) oder gar nicht läuft,
- kein `dhcpcd`-Dienst läuft,
- das Skript alte `dhclient`-Prozesse wirklich beendet (`sudo killall`).

### 10.3 Prüfen, ob es Konflikte gibt

```bash
# Wer läuft überhaupt? Erwartet für Variante A: nur iwd
pgrep -a 'iwd|wpa_supplicant|NetworkManager|dhclient|dhcpcd|connmand'

# Welche Dienste starten beim Booten?
rc-update show | grep -E 'iwd|NetworkManager|dhcpcd|net\.'

# Genau eine IPv4-Adresse auf wlan0?
ip -4 addr show wlan0

# Genau eine Default-Route?
ip route show default

# Wer hat resolv.conf zuletzt geschrieben? (Kommentar in der ersten Zeile)
head -3 /etc/resolv.conf

# Sieht NetworkManager wlan0 als "unmanaged" (Var. A/C) oder "connected" (Var. B)?
nmcli device status

# Was sagt iwd?
iwctl station wlan0 show
```

Warnsignale sind zwei Adressen auf `wlan0`, zwei Default-Routen, `wpa_supplicant` und `iwd` gleichzeitig, oder ein `resolv.conf`, dessen Inhalt sich ohne erkennbaren Grund ändert.

Ein Sonderfall ist `wpa_supplicant -u`. Mit `-u` läuft er als D-Bus-Dienst, ohne von sich aus ein Interface zu übernehmen. So wird er automatisch gestartet (D-Bus-Aktivierung), sobald ein Programm wie NetworkManager ihn anfragt. Er taucht deshalb nicht in `rc-update show` auf. Solange ihm niemand `wlan0` übergibt, stört er iwd nicht, ist aber eine Konfliktquelle, die nur auf ihren Auslöser wartet. Ob er ein Interface hält, zeigt:

```bash
busctl get-property fi.w1.wpa_supplicant1 /fi/w1/wpa_supplicant1 \
    fi.w1.wpa_supplicant1 Interfaces      # erwartet: ao 0
```

Zum Aufräumen lässt sich noch klären, warum NetworkManager überhaupt installiert ist:

```bash
equery depends net-misc/networkmanager
emerge -pv --depclean net-misc/networkmanager
```

### 10.4 Was bringt ein Dienst gegenüber dem manuellen Skript?

| Aspekt | Skript `/nettether` | iwd (oder NM) als Dienst |
|---|---|---|
| Auslöser | Du merkst den Abbruch und startest es von Hand | Der Kernel meldet den Verbindungsverlust per nl80211-Ereignis, und iwd reagiert sofort |
| Wiederverbinden | einmalig; bei erneutem Abbruch wieder von Hand | automatisch, mit Wiederholungen und Backoff |
| Warten auf das Netz | `until … sleep 1`-Schleife, bricht nie ab | Scan-Ereignisse statt Polling |
| DHCP | `dhclient` fragt nach veralteten Adressen und wartet 14–55 s (Abschnitt 6) | Der interne Client ist an die Verbindung gekoppelt und startet sauber neu |
| Lease-Verlängerung | hängt an einem herumliegenden `dhclient`-Prozess | Teil des Dienstes |
| Sicherheit | WLAN-Passwort im Klartext im Skript | Passwort nur in `/var/lib/iwd/mi.psk`, lesbar nur für root |
| Wettläufe | `killall` ohne `sudo` lässt alte Prozesse leben; mehrere `dhclient`-Instanzen möglich | ein Prozess besitzt den Zustand |
| Suspend/Resume | Skript erneut ausführen | wird automatisch behandelt |
| Diagnose | nur die Ausgabe im Terminal | Logs (`iwd -d`, syslog), Statusabfrage mit `iwctl`/`nmcli` |

Was ein Dienst nicht übernimmt, sind die Sendeleistung (`txpower`) und Treiber-Feinheiten. Wer die wirklich setzen will, packt sie in ein kleines `/etc/local.d/wlan.start`. Besser bleibt man aber bei `txpower auto` (Abschnitt 9, Punkt 4). Stromsparen lässt sich bei Bedarf in iwd abschalten:

```ini
# /etc/iwd/main.conf
[DriverQuirks]
PowerSaveDisable=*
```

### 10.5 Fehlersuche: iwd-Logging unter Gentoo/OpenRC

#### Warum man zunächst gar kein Log findet

iwd schreibt seine Meldungen standardmäßig nach stderr. Nur mit `-l syslog` oder `-l journal` landen sie woanders (siehe `src/main.c` im iwd-Quellcode). Das Gentoo-Startskript `/etc/init.d/iwd` ist sehr knapp:

```sh
pidfile="/run/iwd.pid"
command="/usr/libexec/iwd"
command_background="yes"
```

OpenRC schickt das stderr eines Hintergrunddienstes ohne weitere Angabe nach `/dev/null`. Außerdem ist in der `world` kein Syslog-Daemon (syslog-ng, sysklogd, metalog) eingetragen. Es gibt also keine iwd-Logdatei, bis man selbst eine anlegt.

#### Weg 1: dauerhaft über `/etc/conf.d/iwd`

OpenRC liest `/etc/conf.d/<dienst>` vor dem Start ein. Dort lassen sich Startargumente, Umleitungen und Umgebungsvariablen setzen:

```sh
# /etc/conf.d/iwd
# -d: Debug-Meldungen aller iwd-Module
command_args="-d"
# stdout/stderr in eine Datei statt nach /dev/null
output_log="/var/log/iwd.log"
error_log="/var/log/iwd.log"
# zusätzlich jedes DHCP-Paket protokollieren (debug|info|warn|error)
export IWD_DHCP_DEBUG=debug
```

```bash
sudo rc-service iwd restart
sudo tail -f /var/log/iwd.log
```

Mit `-d` wächst die Datei schnell. Nach der Fehlersuche `command_args` und `IWD_DHCP_DEBUG` wieder entfernen.

#### Weg 2: einmalig im Vordergrund

Das ist am schnellsten, aber Vorsicht: Wird iwd gestoppt, fällt das WLAN aus. Läuft die aktuelle SSH-Sitzung über diesen Hotspot, ist sie danach weg. Also an der lokalen Konsole ausführen.

```bash
sudo rc-service iwd stop
sudo env IWD_DHCP_DEBUG=debug /usr/libexec/iwd -d 2>&1 | tee /tmp/iwd.log
# zweites Terminal:
iwctl station wlan0 connect mi
```

Mit Strg+C beenden und danach `sudo rc-service iwd start` ausführen.

#### Was im Log stehen sollte

```bash
grep -i -E 'dhcp|netconfig' /var/log/iwd.log
```

Der erwartete Ablauf: Verbindung aufgebaut, netconfig startet, DHCP-DISCOVER, OFFER, REQUEST, ACK, Adresse gesetzt. Erscheint nach dem Verbindungsaufbau keine einzige DHCP- oder netconfig-Zeile, ist die IP-Konfiguration in iwd nicht aktiv.

#### Checkliste: iwd sendet kein DHCP-DISCOVER

1. **Kommentar hinter dem Wert?** Das ist der häufigste Fehler (siehe Kasten in Variante A).
   ```bash
   grep -n '=.*#' /etc/iwd/main.conf /var/lib/iwd/*.psk
   ```
   Jede Trefferzeile reparieren.
2. **Richtige Sektion?** `EnableNetworkConfiguration` gehört unter `[General]`, `NameResolvingService` unter `[Network]`. Groß- und Kleinschreibung beachten.
3. **Neu gestartet?** iwd liest `main.conf` nur beim Start: `sudo rc-service iwd restart`.
4. **Statische IP im Netzprofil?** Steht in `/var/lib/iwd/mi.psk` eine Sektion `[IPv4]` mit `Address=`, verwendet iwd diese statt DHCP.
5. **Überhaupt verbunden?** DHCP beginnt erst nach erfolgreicher Anmeldung:
   ```bash
   iwctl station wlan0 show     # State: connected, bei aktiver Netconfig auch IPv4 address
   ```
6. **Reste der alten Lösung?** Ein laufender `dhclient` oder eine alte Adresse verwirren die Diagnose:
   ```bash
   pgrep -a 'dhclient|dhcpcd|wpa_supplicant|NetworkManager'
   sudo killall dhclient; sudo ip addr flush dev wlan0
   sudo rc-service iwd restart
   ```
7. **Wireshark richtig angesetzt?** Die Aufzeichnung auf `wlan0` muss vor dem Verbindungsaufbau laufen, Filter `dhcp`. Das DHCP-Paket geht direkt nach dem Verbinden raus. Wer erst danach startet, verpasst es.

#### Die Funk-Ebene mitschneiden: `iwmon`

Für die Frage, warum die Verbindung überhaupt abreißt (Abschnitt 8), zeichnet `iwmon` die Kommunikation zwischen iwd und Kernel auf (nl80211). Das schließt Disconnect-Ereignisse mit Reason-Code ein. `iwmon` ist im Gentoo-Paket über das standardmäßig aktive USE-Flag `monitor` enthalten.

```bash
sudo iwmon -w /tmp/wifi.pcap     # läuft bis Strg+C, Datei in Wireshark öffnen
sudo iwmon                       # oder live im Terminal
```

### 10.6 Umsetzung im E14-Image

Variante A ist für das ThinkPad-E14-Image fest eingebaut. Der Block im `Dockerfile` steht nach dem Erzeugen von `gentoo.squashfs_nv` und vor `gentoo.squashfs_e14`, deshalb bleibt das HP-Z6-Image unverändert.

| Datei im Repo | Ziel im E14-Image | Zweck |
|---|---|---|
| `config/e14-iwd-main.conf` | `/etc/iwd/main.conf` | DHCP durch iwd, echte MAC, DNS über openresolv |
| `config/e14-iwd.confd` | `/etc/conf.d/iwd` | `rc_provide="net"`, Log nach `/var/log/iwd.log`, Debug-Zeilen auskommentiert |
| `config/e14-nm-no-connectivity.conf` | `/etc/NetworkManager/conf.d/90-e14-no-connectivity.conf` | keine `nmcheck.gnome.org`-Abfragen, falls NM doch startet |

Außerdem installiert der Block `net-dns/openresolv` und nimmt NetworkManager aus dem Runlevel `default`, falls er dort steht.

`rc_provide="net"` sorgt dafür, dass iwd für OpenRC als Anbieter von „Netzwerk“ gilt. Dienste, die `need net` verlangen, ziehen dann nicht mehr NetworkManager nach. Dasselbe macht das Gentoo-USE-Flag `standalone` von `net-wireless/iwd`. Wer NetworkManager auf dem E14 bisher gestartet hat, ist nicht abschließend geklärt; diese Abhängigkeitskette ist ein naheliegender Kandidat.

## 11. Zusammenfassung

Das Telefon startet seinen Hotspot neu. Die mDNS-Transaction-ID springt auf 1 zurück, und gleichzeitig kommen IGMP- und MLD-Joins: Das ist der Fingerabdruck dieses Neustarts. Der Laptop verliert dabei die Verbindung.

Das Wiederverbinden ist aus zwei Gründen langsam: `dhclient` fragt zuerst nach einer alten Adresse und wartet auf eine Antwort, die nie kommt. Und der Laptop meldet sich mit einer neuen Zufalls-MAC, für die das Telefon die alte Adresse nicht herausgibt.

Die einzige HTTP-Anfrage ist die Konnektivitätsprüfung von NetworkManager.

Am meisten bringt es, DHCP von iwd erledigen zu lassen und die MAC-Adresse stabil zu halten.

## Anhang: das bisherige Skript `/nettether`

```bash
#!/bin/bash
killall dhclient
rfkill unblock wlan

# 1. Trigger the scan
iwctl station wlan0 scan

# 2. Wait and check until the network appears in the list
export SID=mi
until iwctl station wlan0 get-networks | grep -q $SID; do
    echo "Waiting for "$SID"..."
    sleep 1
done

# 3. Connect once found
iwctl -P 'XXX' station wlan0 connect $SID
sudo iw dev wlan0 set power_save on

sudo iw dev wlan0 set txpower limit 300
sudo iw dev
sudo iw dev wlan0 link
sudo dhclient wlan0
```
