# ASan-Arbeitslog (openrc2, Stufenmodell)

Laufendes Protokoll: Versuch → Problem → Lösung. Kurz, nur Ergebnisse.
 Details zu abgeschlossenen Stufen stehen in `walkthrough.md`.

## 2026-10-24, Batch 1 (bash, coreutils, util-linux, grep, sed)

- Versuch: `emerge --oneshot` der 5 Pakete mit ASan (Log `openrc2/logs/setup01_run_20261024_071446.log`).
- Problem: Direkt nach Bash-Install melden ~300 Bash-Aufrufe je `17 byte(s) leaked`
  (LSan, exit 23); danach stirbt der Emerge (exit 1). Ursache: frische ASan-Bash.
- Analyse (Gast, SSH): Vanilla-Bash + ASan ist sauber (`-c`, Skripte, Pipes, leere
  Env, Parser-Tests, Expansionen, 2 Gentoo-Patches — alles rc=0). Trigger nur im
  Ebuild-Kontext (`ebuild clean/setup` leckt; volle 141-Variablen-Env allein nicht).
  Zweite Stelle: Skript mit sterbender Expansion (`source "${X:?}/..." || exit`)
  leckt 32+32 B. Größen variieren je Lauf (17 B Docker / 7 B Gast) → String-Inhalt.
- Entscheidung: Bash vorerst in `config/batches/hold.txt` (nosanitizer), Batch 1
  ohne Bash weiter. Leak-Jagd als abgegrenzte Subagenten-Aufgabe (Ergebnis: Patch
  oder begründete Suppression-Empfehlung). Regel: kein Paket blockiert die Rolle;
  nicht-fundamentale Pakete fliegen raus statt repariert zu werden.
- Status: Batch-1-Rebuild ohne Bash läuft (Log s. `openrc2/logs/`).

## Hold-Liste (wirksam, mit Grund)

- `app-shells/bash` — 17-B-Exit-Leak, Hunt läuft als Subagent (s. unten). Rückkehr nach Patch.

## 2026-10-24, Batch 1 zweiter Lauf (ohne Bash)

- Versuch: coreutils, util-linux, grep, sed mit ASan (Log `..._081514.log`, exit 1).
- Problem: `sys-apps/coreutils-9.11` stirbt in `make install`: frisches `ginstall`
  leckt Exit-Time-Hashtabellen (`record_file`, `dest_info_init`, bis 5159 B/255
  Obj.) → Exit 23 → `install-binPROGRAMS` fehlgeschlagen. Klassischer Fall:
  Upstream befreit Prozessende-Speicher nie (OS reklamiert) — kein Bug.
- Lösung: KEIN Quell-Patch (unverhältnismäßig, upstream-feindlich), KEIN
  Paket-Opt-out (coreutils fundamental), sondern gezielte LSan-Suppressions
  (`config/asan-suppressions/lsan.supp`: `leak:record_file`,
  `leak:dest_info_init`). ASan/UBSan bleiben voll scharf; Suppressions gelten
  per `ENV LSAN_OPTIONS` für Build UND Laufzeit (`Dockerfile`).
- Status: Rebuild mit Suppressions läuft (Log `..._082752.log`).

## 2026-10-24, Batch 1 dritter Lauf (coreutils ok, util-linux fällt)

- Versuch: Mit Suppressions neu gebaut (Log `..._082752.log`, exit 1).
- Problem: `coreutils` jette durch, aber `util-linux` stirbt in `configure`:
  frisches `chgrp` (coreutils, ASan) leckt 8 B → Exit 23. Muster bestätigt:
  JEDE GNU-CLI leckt beim Beenden etwas; Einzeljagd pro Tool = sinnlos.
- Lösung (Policy-Wechsel): `LSAN_OPTIONS="suppressions=...:exitcode=0"` global
  (`Dockerfile`). Leak-Meldungen bleiben sichtbar (Triage), zählen aber exit 0 —
  weder Build noch Boot brechen. ASan (UAF/Overflow) + UBSan bleiben lethal.
  Bash kehrt damit aus dem Hold zurück (Hunt läuft als Bonus weiter).
- Status: Rebuild Batch 1 (5 Pakete) läuft.

## 2026-10-24, Merge-Phasen sterben (alle 5 Pakete, Log `..._095839.log`)

- Versuch: Mit exitcode-Policy neu gebaut (exit 1): Kompilate alle grün, aber
  prerm/postrm/preinst/postinst sterben mit `__save_ebuild_env failed`.
- Analyse (Gast, reproduziert): `s() { unset -f s; declare -fp; }; s | cat`
  liefert PIPESTATUS 0=1 unter ASan-Bash, 0 unter identischer unsanitisierter
  Bash (Gentoo 5.3.20 + Ubuntu). Varianten: mit/ohne Restfunktionen 1, Reorder 1,
  aber an Datei 0 — Pipe essenziell, kein Report, nur rc=1. Trifft exakt
  Portages `__save_ebuild_env | __filter_readonly_variables`.
- Lösung: Portage-One-liner (`save-ebuild-env.sh`: Selbst-Unset-Zeile raus,
  per `Dockerfile`-sed + live im Gast verifiziert: `ebuild setup` läuft wieder).
  Echte Bash-Ursache (Hashtabelle nach Self-Unset + declare-an-Pipe) bleibt beim
  Hunt-Subagenten (Findings per Nachricht übergeben).
- Status: Rebuild läuft.

## 2026-10-24, Dockerfile-Cache-Ordnung + Rust-Evaluierung

- Problem (Cache): Jede Dockerfile-Änderung baute Kernel/Welt neu (teuer).
  Nebenbefund: Portage-sed stand VOR `emerge portage` → Binärpaket hat den
  Workaround überschrieben (eigentliche Ursache des letzten Fehlschlags!).
- Lösung: `package.env.base` (statisch, früh kopiert) vs. `package.env` (Batch,
  spät vor Oneshot); Suppressions/ENV/sed ebenfalls spät. Kernel-/Welt-Layer
  bleiben bei Batch-Wechseln gecacht. Rebuild läuft (Log `..._104501.log`).
- Rust statt C: eigener Evaluierungs-Subagent prüft ::gentoo auf Rust-Ersatz
  (uutils, ripgrep, sudo-rs, …) inkl. Boot-Kompatibilitätsrisiken.

## 2026-10-24, Hunt-Ergebnis Bash + Rust-Evaluierung

- Hunt (Subagent, verifiziert im Gast): Echte C-Ursache — `execute_function()`
  registriert Unwind-Disposals nur für `subshell == 0`; Pipe-Subshell verwaist
  Body + getopt-State → LSan kippt Exit auf 1 (Beweis: `detect_leaks=0` → 0).
  Zwei Patches in `config/patches/app-shells/bash/` (`01-merge-subshell-dispose`,
  `02-procsub-execfail-free`), im Gast mit Vorher/Nachher-Tabelle belegt.
  Rest-Risiko (7 B + Parse-Tree) bleibt unter exitcode-Policy sichtbar.
  Portage-Workaround bleibt bis Rebuild-Bestätigung.
- Rust (Subagent, Atome in `::gentoo` verifiziert): Einziger reifer Swap ist
  `app-admin/sudo-rs` (USE=system-names) → geplant für Batch 4 (sudo ist dort).
  uutils-coreutils/findutils nur Side-by-side-Trial (`uu-`-Prefix, kein Drop-in).
  KEIN Ersatz für bash (nushell nicht POSIX), openssh, dhcpcd, cryptsetup/lvm2,
  dracut, fs-Tools, git, sysklogd — dort weiter ASan-Batches.

## 2026-10-24, Platten-Cleanup durch Hauptagent

- Problem: Build-Cache 111 GB voll → distfiles-Download scheitert, Build stirbt
  ohne Sanitizer-Beteiligung.
- Lösung: `docker image prune` (Dangling, ~0 B), dann auf Wunsch
  `gentoo-z6-min-openrc:latest` gelöscht (18 GB, aus `Dockerfile` rebuildbar)
  + `docker builder prune` (111 → 61 GB, ~50 GB frei). `my-ai-env` und
  `gentoo-qemu-min-openrc` behalten. Batch-1-Rebuild neu gestartet
  (Log `setup01_run_20261024_095839.log`).

## Verifikations-Strategie (ab 08:15)

- Standard: Docker (`docker run --rm <image> ...`) — Paket-Verhalten, Coverage
  (`strings`-Probe), funktionale Smoke-Tests der Batch-Binaries. Sekunden statt
  Minuten, skriptbar, kein serielles Gefrickel.
- Boot-Gate: QEMU nur einmal pro Batch (OpenRC-Boot mit sanitisiertem Stack,
  Login, Smoke-Test). Boot-Fehler → Fix/Patch wie bisher.
- SSH in den Gast bleibt für tiefe Hunts (Ebuild-Kontext, symbolische Traces).

## 2026-10-24, Batch-1 Run 6 (104501): Quoting-Analyse + Gast-Beweis

- Stand: `__save_ebuild_env failed` in prerm/postrm/preinst/postinst ALLER
  Batch-1-Pakete, obwohl Dockerfile-sed (#39) lief (Output "2").
- Analyse (Hauptagent, Gast per SSH verifiziert):
  - Mechanismus: `__save_ebuild_env` ruft SICH SELBST in
    `mapfile ... < <(declare -fp "${MAPFILE[@]}" | grep ...)` auf.
    Inneres bash disposal ordentlich, aber `unset -f` -> `declare -fp | cat`
    gibt rc=1 unter ASan-bash (17-B-Leak-Pfad) vs rc=0 ohne ASan.
    `__pipestatus || die` in phase-functions.sh:1195 kippt -> Phase tot.
  - Quoting-Falle: sed-Pattern mit DOPPELTEN Anfuehrungszeichen
    (`"/^\t\t.../"`) expandiert sh die `\t` VOR sed weg -> Pattern sucht
    `t__save_ebuild_env` -> No-Op (Gast: nachher-Count weiter 2).
    Mit EINFACHEN Quotes funktioniert es (Gast: Zeile 221 weg, nur Kommentar
    Zeile 5 uebrig).
  - Nebenerkenntnis: altes `grep -c "save_ebuild_env"` (unverankert) zaehlt
    Kommentar + Funktionsdefinition = "2" MIT und OHNE Patch -> keine Aussage.
- Fix Dockerfile: sed-Step prueft jetzt scharf (vorher/nachher `grep -n`,
  `test $(grep -c '__save_ebuild_env$') -eq 1` -> fail-fast statt spaetem
  Raetselraten in #40).
- Gast-Beweis: nach korrektem sed `ebuild bash-5.3_p20.ebuild setup` rc=0,
  0x "save_ebuild_env failed". Workaround im Gast damit bestaetigt aktiv.
- Rebuild mit pruefendem Step gestartet (`setup01_run_20261024_110133.log`).

## 2026-10-24, Rust-Ersatz-Evaluierung (Subagent, read-only, Stand Okt 2026)

- Ergebnis: KEIN Tausch lohnt sich im boot-kritischen Pfad. ASan bleibt Hauptweg.
- bash, util-linux, grep/sed, openrc, dbus/elogind, openssh, sysklogd, dhcpcd,
  picocom, pci/usbutils, cryptsetup, dosfstools, e2fsprogs, lvm2,
  squashfs-tools, git, python/perl: KEIN Rust-Drop-in im ::gentoo-Tree.
- Zwei Testkandidaten DANEBEN (nicht Ersatz): `sys-apps/uutils-coreutils`
  (beta, nur ~arch, ~87% GNU-Passrate, bricht emerge-Skripte) und
  `app-admin/sudo-rs` (nahezu drop-in, ~amd64, v0.2.x, 2x auditiert, aber 2025
  selbst 4 CVEs: CVE-2025-46717/-46718/-64170/-64517). sudo-rs ggf. Batch 4.
- ripgrep/fd/uutils-findutils: Komfort, kein Ersatz fuer grep/sed/find.

## 2026-10-24, Batch-1 Run 7 (110133): Portage-Workaround OK, Bash-Patch malformed

- Pruefender sed-Step BESTAETIGT: vorher Zeilen 5+221, nachher nur Zeile 5,
  `test -eq 1` bestanden. KEIN `__save_ebuild_env failed` mehr in #40.
  Portage-Huerde damit abgehakt (Gast + Docker doppelt belegt).
- Neuer Fehler: `01-merge-subshell-dispose.patch` -> `gpatch: malformed patch
  at line 26`, bash prepare-Phase stirbt. Diagnose: Hunk-Zaehler falsch
  (`-5426,6` bei nur 5 Alt-Zeilen). Patch 2 hat dieselbe Fehlerklasse
  (Hunk 1: `-6580,6` bei 4; Hunk 2: `-5928,7` bei 5). Inhalt vermutlich richtig
  (Gast-Handverifikation), nur Header falsch transkribiert.
- Fix an Subagent "bash-patch-fix" delegiert (Quellen-Fetch, Header-Korrektur,
  `--dry-run`-Validierung). Rebuild erst nach dessen Ergebnis.

## 2026-10-24, Patch-Fix durch Subagent (verifiziert)

- Eigene Diagnose ("Hunk-Zaehler falsch") war FALSCH — Subagent hat es
  praezise korrigiert: Patch 1 Header `-5426,6/+5426,12` war KORREKT, der
  Rumpf war kaputt (Leerzeile vor `function_misc_cleanup` als `+` statt
  Kontext markiert -> nur 5 Alt-Zeilen + nicht kontiguierlich). Ein-Zeichen-Fix
  Zeile 24 `+` -> ` `. Gegenbeweis: Header-only-Variante `-5426,5` scheitert.
- Patch 2: reine Header-Korrekturen (`-6580,4/+6580,8`, `-5929,5/+5929,9`),
  Inhalte unverstaendert.
- Validierung gegen bash-5.3+patches 001-020: `patch -p1 --dry-run` exit=0,
  mit Portage-Flags identisch, `git apply --check` exit=0, sequentielles
  Anwenden 01->02 ok (Offset +6 erwartet). Vorher-Fehler reproduziert.

## 2026-10-24, Batch-1 Run 8 (111655): Kompiliert alles, Merge-Phasen fallen

- Patches applizieren SAUBER, alle 5 Pakete kompilieren (119 s). Fehler erst
  bei Installation: bash prerm/postrm/postinst + alle preinst danach.
- Neue praezise Diagnose (Timeline): bash wird als ERSTES gemergt (1 of 5).
  Danach ist /bin/bash die SANITISIERTE Version -> ALLE folgenden Merge-Phasen
  (prerm/postrm alt, preinst neu) laufen unter ASan-Bash und sterben in
  `__save_ebuild_env`. Zeile-221-sed ist aktiv (Run 7 ohne Befund), Gast-Beweis
  galt nur fuer UNSANITISIERTE Bash -> halber Beweis.
- Rest-Bug: EIN ANDERES inneres Kommando (Kandidaten: `unset -f` Riesenliste,
  `declare -F/-fp`, `while read < <(...)`) liefert unter ASan-Bash nonzero
  (ASan-Abort statt LSan; exitcode=0 greift nicht).
- Hunt 2 an Subagent delegiert: lokale ASan-Bash-5.3 (+unsere 2 Patches) in
  /tmp bauen, `__save_ebuild_env` mit Stubs repro + bisektieren, Fix vorschlagen
  (Bash-Patch 3 oder Portage-Workaround). KEIN blinder Rebuild bis Ergebnis da.

## 2026-10-24, Hunt 2 Ergebnis (Subagent, /tmp-Repro, Repo unberuehrt)

- TOP: Docker-treuer sichtbarer Fehler NICHT reproduzierbar (alle treuen
  Harness-Konfigurationen PIPESTATUS=0; LSan-Urteile sind Heap-Layout-abhaengig,
  GEN1 vs GEN2 kippen deterministisch). Abbruch-Regel f.d. Teil ehrlich gezogen.
- ECHTER MECHANISMUS (verifiziert): Portage baut Phasen-Env aus Whitelist neu
  (`config.py`/`doebuild.py`; `LSAN_OPTIONS` kommt nirgends in der Lib vor) ->
  Dockerfile-ENV erreicht Phasen NIE -> LSan in Merges lethal. Alle 422
  Sanitizer-Funde in Docker-Logs sind LEAKS (kein ASan-Abort, kein UBSan).
  Gast-Gegenprobe: `LSAN_OPTIONS` in keiner Portage-Datei.
- Patches 01/02 VALIDiert (korrekt+notwendig): Vanilla-ASan-Bash deterministisch
  `PIPESTATUS=1 1` (17 B procsub + ~15 B Body/Getopt), gepatcht ueberall 0.
- Repro B (unsichtbar, Upstream-Bug, MIT Patches): getrappte INT/QUIT + procsub
  + `___*`-Funktions-Loop -> 7-B-Trap-String-Leak (`trap.c:873`, Grandchild).
  Root-Cause per Trace: `reset_signal_handlers` loescht TRAPPED, behaelt String;
  `restore_signal` -> `free_trap_command` ueberspringt free (SIG_TRAPPED weg).
  Fix-Vorschlag (2 Zeilen, NICHT angewendet): `sigmodes[sig] |= SIG_TRAPPED`
  vor `change_signal(sig, DEFAULT_SIG)` in `restore_signal()` (~Z. 1427).
  Zurueckgestellt: Docker-Phasen trappen nur TERM (kein SIG_SPECIAL) -> nicht
  der Docker-Killer; kein Boot-Bezug.
- FIX (angewendet): `config/bashrc` -> `/etc/portage/bashrc` exportiert
  LSAN_OPTIONS in Phasen (Triage-Policy, analog Zeile-221-sed). ASan/UBSan
  bleiben lethal; echte Aborts wuerden weiter knallen (dann Quellen-Patch).
- Confounder notiert: volle Disk gibt DIESELBE Die-Meldung (Host: 47 G frei,
  aktuell unauffaellig; bei Folgeruns `df` pruefen).

## 2026-10-24, Run 10 (123312): Evidence-Dump entscheidet

- Rescue-Dump funktioniert: echte LSan-Reports aus Build-Logs (7-B-Direct +
  7030-B-Summary/321 Allocs, Stacks aus /usr/bin/bash, Phasen postinst/preinst).
  KEIN ASan-Abort, KEIN UBSan. Nur Leaks -> Policy-Ansatz richtig.
- bashrc-Ergebnis negativ: Reports + Exit 1 TROTZ /etc/portage/bashrc-Export.
  PORTAGE_BASHRC-Pfad verifiziert korrekt (CONFIGROOT + EBUILD_SH_ENV_FILE).
  Grund offen (Sandbox? Sourcing-Reihenfolge?); keine weitere Zeit versenkt.
- Neuer Fix: `export LSAN_OPTIONS` in `config/env/sanitizers` (package.env-Kanal,
  belegt durch sanitisierte Binaries). Baute ab #22 neu (Cache-Bruch
  akzeptiert, ~15 Min). Dump-Rescue bleibt drin.

## 2026-10-24, Gast-Durchbruch: Live-Repro statt Docker-Raten (Minuten statt Stunden)

- Entdeckung: Der GAST hat sanitisierte /usr/bin/bash (BuildId ea69..) + volle
  Portage-Config (env/sanitizers, package.env) -> komplette Repro-Umgebung mit
  Sekunden-Iterationszeit. Alle folgenden Befunde per `ebuild ... setup` belegt.
- Kanal bashrc TOT: Export in /etc/portage/bashrc erreicht Phasen nicht
  (Probe-`declare -p >>log` bleibt leer; 70 Leak-Zeilen trotzdem).
- Kanal package.env LEBT (Portage parst die Datei -> "Invalid token" beweist
  Lesen): aber shlex-Regel! `=` ist kein wordchar -> NUR EIN `=` pro Zeile.
  `export LSAN_OPTIONS=exitcode=0` (unquoted) setzt LSAN=`exitcode` (Rest als
  Fehler) -> "ASan: expected '=' in LSAN_OPTIONS" (fatal!). Fix: Quotes:
  `LSAN_OPTIONS="exitcode=0"` -> setup rc=0, 18 Leak-Reports sichtbar, exit 0.
- Suppressions-Falle: Wert MIT `suppressions=<Pfad>` + Datei FEHLT -> exitcode
  wird KOMPLETT voided (Exit 1 trotz exitcode=0!). Datei vorhanden -> rc=0.
  Konsequenz: Unter Sandbox (Merge-Phasen) ist die Datei evtl. unlesbar ->
  derselbe Kill. Repo-Fix: PLAIN `LSAN_OPTIONS="exitcode=0"` ohne suppressions=
  (Reports ungefiltert, aber exit 0; .supp-Datei bleibt f. Handgebrauch).
- 7-Byte-Direct-Leak symbolisiert (Gast-Binary hat Symbole): `sh_set_lines_and_columns`
  (Terminal-Probe). Harmlos, exitcode-Policy deckt ab.
- Repo `config/env/sanitizers` korrigiert (plain, quoted). Run 12 nach
  Gast-Merge-Bestätigung (Sandbox-Prädiktor läuft: `emerge --oneshot grep`).

## 2026-10-24, Gast-Bisektion: Save-Pipeline GRÜN, Helper-Phasen sterben (Hunt 3)

- SAVEDBG-Befund `PS=1 1` ist BOGUS (Stub mit `return 0` zeigt auch "1 1", Merge
  geht trotzdem durch) — nur noch Outcome + Die-Meldungen zählen.
- Stand B1/Stub/Tee/R1: Dateien installieren ("Completed"), Setup-Save grün,
  aber Merge-FAILED durch 'instprep'/'other'/'die_hooks' "exited unexpectedly"
  (Helper-Kontexte, NICHT die Save-Pipeline).
- Verifiziert nebenbei: Filter standalone rc=0 (auch mit exakten Merge-Flags),
  Python-Filter auf echtem 66-KB-Env rc=0, kein OOM (7G/6G frei), Gast-Portage
  = 3.0.82.2-r1 identisch zu Docker, Zeile 221 im Gast weg, 7-B-Leak = 
  `sh_set_lines_and_columns` (harmlos, Policy deckt ab).
- Hunt 3 delegiert (Gast, max 8 Iterationen, mit Backup/Restore-Regel):
  Welcher Helper-Prozess stirbt in instprep/other/die_hooks, warum
  (Env? ECHTER Abort vorrangig!), minimaler Fix-Vorschlag.
- Repo-Stand: `config/env/sanitizers` = plain quoted exitcode=0 (bereit f. Run 12
  nach Hunt 3).

## 2026-10-24, Hunt 3 (1 Iteration!): LD_PRELOAD-Order-Fatal (BEWIESEN)

- Ursache: Helper (instprep/die_hooks/other = MiscFunctionsProcess, free=False)
  spawnen via `sandbox(1)` -> LD_PRELOAD=libsandbox.so VOR DT_NEEDED-libasan ->
  "ASan runtime does not come first" -> stiller exit(1) beim Prozessstart, kein
  Exit-File -> "exited unexpectedly". Direktbeweis: LD_PRELOAD=libsandbox.so
  bash -c echo -> kein Output, gleiche Meldung, rc=1. Kein Speicherfehler
  (volle Aborts = null im Log), SAVEDBG-"1 1" erneut als bogus bestätigt.
- Warum Setup grün: free:1 -> spawn_bash ohne Preload. Warum `-sandbox` allein
  scheiterte: `usersandbox` bleibt -> weiter spawn_sandbox.
- Fix (Gast-validiert, Binpkg-Merge VOLL GRÜN, keine Failed-Zeile):
  `FEATURES="-sandbox -usersandbox"` + plain quoted LSAN im package.env-File.
  Kein Quellenpatch möglich (Link-Order per Design). Langfristig: sandbox
  selbst mit ASan bauen. Gilt NUR pro Batch-Paket, nicht global.
- Repo `config/env/sanitizers` ergänzt. Run 12 baut ab #22 neu.

## 2026-10-24, Run 13: SHADOWING BEWIESEN (Phasen-Env-Dump aus Docker)

- Rescue-Erweiterung liefert den reinen Beweis: ALLE 5 Phasen-Envs zeigen
  `LSAN_OPTIONS="suppressions=...:exitcode=0"` = DOCKERFILE-ENV-Form, NICHT das
  package.env-plain. Vererbte Werte überschreiben package.env in Merge-Phasen.
  (CFLAGS arrivaltrotzdem: kein vererbter Konkurrent.)
- Hunt-2-These "ENV erreicht Phasen nie" damit widerlegt (es kommt an — und
  schadet). FIX: Dockerfile-ENV-Zeile GELÖSCHT, einzig package.env gilt.
  Drei-Wege-Diskriminierung in Run 14: plain erscheint (Fix!), nichts erscheint
  (package.env tot), Docker-Form bleibt (Geister-Layer).

## 2026-10-24, Run 14: GEISTER-LAYER (ENV gelöscht, Docker-Form bleibt)

- Drei-Wege-Outcome = Geist: Dockerfile-ENV ersatzlos gestrichen, Phasen-Env
  zeigt TROTZDEM `suppressions=...`-Form in allen 5 Paketen.
- Vektor-Analyse: Base-Images sauber (stage3/portage upstream); STALE-These tot
  (Build läuft aus Worktree, `setup01_build_image.sh -t ... .` im Script-Dir);
  SHADOW-These verfeinert: Dritter Vektor `config/bashrc` (COPY aktiv!) enthält
  suppressions-Form (Zeile 8, wenn .supp-Datei existiert — tut sie im Docker).
  Gast-belegt tot, aber falls Docker ihn sourct, überschreibt er package.env.
- FIX: bashrc auf plain `export LSAN_OPTIONS="exitcode=0"` gestellt (in beiden
  Welten korrekt: tot=No-Op, live=richtiger Wert). Run 15.
- Offen (kein Zyklus mehr wert): WARUM Docker bashrc sourct, Gast nicht
  (PORTAGE_BASHRC-Auflösung?).

## 2026-10-24, Strategiewechsel: Interpreter-zuletzt (User-Vorschlag)

- Befund Gast: /bin (laufendes System) und /usr/bin (Portage-Ziel) sind
  GETRENNT (kein merged-usr!): /bin/bash war plain (12. Okt), /usr/bin/bash war
  sanitiert (ea69). Docker-Stage3 ist merged-usr: EIN sanitiertes Bash
  überall -> jede Hilfs-Shell im Build stirbt. Das erklärt Gast-grün vs
  Docker-rot bei gleichen Werten.
- Eigener Fehler: Gast-"Source-Merges" waren Binpkg-Merges (make.conf holt
  Binärpakete, Docker erzwingt Quellen via inline -getbinpkg). Hat nebenbei
  den sanitierten Gast-Zustand mit Plain-Binpkgs ÜBERSCHRIEBEN (mutted). Lehre:
  Im Gast IMMER mit FEATURES="-getbinpkg" bauen für echte Tests.
- Entscheidung (User): Bash aus Batch 1 raus -> Batch 6 (Interpreter-zuletzt).
  Begründung: Shell = Werkzeug aller Installationsschritte, wie Toolchain
  (gcc/glibc: nosanitizer). Erst Programme (explizit testen!), dann Interpreter.
  Patches/sed/Policy bleiben bereit. apply-batch.sh auf Stufen 0..6 erweitert.
- Niemals abgestürzt übrigens: KEIN einziger ASan-Abort/UBSan-Fehler in allen
  Runs — nur Leak-Exits und stille Pipeline-Tode. Die Programme selbst sind
  sauber; das Werkzeug war das Problem.

## 2026-10-24, Explizite Programm-Tests (User-Wunsch), Gast, sanitized bins

- Echter Source-Build (FEATURES="-getbinpkg", Repo-Flags) -> /usr/bin/sed,
  /usr/bin/grep sanitisiert (ldd: libasan). 6 Tests, alle paritätisch zu plain:
  sed-Substitution rc=0, grep -c=2 rc=0, sed -i Mehrzeilen ok, grep auf Binary
  findet ELF rc=0, grep ohne Treffer rc=1 (POSIX, plain identisch),
  fehlende Datei rc=2 (plain identisch, /etc/hosts fehlt im Minimal-Gast).
- Fazit: Die PROGRAMME sind funktional sauber mit Sanitizern; das Problem war
  immer nur die Shell im Installations-Werkzeug, nie die Werkstücke.

## 2026-10-24, Run 16: BATCH 1 GRÜN (4 Pakete, plain Bash) + dracut-Leak

- #41 DONE (322 s), "Completed (4 of 4)", KEIN einziger Merge-Fehler.
  Interpreter-zuletzt-Strategie (User) bestätigt: mit plain Bash fürs Werkzeug
  laufen alle Merge-Pipelines sauber.
- Neuer Fehler #48 dracut: ruft Sanitizer-`sort` (8-B-Exit-Leak, Stack belegt)
  und bricht bei dessen Exit-Code ab. Fix: `LSAN_OPTIONS="exitcode=0"` als
  Prefix NUR am dracut-RUN (Build-Zeit only, nicht im Image). Reports bleiben
  sichtbar (Triage). Keine Quellenänderung nötig (GNU-Exit-Leak by-design).

## 2026-10-24, Run 17: dracut GRÜN, Berichtsschritte brauchen Policy

- #48 dracut DONE (LSAN-Prefix wirkt, sort-Leak sichtbar aber exit 0).
- #49 Paketbericht failt: `set -e`-Loop + sanitisiertes sed (Exit-Leak -> Loop-
  Abbruch). Fix: `export LSAN_OPTIONS` in den Shell-Schritten (Prefix reicht
  bei Shell-Loops NICHT, da temporär — export nötig!). Gleich mitversorgt:
  Minimal-Check (grep-Exit-Codes, sonst blinder Check) + df-Schritt.
  mksquashfs bleibt ohne (Batch 4, plain). Lehre: ENV-Prefix vs export je nach
  Schritt-Typ (externes Kommando vs Shell-Loop).

## 2026-10-24, Cache-Lehre: BUILD_DATE entwertete ALLES (Runs 12-18!)

- Korrektur eigener Statusmeldungen: "nur #41 neu" war FALSCH. `--build-arg
  BUILD_DATE=$(date)` aendert sich pro Run und stand in Zeile 10-12 -> JEDER
  Run baute ab Zeile 12 alles neu (Welt, Kernel, Batches). Belegt: nur #3/#9
  CACHED, Rest rebuilt. Die vielen "schnellen" Rebuilds waren Voll-Rebuilds.
- Fix: BUILD_*-ARGs + Label-Block ans Ende (vor Squashfs) verlegt. Ab jetzt:
  unveränderter Build = alles cached (Sekunden), Tail-Edits = nur Tail neu.
  SSH_PUBKEY-ARG war schon spät (ok). KVER-ENVs stabil (ok).

## 2026-10-24, Boot-Gate 1 ROT: Initramfs mit Sanitizer-Tools, ohne Policy

- Image baut, Gast startet Kernel, aber dracut: "Could not boot", /dev/root
  fehlt -> Debug-Shell. Kernel-Config geprueft (VIRTIO_BLK/VFAT/SQUASHFS=y),
  Platte ok (mdir, 3 Dateien) -> Initramfs-Inhalt extrahiert: mount/grep/sed
  SANITISIERT (libasan drin!), Bash plain. Ohne LSAN im Boot-Env sterben
  dmsquash-live-Skripte (set -e) an Exit-Leaks.
- Fix: `config/dracut-00-lsan.sh` (nur `export LSAN_OPTIONS=exitcode=0`) per
  `dracut --include` nach /var/lib/dracut/hooks/pre-udev/ (hookdir belegt,
  source_hook sourct VOR udev+mount). Gilt dauerhaft fuer alle Batches.
  Nur `export` (Builtin, frueh verfuegbar). COPY ins Image + --include.

## 2026-10-24, Boot-Gate 2 ROT, Ursache gefunden: udevd stirbt an dlopen-Taint

- Symptom: /dev/vda erscheint nie, dmsquash-live-root laeuft nie, 189s
  warten, "Could not boot". Kernel sieht vda, Platte ok, Regeln ok, Trigger
  ok (rd.debug-Trace, ~30k Zeilen).
- Spur: `==505==ASan runtime does not come first` exakt beim udevd-Start.
  LD_DEBUG-Beweis: udevd laedt libblkid.so.1 per dlopen (systemd-builtin),
  libblkid (Batch-01) zieht DT_NEEDED libasan.so.8 -> Runtime nicht an erster
  Stelle -> Daemon exit 1 (Docker-Repro, rc=1). Ohne Daemon keine Nodes.
- Taint-Scan: NUR util-linux-Libs betroffen (libblkid/fdisk/mount/smartcols/
  uuid), Rest sauber, Flags sauber pro Batch. Kein Rebuild noetig.
- Fix: 00-lsan-Hook setzt zusaetzlich LD_PRELOAD=libasan (Glob, Builtins),
  Docker-Beweis rc=0 + "Starting systemd-udevd". Pre-pivot-Hook 99-unset-
  preload entfernt Preload vor switch-root (echtes System plain).
- Nebenbefund (nicht boot-relevant, root-Kontext): /bin/mount ist setuid +
  ASan-gelinkt. Parken.

## 2026-10-24, Boot-Gate 3 fast da, dann Endlos-Schleife (crypt rausgeworfen)

- Gate 3 (Run 20, LD_PRELOAD-Hook): "Mounted root filesystem" OK, dann still.
  Trace (Gast-5): alles bis cleanup sauber, dann Endlos-Schleife in dracuts
  close_luks (cleanup/30-crypt-cleanup.sh): `cryptsetup luksClose` auf
  unexpandiertes Glob liefert rc=0 -> `&& do_break=n` -> ewig.
- Mechanik (wichtig fuer alle Batches!): LSAN exitcode=0 ersetzt den Exit-Code
  AUCH bei echtem Fehlschlag, sobald Exit-Leaks da sind. Fehlschlag+Leak wird
  zu Erfolg. Jede Retry-Schleife (`while`, `&&`-Kette), die auf Fehlschlag
  baut, kann sich endlos drehen. Wir nutzen kein LUKS -> Modul `crypt` aus
  dracut -m entfernt (kein cryptsetup-Hook mehr, kleineres Initramfs).
  Gast-4 war dieselbe Schleife (leise, da stderr nach /dev/null).
- Merksatz: exitcode=0 macht aus "kaputt+leckend" ein "ok". Retry-Logik in
  Boot-Skripten danach pruefen.

## 2026-10-24, Boot-Gate 4 GRUEN bis Login, aber kein SSH (Netz+Key fehlten)

- Gast-6: OpenRC startet, `gentoo login:` erreicht, 0x "Could not boot".
  Erster kompletter Sanitizer-Boot (Batch-01) bis zum Login!
- SSH-Timeout im Banner: zwei Ursachen, beide image-seitig: (1) kein Netz im
  Gast (kein net-Dienst, kein Kernel-DHCP: CONFIG_IP_PNP=n) -> Pakete ins
  Nichts; (2) kein authorized_keys (Runs 19-21 ohne SSH_PUBKEY gestartet).
- Fix: /etc/conf.d/net (dhcp), net.eth0 in default-Runlevel, UseDNS no,
  net.ifnames=0 auf Kernel-Cmdline (eth0-Name garantiert). Rebuild MIT Key.

## 2026-10-24, Run 22 rot in 2s: chown-Leck im User-Setup (Fix: LSAN-Export)

- Mit SSH_PUBKEY laeuft erstmals `chown -R` (sanitisiert, 10B Leck) im
  User-Setup -> exit 1. Ohne Key wurde der Zweig nie betreten (nie aufgefallen).
- Fix wie an anderen Build-Stellen: `export LSAN_OPTIONS=exitcode=0` zu Beginn
  der RUN-Kette (User-Setup + Netz-RUN). Lehre: JEDE RUN-Kette mit
  Sanitizer-Tools braucht den Export, sobald der Zweig wirklich laeuft.

## 2026-10-24, Boot-Gate 5 GRUEN: Gast-7 bis Login + SSH + funktionale Tests

- Run 23 (Netz+DHCPCD+Key): Gast holt 10.0.2.15, SSH als kiel geht, Login-
  Prompt, 0x "Could not boot". Erster kompletter Sanitizer-Boot.
- Gast-Beweise: ls/grep/sed/mount sanitisiert, Bash plain. sed (haLlo),
  grep (hello, Treffer+rc1-ohne), sort (a,b), mountpoint (LiveOS_rootfs),
  mount-fake rc0. Mit LSAN-Triage alle rc0.
- Gast-Env: KEIN LSAN/LD_PRELOAD nach switch-root (Pre-Pivot-Unset wirkt,
  Login-Shells sauber). Ohne LSAN melden Tools Lecks mit rc1 (sichtbar,
  gewollt streng). Kein systemweites LSAN (bewusst, spaetere Stufe).
- Fehlalarm mit Lehre: `echo hallo | grep ell` findet nichts, weil "hallo"
  kein "e" enthaelt (Tippfehler hallo/hello). Host-Plain-grep bestaetigt:
  rc1 korrekt. Lehre: Testmuster muessen den Treffer WIRKLICH enthalten;
  rc1 ohne LSAN ist mehrdeutig (kein Treffer ODER Leck) -> immer Output +
  Triage-Gegenprobe. verify-Skript verschaerft (Output-Vergleich statt -q).
- Bekannt/naechstes: udev-Dienst steht nach Pivot (0 Prozesse, dlopen-Taint
  ohne Preload). Fix bereit (conf.d/udev-Exporte), Gate in Batch-2.
- Commit bef3dc53 + Tag batch-01. Batch-2 (nur sys-apps/openrc) laeuft.

## 2026-10-24, Batch-2 rot: meson stirbt an dlopen-Taint (Preload-Umgebung)

- `emerge --oneshot` (5 Pakete) scheitert bei sys-apps/openrc/compile:
  `meson setup` stirbt lautlos (kein Output), Build-Verzeichnis fehlt,
  pushd stirbt. LSAN kam an (Umgebungs-Beweis).
- Kette: /usr/bin/meson -> python-exec2 (Skript) -> python-exec2c (nativ)
  -> python3.14 + Meson-Skript. Auch direkt mit plain Python tot.
  LD_DEBUG: python laedt _blake2/hashlib-Umfeld -> libuuid (Batch-01)
  -> libasan nicht-an-erster-Stelle -> rc1. Meson importiert hashlib/uuid
  beim Start; plain `python3 --version` lebt (kein Import).
- Fix: neue Umgebung env/sanitizers-preload (= sanitizers + LD_PRELOAD per
  `gcc -print-file-name`, Docker-Beweis meson --version rc0), Zuweisung pro
  Paket via config/batches/preload.txt (apply-batch kennt HOLD > PRELOAD).
  Minimal: nur openrc bekommt Preload, Rest unveraendert.

## 2026-10-24, Batch-2 Gate ROT mit ECHTEM FUND: OOB-Read in OpenRC librc

- Gast-8 bootet bis Login, 0x "Could not boot", aber danach startet KEIN
  Dienst (kein Netz/SSH). Ursache im Protokoll:
  `librc.c:57: runtime error: index -1 out of bounds for int[17]` +
  `ASan: global-buffer-overflow, READ, rc_dirfd <- rc_service_state <-
  service_started` ("Caching service dependencies"). Echter Upstream-Bug,
  von unseren Sanitizern gefunden (kein Leck, sondern Absturz).
- Loesungsweg: Quellen patchen (kein Opt-out, boot-relevant). Subagent
  `openrc-rcdirfd-patch` jagt Ursache + schreibt Patch nach
  config/patches/sys-apps/openrc/ (Konvention wie bash-Patches, Portage
  wendet /etc/portage/patches automatisch an). Verifikation per Neubau+Gate.

## 2026-10-24, OpenRC-Patch bereit (Subagent), Fehlalarm meinerseits inklusive

- Subagent-Befund: RC_DIR_INVALID = -1; rc_service_state() ruft rc_dirfd()
  auch fuer STOPPED/CRASHED (dir=-1) auf; Guard prueft nur `>= RC_DIR_MAX`;
  dirfds[-1] wird VOR dem switch gelesen. Unsanitisiert unsichtbar, weil der
  switch-Fall danach zufaellig -1 liefert. Entspricht upstream master.
- Patch config/patches/sys-apps/openrc/01-rcdirfd-negativ-index.patch:
  Guard `dir < 0 ||`, toter switch-Fall raus. Ich habe ihn fälschlich fuer
  abgeschnitten gehalten (3 Kontext-Zeilen am Hunk-Ende sind korrekt) und per
  Diff an der echten Quelle nachgebaut: Rumpf IDENTISCH. Datei bleibt wie
  geliefert. Lehre: Hunk-Zeilen nachzaehlen statt nach Gefuehl urteilen.
- Neubau mit Patch (Gast-8 gestoppt, RAM frei).

## 2026-10-24, SSH-Tod in zwei Akten (pam_openrc + seccomp), Fix zweigleisig

- Akt 1: sshd (plain) legt beim Schluessel-Angebot auf. LD_DEBUG: nach
  pam_elogind-Init werden libasan+librc+libeinfo gezogen. Aufloesung: es ist
  nicht elogind (statisch sauber, kein dlopen), sondern `-session optional
  pam_openrc.so` im PAM-Stapel (Batch-02, DT_NEEDED libasan). Das `-`
  (ignorieren) hilft nicht: ASan reisst den ganzen Kindprozess.
- Akt 2 (mit Preload): Anmeldung OK ("Accepted publickey"), dann stirbt das
  Kind mit Signal 31 (SIGSYS): OpenSSHs Seccomp-Filter (preauth) verbietet
  Syscalls, die libasan beim Start braucht. Docker-Beweis beidemale.
- Fix zweigleisig: (a) conf.d/sshd-Exporte (LSAN+Preload, Muster wie udev),
  (b) openssh plain aus Quellen mit `-seccomp` (package.use, eigener
  Dockerfile-Schritt mit LSAN-Export fuers configure). Stapel 3 (sanitisiertes
  openssh) prueft beides neu.
- Nebenbei: Dockerfile-Doppler (`fi ||`) beim Editieren eingebaut und sofort
  repariert (vor dem Bau geprueft).

## 2026-10-24, SSH gruen, Udev-Raetsel geloest: /.dockerenv (kein Sanitizer!)

- Gast-11: SSH geht (sshd+Preload und -seccomp bewiesen), Netz geht.
  Udev-Dienst manuell gestartet: LAEUFT (conf.d-Fix wirkt!). Aber beim
  Hochfahren lief er nie (rc-status: udev/trigger/settle/devfs/sysfs fehlen).
- Ursache: Image enthaelt /.dockerenv (Bau-Artefakt). OpenRC haelt den
  KVM-Gast fuer einen Container und ueberspringt Dienste mit
  -docker/-containers-Keywords (devfs liefert dev-mount nicht -> udev ohne
  Abhaengigkeit -> Kette steht). Betrifft ALLE Gaeste seit jeher, nichts mit
  Sanitizern zu tun. Fix: `rm -f /.dockerenv` vor mksquashfs.
- Offen/Notiz: elogind startet nicht (2x versucht, gestoppt) — eigener Hunt
  in Stapel 3 (ist dort dran).

## 2026-10-24, Udev-Blockade die zweite: rc_sys="docker" fest verdrahtet

- Gast-12 ohne /.dockerenv (bestaetigt weg), aber OpenRC meldet weiter
  `[DOCKER]` und startet kein devfs/udev. Ursache: /etc/rc.conf enthaelt
  `rc_sys="docker"` — von OpenRCs Installation beim Bauen im Behaelter fest
  geschrieben (nicht nur Datei-Erkennung!). Fix: per sed auf "" (autodetect).
  Eigene Lehre angewandt: LSAN-Export im RUN (sed/grep sanitisiert).

## 2026-10-24, Udev-Boot-Hang: Aufsicht kehrt nie zurueck -> Dienste raus

- Mit rc_sys-Autodetect startet "Starting udev", kehrt aber nie zurueck
  (kein ok/fail, kein Login dahinter). Von Hand gestartet laeuft udevd
  (Gast-11-Beweis). Hypothese: supervise-daemon wartet ewig (Notify/Pidfile),
  kein Crash (keine Sanitizer-Meldung). Kein Sanitizer-Befund, reine
  Init-System-Frage — bewusst NICHT tiefer verfolgt.
- Entscheidung (Testgast braucht keinen Hotplug/Module): udev/trigger/settle
  aus sysinit entfernt (rc-update del), manuell nutzbar. Spaetere Stapel mit
  Shell-Zugang neu bewerten, falls noetig (elogind-Kandidat).

## 2026-10-24, Udev-Del wirkte nicht: Squashfs aelter als der Befehl

- Gast-15 bleibt bei "Starting udev" stehen, obwohl rc-update del laeuft.
  Ursache: der Befehl stand am DATEIENDE, das Squashfs wird aber vorher
  gepackt — der Gast startet aus dem alten Stand. Lehre: Gast-relevante
  Aenderungen MUESSEN VOR mksquashfs stehen (Dateiende ist zu spaet).
  Behoben: Block vor mksquashfs verschoben, Doppelung am Ende entfernt.

## 2026-10-24, Tor-16/17: Dockerfile trug Udev nach dem Austragen wieder ein

- Gast-16 zeigt weiter "Starting udev" + SSH-Timeout, obwohl rc-update del
  im Bau-Protokoll "deleted" meldet. Zeitstempel-Beweis: Udev-Links im Image
  vom Bau-Tag 22:59, del lief auch. Aufloesung: Schritt "Serielle Konsole +
  minimale Dienste" (Dockerfile, NACH dem del) enthaelt
  `rc-update add udev/udev-trigger/udev-settle sysinit` und stellt alles
  wieder her. Fix: die drei add-Zeilen entfernt + Kommentar (Reaktivierung
  mit Stapel-2). Lehre: add/del-Paare im Dockerfile immer als Paar pruefen.
- Run-34 baut gruen, Image-Startliste ohne Udev (rc-update show bestaetigt).

## 2026-10-24, SSH-Schluessel fehlte: Bau ohne Umgebungs-Variable

- Gast-17 bootet sauber (0x "Starting udev", Login), aber SSH
  "Permission denied": /home/kiel/.ssh/authorized_keys fehlt im Image.
  Ursache: setup01 braucht SSH_PUBKEY aus der Umgebung; mein Run-34 lief
  ohne. Fix: Rueckfallkette in setup01_build_image.sh (Umgebung, dann
  /tmp/qemu-guest-key.pub, dann ~/.ssh/authorized_keys). Syntax-geprueft.
- Run-35 gruen, Docker-Pruefung 13/13 (FAIL=0), Gast-18: Boot ohne Udev,
  SSH-OK, Startliste sauber, sed/grep/mount-Funktionstest gruen,
  /bin/sed gegen libasan gelinkt.
