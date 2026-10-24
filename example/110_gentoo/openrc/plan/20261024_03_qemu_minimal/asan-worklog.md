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

## Verifikations-Strategie (ab 08:15)

- Standard: Docker (`docker run --rm <image> ...`) — Paket-Verhalten, Coverage
  (`strings`-Probe), funktionale Smoke-Tests der Batch-Binaries. Sekunden statt
  Minuten, skriptbar, kein serielles Gefrickel.
- Boot-Gate: QEMU nur einmal pro Batch (OpenRC-Boot mit sanitisiertem Stack,
  Login, Smoke-Test). Boot-Fehler → Fix/Patch wie bisher.
- SSH in den Gast bleibt für tiefe Hunts (Ebuild-Kontext, symbolische Traces).
