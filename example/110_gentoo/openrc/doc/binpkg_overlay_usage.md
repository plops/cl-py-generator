# Binpkgs aus dem OverlayFS nachinstallieren (`emerge-binpkg`)

Der Export neben den Squashfs-Images enthält zusätzlich:

- `binpkgs/` — nur **echt lokal gebaute** Binpkgs (`@world` + beim Bau
  gezogene BDEPENDs, z. B. `dev-lang/rust-bin`, `dev-lang/go`-Abhängigkeiten
  aus Quellbuilds). Erzeugt von `config/extract_local_binpkgs.py` im Builder
  **vor** `quickpkg`/`genlop` (Filter: `/var/log/emerge.log`, letzte
  Merge-Art gewinnt; `BINPKGMD5` ist kein Kriterium).
- `upstream_binpkgs.tsv` / `upstream_binpkgs.txt` — alle übrigen installierten
  Pakete mit Upstream-Herkunft (`sync-uri` + exakte URL aus dem
  Gentoo-Binhost-`Packages`-Index bzw. Stage3-Vermerk).

`install_on_e14.sh` kopiert `binpkgs/` bewusst **nicht** auf `/1p2` (Platz).

## Benutzung im laufenden OverlayFS

Das Binpkg-Verzeichnis wird **nicht automatisch gemountet**. Einmalig, z. B. vom
USB-Stick:

```bash
export BINPKG_DIR=/pfad/zu/gentoo-z6-min-openrc_YYYYMMDD/binpkgs
sudo -E emerge-binpkg --list
```

| Modus | Befehl | Was passiert |
| :--- | :--- | :--- |
| Hybrid (Default) | `sudo -E emerge-binpkg dev-lang/rust-bin` | Lokale PKGDIR ohne Signaturzwang + Upstream-Binhost via `binrepos.conf` **mit** Signaturprüfung (Keys baked im Image unter `/etc/portage/gnupg`) |
| Strikt lokal | `sudo -E emerge-binpkg --offline dev-build/cmake` | `FEATURES="-getbinpkg -binpkg-request-signature"` — kein Netz, nur `binpkgs/` |
| Entfernen | `sudo emerge-binpkg --remove --yes dev-lang/rust-bin` | `emerge -C` + `emerge --depclean` (schützt `@world` ohne `--force`) |

Der Wrapper liegt im Image unter `/usr/local/sbin/emerge-binpkg`
(`emerge-binpkg --help`).

## Zwischen `firefox-bin` und Quell-`firefox` wechseln

In `config/var/lib/portage/world` genau **eine** der beiden Zeilen
aktivieren, danach neu bauen:

```text
www-client/firefox-bin
#www-client/firefox
```

`firefox-bin` spart ~20 min Bauzeit (kein `clang`/`rust` nötig);
Quell-`firefox` erlaubt eigene USE-Flags (siehe `config/etc/portage/package.use`,
Abschnitt „Source-Firefox-Flags“).
