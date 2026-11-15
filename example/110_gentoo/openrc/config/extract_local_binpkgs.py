#!/usr/bin/env python3
"""Exportiert nur echt lokal gebaute @world/BDEPEND-Binpkgs aus /var/cache/binpkgs
und dokumentiert alle per Upstream-Binhost oder Stage3 bezogenen Pakete.

Klassifikation (Ground Truth ist /var/log/emerge.log, letzte Merge-Art gewinnt):
- Quelle (Compiling/Packaging bzw. Compiling/Merging) -> lokal gebaut,
  unabhaengig von BINPKGMD5 (Portage schreibt BINPKGMD5 auch bei
  FEATURES=buildpkg aus Quellbuilds).
- "Merging Binary" -> Upstream-Binhost (exakte Datei liegt im
  Binhost-Cache /var/cache/binhost).
- Kein Log-Eintrag -> aus dem Stage3-Tarball (Basis), nie gemergt.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import os
import re
import shutil
import urllib.request
from pathlib import Path


def parse_emerge_log(log_path: Path) -> tuple[dict[str, str], set[str]]:
    """Liest /var/log/emerge.log und liefert:
    - last_mode[cpv]: 'source' (Compiling/...) oder 'binary' (Merging Binary)
    - genlop_only_cpvs: CPVs, die erst nach '--oneshot app-portage/genlop' gemerged wurden.
    """
    last_mode: dict[str, str] = {}
    pre_genlop_cpvs: set[str] = set()
    post_genlop_cpvs: set[str] = set()
    in_genlop_phase = False

    re_cmd = re.compile(r"^\d+:\s+\*\*\*\s+emerge\s+(.*)$")
    # Quellbuilds loggen "Compiling/Packaging" (modern) bzw. "Compiling/Merging" (alt).
    re_compile = re.compile(
        r"^\d+:\s+===\s+\(\d+\s+of\s+\d+\)\s+Compiling/(?:Packaging|Merging)\s+\(([^:]+)::"
    )
    # Binaer-Merges loggen "Merging Binary (<cpv>::<pfad-zur-gpkg>)".
    re_binary = re.compile(
        r"^\d+:\s+===\s+\(\d+\s+of\s+\d+\)\s+Merging\s+Binary\s+\(([^:]+)::"
    )

    if not log_path.is_file():
        return last_mode, set()

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m_cmd = re_cmd.match(line)
        if m_cmd:
            args = m_cmd.group(1)
            if "app-portage/genlop" in args and "--oneshot" in args:
                in_genlop_phase = True
            elif "x11-misc/xkeyboard-config" in args:
                in_genlop_phase = False
            continue

        m_c = re_compile.match(line)
        if m_c:
            cpv = m_c.group(1)
            last_mode[cpv] = "source"
            (post_genlop_cpvs if in_genlop_phase else pre_genlop_cpvs).add(cpv)
            continue

        m_b = re_binary.match(line)
        if m_b:
            cpv = m_b.group(1)
            last_mode[cpv] = "binary"
            (post_genlop_cpvs if in_genlop_phase else pre_genlop_cpvs).add(cpv)

    genlop_only = post_genlop_cpvs - pre_genlop_cpvs
    return last_mode, genlop_only


def read_binrepos(conf_dir: Path) -> list[dict[str, str]]:
    """Liest /etc/portage/binrepos.conf/*.conf sortiert nach absteigender Prioritaet."""
    repos: list[dict[str, str]] = []
    if not conf_dir.is_dir():
        return repos

    cp = configparser.ConfigParser()
    for p in sorted(conf_dir.glob("*.conf")):
        cp.read(p, encoding="utf-8")

    for sec in cp.sections():
        if sec == "DEFAULT":
            continue
        sync_uri = cp.get(sec, "sync-uri", fallback="").strip()
        if not sync_uri:
            continue
        location = cp.get(sec, "location", fallback=f"/var/cache/binhost/{sec}").strip()
        try:
            priority = int(cp.get(sec, "priority", fallback="0").strip())
        except ValueError:
            priority = 0
        repos.append({
            "name": sec,
            "sync_uri": sync_uri.rstrip("/"),
            "location": location,
            "priority": str(priority),
            "_prio": priority,
        })

    repos.sort(key=lambda r: int(r["_prio"]), reverse=True)
    return repos


def parse_packages_index(index_path: Path) -> list[dict[str, str]]:
    """Parst eine Gentoo-Binhost-Packages-Datei in eine Liste von Stanza-Dicts."""
    if not index_path.is_file():
        return []
    stanzas: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for raw in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            if "CPV" in cur:
                stanzas.append(cur)
            cur = {}
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            cur[k.strip()] = v.strip()
    if "CPV" in cur:
        stanzas.append(cur)
    return stanzas


def fetch_packages_index(sync_uri: str, dest: Path, timeout: int = 120) -> bool:
    """Laedt <sync_uri>/Packages herunter (fuer Upstream-URLs/Groessen/Hashes)."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            f"{sync_uri}/Packages", headers={"User-Agent": "extract_local_binpkgs/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        return dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001 - reiner Info-Fallback
        print(f"WARNUNG: Packages-Index von {sync_uri} nicht ladbar: {exc}")
        return False


def stanza_url(sync_uri: str, stanza: dict[str, str]) -> str:
    rel = stanza.get("PATH", "").strip()
    if not rel:
        cpv = stanza["CPV"]
        cat, pf = cpv.split("/", 1)
        pn = re.sub(r"-[0-9].*$", "", pf)
        bid = stanza.get("BUILD_ID", "").strip()
        fmt = stanza.get("BINPKG_FORMAT", "gpkg").strip()
        ext = ".gpkg.tar" if fmt == "gpkg" else ".tbz2"
        if bid:
            rel = f"{cat}/{pn}/{pf}-{bid}{ext}"
        else:
            rel = f"{cat}/{pn}/{pf}{ext}"
    return f"{sync_uri}/{rel.lstrip('/')}"


def find_local_binpkg(pkgdir: Path, cat: str, pf: str) -> Path | None:
    """Sucht passendes .gpkg.tar im PKGDIR; bei mehreren BUILD_IDs gewinnt die neueste."""
    cat_dir = pkgdir / cat
    if not cat_dir.is_dir():
        return None
    # Portage legt unter <cat>/<pn>/<pf>-<build_id>.gpkg.tar ab (oder direkt <cat>/<pf>.gpkg.tar)
    candidates: list[tuple[int, float, Path]] = []
    pattern = re.compile(rf"^{re.escape(pf)}(?:-(\d+))?\.gpkg\.tar$")
    for root_dir in [cat_dir] + [d for d in cat_dir.iterdir() if d.is_dir()]:
        for f in root_dir.iterdir():
            if not f.is_file():
                continue
            m = pattern.match(f.name)
            if m:
                bid = int(m.group(1)) if m.group(1) else 0
                candidates.append((bid, f.stat().st_mtime, f))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def find_cached_upstream_file(
    binhost_cache: Path, repos: list[dict[str, str]], cat: str, pf: str
) -> tuple[dict[str, str], Path] | tuple[None, None]:
    """Sucht die exakt heruntergeladene Upstream-.gpkg.tar im Binhost-Cache."""
    pattern = re.compile(rf"^{re.escape(pf)}(?:-(\d+))?\.gpkg\.tar$")
    for repo in repos:
        base = binhost_cache / Path(repo["location"]).name
        cat_dir = base / cat
        if not cat_dir.is_dir():
            continue
        roots = [cat_dir] + [d for d in cat_dir.iterdir() if d.is_dir()]
        hits: list[tuple[int, Path]] = []
        for root_dir in roots:
            for f in root_dir.iterdir():
                if not f.is_file():
                    continue
                m = pattern.match(f.name)
                if m:
                    hits.append((int(m.group(1)) if m.group(1) else 0, f))
        if hits:
            hits.sort(key=lambda t: t[0], reverse=True)
            best = hits[0][1]
            return repo, best
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkgdir", default="/var/cache/binpkgs")
    ap.add_argument("--vdb", default="/var/db/pkg")
    ap.add_argument("--emerge-log", default="/var/log/emerge.log")
    ap.add_argument("--binrepos-conf", default="/etc/portage/binrepos.conf")
    ap.add_argument("--binhost-cache", default="/var/cache/binhost")
    ap.add_argument("--out-binpkgs", required=True)
    ap.add_argument("--out-upstream-tsv", required=True)
    ap.add_argument("--out-upstream-txt", required=True)
    ap.add_argument(
        "--no-fetch-index",
        action="store_true",
        help="Kein Download des Upstream-Packages-Index (nur lokaler Cache)",
    )
    args = ap.parse_args()

    pkgdir = Path(args.pkgdir)
    vdb = Path(args.vdb)
    binhost_cache = Path(args.binhost_cache)
    out_binpkgs = Path(args.out_binpkgs)
    out_tsv = Path(args.out_upstream_tsv)
    out_txt = Path(args.out_upstream_txt)

    if out_binpkgs.exists():
        shutil.rmtree(out_binpkgs)
    out_binpkgs.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    last_mode, genlop_only = parse_emerge_log(Path(args.emerge_log))
    repos = read_binrepos(Path(args.binrepos_conf))

    # Upstream-Index laden: zuerst lokale Packages-Datei, sonst Download vom sync-uri.
    by_cpv: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    by_cp: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    tmp_idx_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "extract_binpkgs_idx"
    for repo in repos:
        idx_path = Path(repo["location"]) / "Packages"
        if not idx_path.is_file() and not args.no_fetch_index:
            dl_path = tmp_idx_dir / f"Packages.{repo['name']}"
            if fetch_packages_index(repo["sync_uri"], dl_path):
                idx_path = dl_path
        for st in parse_packages_index(idx_path):
            cpv = st["CPV"]
            by_cpv.setdefault(cpv, []).append((repo, st))
            cat, pf = cpv.split("/", 1)
            pn = re.sub(r"-[0-9].*$", "", pf)
            by_cp.setdefault(f"{cat}/{pn}", []).append((repo, st))

    counts = {
        "local_built": 0,
        "upstream_binhost": 0,
        "upstream_binhost_cache": 0,
        "stage3_superseded": 0,
        "stage3_only": 0,
        "skipped_genlop": 0,
        "skipped_virtual": 0,
        "missing_local_file": 0,
    }
    local_bytes = 0
    upstream_rows: list[tuple[str, str, str, str, str, str, str]] = []

    for cat_dir in sorted(p for p in vdb.iterdir() if p.is_dir()):
        cat = cat_dir.name
        for pkg_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            pf = pkg_dir.name
            cpv = f"{cat}/{pf}"

            if cat == "virtual":
                counts["skipped_virtual"] += 1
                continue
            if cpv in genlop_only:
                counts["skipped_genlop"] += 1
                continue

            mode = last_mode.get(cpv, "")
            # Hinweis: BINPKGMD5 ist KEIN Kriterium — Portage schreibt es auch
            # bei FEATURES=buildpkg aus Quellbuilds. Nur das Log zaehlt.
            is_locally_compiled = mode == "source"

            if is_locally_compiled:
                src_file = find_local_binpkg(pkgdir, cat, pf)
                if src_file is None:
                    counts["missing_local_file"] += 1
                    raise SystemExit(f"FEHLER: Lokal gebautes Paket ohne .gpkg.tar in {pkgdir}: {cpv}")
                rel_sub = src_file.relative_to(pkgdir)
                dst_file = out_binpkgs / rel_sub
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                counts["local_built"] += 1
                local_bytes += dst_file.stat().st_size
                continue

            use_val = (pkg_dir / "USE").read_text(encoding="utf-8", errors="replace").strip() if (pkg_dir / "USE").is_file() else ""
            md5_val = (pkg_dir / "BINPKGMD5").read_text(encoding="utf-8", errors="replace").strip() if (pkg_dir / "BINPKGMD5").is_file() else ""
            pn = re.sub(r"-[0-9].*$", "", pf)
            cp = f"{cat}/{pn}"

            if cpv in by_cpv:
                chosen_repo, chosen_st = by_cpv[cpv][0]
                if md5_val:
                    for r, st in by_cpv[cpv]:
                        if st.get("MD5", "") == md5_val:
                            chosen_repo, chosen_st = r, st
                            break
                url = stanza_url(chosen_repo["sync_uri"], chosen_st)
                upstream_rows.append((
                    cpv,
                    "upstream_binhost",
                    chosen_repo["name"],
                    url,
                    chosen_st.get("SIZE", "0"),
                    chosen_st.get("SHA512", "-"),
                    use_val,
                ))
                counts["upstream_binhost"] += 1
            elif cp in by_cp:
                chosen_repo, chosen_st = by_cp[cp][-1]
                url = stanza_url(chosen_repo["sync_uri"], chosen_st)
                upstream_rows.append((
                    cpv,
                    f"stage3_superseded_by_{chosen_st['CPV']}",
                    chosen_repo["name"],
                    url,
                    chosen_st.get("SIZE", "0"),
                    chosen_st.get("SHA512", "-"),
                    use_val,
                ))
                counts["stage3_superseded"] += 1
            else:
                # Exakte Upstream-Datei aus dem Binhost-Cache (byte-identisch).
                cache_repo, cache_file = find_cached_upstream_file(
                    binhost_cache, repos, cat, pf
                )
                if cache_repo is not None and cache_file is not None:
                    rel = cache_file.relative_to(
                        Path(cache_repo["location"])
                        if Path(cache_repo["location"]).is_absolute()
                        else binhost_cache / Path(cache_repo["location"]).name
                    )
                    url = f"{cache_repo['sync_uri']}/{rel.as_posix()}"
                    upstream_rows.append((
                        cpv,
                        "upstream_binhost_cache",
                        cache_repo["name"],
                        url,
                        str(cache_file.stat().st_size),
                        "-",
                        use_val,
                    ))
                    counts["upstream_binhost_cache"] += 1
                else:
                    upstream_rows.append((cpv, "stage3_only", "-", "-", "0", "-", use_val))
                    counts["stage3_only"] += 1

    with out_tsv.open("w", encoding="utf-8") as fh:
        fh.write("cpv\torigin\tbinrepo\turl\tsize_bytes\tsha512\tuse\n")
        for row in upstream_rows:
            fh.write("\t".join(row) + "\n")

    with out_txt.open("w", encoding="utf-8") as fh:
        fh.write("Gentoo Builder Package Origin Summary\n")
        fh.write("=====================================\n")
        for k, v in counts.items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"local_binpkgs_bytes: {local_bytes} ({local_bytes / (1024*1024):.1f} MiB)\n\n")
        fh.write(f"{'CPV':<45} {'ORIGIN':<32} {'BINREPO':<16} URL\n")
        for cpv, origin, repo, url, *_ in upstream_rows:
            fh.write(f"{cpv:<45} {origin:<32} {repo:<16} {url}\n")

    print(f"extract_local_binpkgs: {counts}, local_mib={local_bytes / (1024*1024):.1f}")
    if counts["local_built"] == 0:
        raise SystemExit("FEHLER: 0 lokale Binpkgs gefunden!")


if __name__ == "__main__":
    main()
