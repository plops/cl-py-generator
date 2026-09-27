"""Titel-Store mit Content-Hash + inkrementeller Update-Planung (Phase B+).

Problem: LLM-Betitelung ist teuer (pro Cluster Samples + Nachbarn lesen).
Loesung: Titel-Store mit Member-Signaturen ueber stabilen DB-Identifiers;
bei Re-Clustering matched Jaccard alte→neue Cluster — nur wesentlich
geaenderte (Jaccard < Schwelle) oder neue Cluster werden neu betitelt.
v1 ist membership-basiert: Nachbar-Titel fliessen als Job-Kontext ein,
triggern aber selbst kein Retitling (s. TITLING_de.md).
"""

import hashlib
import json

STORE_VERSION = 1
KEEP_JACCARD = 0.7


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def member_sig(member_identifiers):
    """Signatur der Cluster-Mitgliedschaft (Identifiers, reihenfolgefrei)."""
    return _sha1(",".join(
        str(i) for i in sorted(set(int(i) for i in member_identifiers))))


def job_sig(exemplar_identifiers, neighbor_cids):
    """Signatur des Titel-Jobs (was das LLM sah)."""
    return _sha1("ex:" + ",".join(
        str(i) for i in sorted(set(int(i) for i in exemplar_identifiers))
    ) + "|nb:" + ",".join(str(int(c)) for c in sorted(neighbor_cids)))


def make_store(meta, assign, titles, exemplars, neighbors):
    """Store bauen. assign: {identifier: cluster}; titles/exemplars/neighbors
    je Cluster (neighbors: Nachbar-Cluster-IDs). Noise (-1) ohne Titel."""
    clusters = {}
    members = {}
    for ident, c in assign.items():
        if int(c) != -1:
            members.setdefault(int(c), []).append(int(ident))
    for c, mids in members.items():
        ex = [int(i) for i in exemplars.get(c, exemplars.get(str(c), []))]
        nb = [int(i) for i in neighbors.get(c, neighbors.get(str(c), []))]
        clusters[str(c)] = {
            "title": titles.get(c, titles.get(str(c), "")),
            "n": len(mids),
            "member_sig": member_sig(mids),
            "job_sig": job_sig(ex, nb),
            "members": sorted(int(i) for i in mids),
            "exemplars": sorted(ex),
            "neighbors": sorted(nb),
        }
    return {"version": STORE_VERSION, "meta": meta, "titles": clusters}


def save_store(store, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1)


def load_store(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match_clusters(old_assign, new_assign):
    """Alte→neue Cluster per Jaccard ueber Identifier-Mengen matchen.

    assign: {identifier: cluster}, -1 = Noise (ausgeschlossen).
    Returns {new_c: {old, jaccard, n_new, n_old, n_overlap}}; old=None,
    wenn keine Ueberlappung mit einem betitelten alten Cluster existiert.
    """
    old_members = {}
    for ident, c in old_assign.items():
        if int(c) != -1:
            old_members.setdefault(int(c), set()).add(int(ident))
    new_members = {}
    for ident, c in new_assign.items():
        if int(c) != -1:
            new_members.setdefault(int(c), set()).add(int(ident))
    old_of = {}
    for c, mids in old_members.items():
        for i in mids:
            old_of[i] = c
    out = {}
    for nc, nmids in new_members.items():
        overlap = {}
        for i in nmids:
            if i in old_of:
                oc = old_of[i]
                overlap[oc] = overlap.get(oc, 0) + 1
        best, bj = None, 0.0
        for oc, inter in overlap.items():
            union = len(nmids) + len(old_members[oc]) - inter
            j = inter / union if union else 0.0
            if j > bj:
                best, bj = oc, j
        out[nc] = {"old": best, "jaccard": round(bj, 4), "n_new": len(nmids),
                   "n_old": len(old_members[best]) if best is not None else 0,
                   "n_overlap": overlap.get(best, 0) if best is not None
                   else 0}
    return out


def plan_update(store, new_assign, keep_threshold=KEEP_JACCARD):
    """Update-Plan: keep (Titel+Signatur uebernehmen) vs. retitle (Jobs).

    keep iff Jaccard >= Schwelle UND alter Titel vorhanden UND nicht leer.
    Gibt (keep_dict, retitle_list) zurueck; keep: {new_c: {...Titel, von, j}}.
    """
    old_assign = {}
    for c, t in store.get("titles", {}).items():
        for i in t.get("members", []):
            old_assign[int(i)] = int(c)
    if not old_assign:
        # Kompakt-Store ohne Memberlisten: alles neu betiteln.
        new_cs = sorted(set(int(c) for c in new_assign.values()
                            if int(c) != -1))
        return {}, [{"new": c} for c in new_cs]
    match = match_clusters(old_assign, new_assign)
    keep, retitle = {}, []
    for nc, m in match.items():
        old = m["old"]
        title = (store["titles"].get(str(old), {}).get("title", "")
                 if old is not None else "")
        if old is not None and m["jaccard"] >= keep_threshold and title:
            keep[nc] = {"title": title, "from_old": old,
                        "jaccard": m["jaccard"]}
        else:
            retitle.append({"new": nc, "best_old": old,
                            "jaccard": m["jaccard"]})
    return keep, retitle
