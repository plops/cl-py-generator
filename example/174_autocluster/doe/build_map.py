"""Jobs/Store/Plotly-Karte/Labels fuers Produktions-Clustering (Phase-B-Winner).

Winner: HDBSCAN mcs=11/ms=8 auf d=11/nn=45/md=0.09 (k=3072, seed=42).
Schritte: jobs (Titel-Jobs + 12 Batches) | store (Titel mergen+validieren)
| html (Plotly-Karte, riesig, NICHT committen) | labels | demo (Update-Demo).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DB = os.path.join(ROOT, "summaries_compact_20260924.db")
WINNER = {"method": "hdbscan", "mcs": 11, "ms": 8, "d": 11, "n_neighbors": 45,
          "min_dist": 0.09, "seed": 42, "k": 3072}
EMB_KEY = "pb_emb_d11_s42.npy"
EMB2_KEY = "pb_emb_d02_s42.npy"
LABELS = os.path.join(HERE, "results", "labels_pb_hdbscan.npy")
N_BATCH = 12
N_EX = 8
N_NB = 3
N_NB_EX = 4


def load_prod(db_path):
    from data import load_full, truncate_norm

    base = load_full(db_path)
    X, _ = truncate_norm(base["X"], 3072)
    Xe = np.load(os.path.join(HERE, "umap_cache_doe", EMB_KEY))
    labels = np.load(LABELS)
    return base, X, np.asarray(Xe), np.asarray(labels)


def excerpt(text, n):
    return " ".join(str(text or "").split())[:n]


def step_jobs(a):
    rng = np.random.default_rng(7)
    base, X, Xe, labels = load_prod(a.db)
    ids = np.array(base["ids"])
    clusters = sorted(set(labels) - {-1})
    cents = {c: Xe[labels == c].mean(axis=0) for c in clusters}
    jobs = []
    for c in clusters:
        m = np.where(labels == c)[0]
        d0 = np.linalg.norm(Xe[m] - cents[c], axis=1)
        near = m[np.argsort(d0)[:5]]
        rest = np.setdiff1d(m, near)
        extra = rng.choice(rest, min(3, len(rest)),
                           replace=False) if len(rest) else []
        ex_idx = list(near) + [int(i) for i in extra]
        nbs = []
        dists = sorted(
            ((float(np.linalg.norm(cents[c] - cents[o])), o)
             for o in clusters if o != c))[:N_NB]
        for dist, o in dists:
            mo = np.where(labels == o)[0]
            d1 = np.linalg.norm(Xe[mo] - cents[o], axis=1)
            nbs.append({
                "cluster": int(o), "dist": round(dist, 3),
                "size": int(mo.size),
                "exemplars": [{
                    "identifier": int(ids[i]),
                    "excerpt": excerpt(base["summaries"][i], 150)}
                    for i in mo[np.argsort(d1)[:N_NB_EX]]]})
        jobs.append({
            "cluster": int(c), "size": int(m.size),
            "exemplars": [{
                "identifier": int(ids[i]), "link": base["links"][i],
                "excerpt": excerpt(base["summaries"][i], 350)}
                for i in ex_idx],
            "neighbors": nbs})
    out = os.path.join(HERE, "results", "title_jobs.json")
    json.dump({"meta": {**WINNER, "db": os.path.basename(a.db),
                        "n_clusters": len(jobs)},
               "jobs": jobs}, open(out, "w"), ensure_ascii=False)
    print("wrote %s (%d jobs)" % (out, len(jobs)), flush=True)
    for b in range(N_BATCH):
        part = jobs[b::N_BATCH]
        p = os.path.join(HERE, "results", "title_batch_%02d.json" % b)
        json.dump({"batch": b, "jobs": part}, open(p, "w"),
                  ensure_ascii=False)
    print("wrote %d batches" % N_BATCH, flush=True)


def step_store(a):
    from titles import make_store, save_store

    jobs = json.load(open(os.path.join(HERE, "results", "title_jobs.json")))
    base, _, _, labels = load_prod(a.db)
    ids = np.array(base["ids"])
    titles, exemplars, neighbors = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(
            HERE, "results", "title_titles_batch_*.json"))):
        for t in json.load(open(path))["titles"]:
            titles[int(t["cluster"])] = t["title"]
    for job in jobs["jobs"]:
        c = job["cluster"]
        exemplars[c] = [e["identifier"] for e in job["exemplars"]]
        neighbors[c] = [n["cluster"] for n in job["neighbors"]]
    missing = sorted(set(int(j["cluster"]) for j in jobs["jobs"])
                     - set(titles))
    if missing:
        raise SystemExit("FEHLENDE TITEL: %s" % missing)
    seen, dups = {}, []
    for c, t in titles.items():
        if t in seen:
            dups.append((c, seen[t], t))
        seen.setdefault(t, c)
    if dups:
        print("WARNUNG doppelte Titel: %s" % dups, flush=True)
    long_ = [(c, len(t)) for c, t in titles.items() if len(t) > 80]
    if long_:
        print("WARNUNG lange Titel (>80): %s" % long_, flush=True)
    assign = {int(i): int(l) for i, l in zip(ids, labels)}
    store = make_store({**WINNER, "db": os.path.basename(a.db)}, assign,
                       titles, exemplars, neighbors)
    out = os.path.join(ROOT, "cluster_titles_phaseb.json")
    save_store(store, out)
    print("wrote %s (%d Titel)" % (out, len(titles)), flush=True)


def step_html(a):
    import plotly.graph_objects as go

    store = json.load(open(os.path.join(ROOT, "cluster_titles_phaseb.json")))
    base, X, _, labels = load_prod(a.db)
    p2 = os.path.join(HERE, "umap_cache_doe", EMB2_KEY)
    if os.path.exists(p2):
        X2 = np.load(p2)
    else:
        import cupy as cp

        from experiment import run_umap_seed

        print("fit 2D-UMAP ...", flush=True)
        X2 = cp.asnumpy(run_umap_seed(
            cp.asarray(X), {"d": 2, "n_neighbors": WINNER["n_neighbors"],
                            "min_dist": WINNER["min_dist"]}, WINNER["seed"]))
        cp.save(p2, X2)
    X2 = np.asarray(X2)
    ids = np.array(base["ids"])
    summ = {int(i): (excerpt(base["summaries"][k], 220), base["links"][k])
            for k, i in enumerate(ids)}
    fig = go.Figure()
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        t = store["titles"].get(str(int(c)), {}).get("title", "?")
        name = "%d %s (%d)" % (c, t, m.sum())
        hover = ["<b>%s</b><br>%s<br><a href='%s'>video</a><br>%s" % (
            t, ident, summ.get(int(ident), ("", ""))[1],
            summ.get(int(ident), ("", ""))[0]) for ident in ids[m]]
        fig.add_trace(go.Scattergl(x=X2[m, 0], y=X2[m, 1], mode="markers",
                                   name=name, text=hover, hoverinfo="text",
                                   marker=dict(size=3, opacity=0.7)))
    n = labels == -1
    fig.add_trace(go.Scattergl(x=X2[n, 0], y=X2[n, 1], mode="markers",
                               name="Noise (%d)" % n.sum(), hoverinfo="skip",
                               marker=dict(size=2, color="lightgrey")))
    fig.update_layout(
        title="Video-Summary-Cluster (k=3072, UMAP d11, HDBSCAN mcs11/ms8)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(itemsizing="constant"), dragmode="zoom",
        hovermode="closest")
    out = os.path.join(ROOT, "plots", "clusters.html")
    fig.write_html(out, include_plotlyjs=True)
    print("wrote %s (%d traces) — NICHT committen (riesig, ignoriert)" % (
        out, len(fig.data)), flush=True)


def step_labels(a):
    base, _, _, labels = load_prod(a.db)
    out = os.path.join(ROOT, "plots", "labels_phaseb.csv")
    np.savetxt(out, np.column_stack([np.array(base["ids"]), labels]),
               fmt="%d", header="identifier,cluster", comments="")
    print("wrote %s" % out, flush=True)


def step_demo(a):
    """Update-Demo: Seed-1337-Reclustering als 'Zukunft' gegen den Store."""
    import cupy as cp

    from experiment import run_umap_seed
    from run_phaseb import cluster_labels
    from titles import load_store, plan_update

    store = load_store(os.path.join(ROOT, "cluster_titles_phaseb.json"))
    base, X, _, _ = load_prod(a.db)
    p = os.path.join(HERE, "umap_cache_doe", "pb_emb_d11_s1337.npy")
    Xe = np.load(p) if os.path.exists(p) else cp.asnumpy(run_umap_seed(
        cp.asarray(X), {"d": 11, "n_neighbors": 45, "min_dist": 0.09}, 1337))
    lab = cluster_labels("hdbscan", np.asarray(Xe), cp.asarray(Xe),
                         {"mcs": 11, "ms": 8})
    new_assign = {int(i): int(l) for i, l in zip(base["ids"], lab)}
    keep, retitle = plan_update(store, new_assign)
    print("demo: %d keep, %d retitle (von %d neuen Clustern)" % (
        len(keep), len(retitle),
        len(set(int(l) for l in lab if l != -1))), flush=True)
    js = sorted((v["jaccard"] for v in keep.values()))
    print("demo: Jaccard keep median=%.3f min=%.3f" % (
        float(np.median(js)) if js else float("nan"),
        min(js) if js else float("nan")), flush=True)
    print("demo: retitle-Cluster (neu, best-alt, J): %s" % (
        [(r["new"], r["best_old"], r["jaccard"]) for r in retitle][:10]),
        flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--step", default="all",
                    choices=["all", "jobs", "store", "html", "labels",
                             "demo"])
    a = ap.parse_args()
    steps = ([a.step] if a.step != "all" else
             ["jobs", "store", "html", "labels", "demo"])
    for s in steps:
        {"jobs": step_jobs, "store": step_store, "html": step_html,
         "labels": step_labels, "demo": step_demo}[s](a)


if __name__ == "__main__":
    main()
