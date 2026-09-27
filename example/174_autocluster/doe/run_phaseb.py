"""Phase-B-Treiber: Methoden-Bake-off + Stabilitaet + Subsamples + Kohaerenz.

Bake-off auf EINER fixen Einbettung (d=11, nn=45, md=0.09, k=3072, seed=42):
HDBSCAN (15) × DBSCAN (12) × Leiden (16) × Agglo (8) Tuning-Runs, Response
stets Silhouette_orig × (1-Noise) auf Voll-N. Danach: Winner-Stabilitaet
(3 Seeds), Subsample-Persistenz (80/90 % vs. voll), pro-Cluster-NPMI +
Kohäsion (externe Validitaet: r über alle Cluster) und verblindete
Author-Rating-Samples (Units ohne Methode + versiegelter Key).
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "summaries_compact_20260924.db")
EMB = {"d": 11, "n_neighbors": 45, "min_dist": 0.09}
SEEDS = [42, 1337, 2026]
SUB_FRACS = [0.8, 0.9]

GRID = {
    "hdbscan": [{"mcs": a, "ms": b} for a in (8, 11, 15, 20, 30)
                for b in (5, 8, 15)],
    "dbscan": [{"eps": e, "ms": m} for e in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
               for m in (10, 25)],
    "leiden": [{"knn": k, "res": r} for k in (15, 30)
               for r in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0,
                         5.0, 8.0)],
    "agglo": [{"link": l, "nc": n} for l in ("average", "ward")
              for n in (75, 150, 225, 300)],
}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


def r4(v):
    return round(float(v), 4) if isinstance(v, float) else v


def umap_fit(X_gpu, seed, cache, tag):
    import cupy as cp

    from experiment import run_umap_seed

    path = os.path.join(cache, "pb_emb_%s_s%d.npy" % (tag, seed))
    if os.path.exists(path):
        return cp.asarray(np.load(path))
    Xr = run_umap_seed(X_gpu, EMB, seed)
    cp.save(path, Xr)
    return Xr


def cluster_labels(method, X_emb_cpu, X_emb_gpu, p, seed=42):
    """Labels je Methode auf gegebener Einbettung (CPU- oder GPU-Array)."""
    if method == "hdbscan":
        from cuml.cluster import HDBSCAN

        lab = HDBSCAN(min_cluster_size=p["mcs"],
                      min_samples=p["ms"]).fit_predict(X_emb_gpu)
        return np.asarray(lab.get())
    if method == "dbscan":
        from cuml.cluster import DBSCAN

        lab = DBSCAN(eps=p["eps"],
                     min_samples=p["ms"]).fit_predict(X_emb_gpu)
        return np.asarray(lab.get())
    if method == "leiden":
        from sklearn.neighbors import kneighbors_graph
        import igraph as ig
        import leidenalg

        a = kneighbors_graph(X_emb_cpu, p["knn"], mode="connectivity",
                             include_self=False, n_jobs=-1)
        e = a.maximum(a.T).tocoo()
        g = ig.Graph(n=X_emb_cpu.shape[0],
                     edges=list(zip(e.row.tolist(), e.col.tolist())))
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=p["res"], seed=seed)
        lab = np.empty(X_emb_cpu.shape[0], dtype=int)
        for cid, members in enumerate(part):
            lab[members] = cid
        return lab
    if method == "agglo":
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        con = kneighbors_graph(X_emb_cpu, 30, mode="connectivity",
                               include_self=False, n_jobs=-1)
        return AgglomerativeClustering(
            n_clusters=p["nc"], linkage=p["link"], connectivity=con.maximum(
                con.T)).fit_predict(X_emb_cpu)
    raise ValueError(method)


def score_on_full(Xn_full_gpu, labels):
    from validate import score_config

    return score_config(Xn_full_gpu, np.asarray(labels))


def part_bake(a, Xn_full_gpu, X_emb_gpu, X_emb_cpu, methods):
    rows, winners = [], {}
    for method in methods:
        best = None
        for p in GRID[method]:
            t0 = time.time()
            lab = cluster_labels(method, X_emb_cpu, X_emb_gpu, p)
            res = score_on_full(Xn_full_gpu, lab)
            secs = round(time.time() - t0, 1)
            rows.append({"method": method, "param": json.dumps(p, sort_keys=True),
                         "secs": secs,
                         **{k: r4(v) for k, v in res.items()
                            if k != "n_scored"}})
            print("bake %s %s adj=%s ncl=%s noise=%s (%.1fs)" % (
                method, p, res.get("adjusted"), res.get("n_clusters"),
                res.get("noise_ratio"), secs), flush=True)
            if res.get("accepted") and (best is None or
                                        res["adjusted"] > best[0]["adjusted"]):
                best = (res, p, np.asarray(lab))
        winners[method] = {"param": best[1],
                           **{k: r4(v) for k, v in best[0].items()
                              if k != "n_scored"}} if best else None
        print("WINNER %s: %s" % (method, winners[method]), flush=True)
        if best is not None:
            np.save(os.path.join(a.out, "labels_pb_%s.npy" % method), best[2])
    write_csv(os.path.join(a.out, "results_phaseb_bake.csv"), rows)
    with open(os.path.join(a.out, "results_phaseb_winners.json"), "w") as f:
        json.dump(winners, f, indent=2, sort_keys=True)
    return winners


def part_stab(a, X_gpu, Xn_full_gpu, winners):
    import cupy as cp

    from design import pairwise_ari

    rows, out = [], {}
    for method, w in winners.items():
        if w is None:
            continue
        labs, sc = [], []
        for s in SEEDS:
            Xr = umap_fit(X_gpu, s, a.cache, "d11")
            Xe = cp.asnumpy(Xr)
            lab = cluster_labels(method, Xe, Xr, w["param"], seed=s)
            res = score_on_full(Xn_full_gpu, lab)
            labs.append(np.asarray(lab))
            sc.append(res.get("adjusted"))
            rows.append({"method": method, "seed": s,
                         **{k: r4(v) for k, v in res.items()
                            if k != "n_scored"}})
            print("stab %s seed=%d adj=%s" % (method, s, res.get("adjusted")),
                  flush=True)
            del Xr
            cp.get_default_memory_pool().free_all_blocks()
        out[method] = {"ari_seeds": round(pairwise_ari(labs), 4),
                       "mean_adj": round(float(np.mean(sc)), 4),
                       "std_adj": round(float(np.std(sc)), 4)}
    write_csv(os.path.join(a.out, "results_phaseb_stab.csv"), rows)
    with open(os.path.join(a.out, "results_phaseb_stab.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)


def part_sub(a, X, X_gpu, Xn_full_gpu, winners):
    import cupy as cp
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(7)
    n = X.shape[0]
    rows = []
    for frac in SUB_FRACS:
        idx = np.sort(rng.choice(n, int(n * frac), replace=False))
        Xr = umap_fit(cp.asarray(X[idx]), 42, a.cache,
                      "sub%02d" % int(100 * frac))
        Xe = cp.asnumpy(Xr)
        Xn_sub = Xn_full_gpu[idx]  # cupy fancy indexing
        for method, w in winners.items():
            if w is None:
                continue
            lab = cluster_labels(method, Xe, Xr, w["param"])
            ref = np.load(os.path.join(a.out, "labels_pb_%s.npy" % method))
            ari = float(adjusted_rand_score(ref[idx], np.asarray(lab)))
            res = score_on_full(Xn_sub, np.asarray(lab))
            rows.append({"method": method, "frac": frac,
                         "ari_vs_full": round(ari, 4),
                         **{k: r4(v) for k, v in res.items()
                            if k != "n_scored"}})
            print("sub %s frac=%.1f ARI=%.4f adj_sub=%s" % (
                method, frac, ari, res.get("adjusted")), flush=True)
        del Xr
        cp.get_default_memory_pool().free_all_blocks()
    write_csv(os.path.join(a.out, "results_phaseb_sub.csv"), rows)


def part_coh(a, base, Xn, winners):
    import pandas as pd

    from coherence import NpmiCorpus, cohesion

    corp = NpmiCorpus(min_df=5).fit(base["summaries"])
    print("coh: vocab=%d docs=%d" % (len(corp.vocab), corp.n_docs), flush=True)
    rows = []
    for method, w in winners.items():
        if w is None:
            continue
        lab = np.load(os.path.join(a.out, "labels_pb_%s.npy" % method))
        for c in sorted(set(lab) - {-1}):
            idx = np.where(lab == c)[0]
            if len(idx) < 10:
                continue
            rows.append({"method": method, "cluster": int(c),
                         "size": len(idx),
                         "npmi": round(corp.coherence(idx), 4),
                         "cohesion": round(cohesion(Xn, idx), 4)})
    write_csv(os.path.join(a.out, "results_phaseb_coh.csv"), rows)
    df = pd.DataFrame(rows)
    summ = {}
    for method in sorted(set(df["method"])):
        sub = df[df["method"] == method].dropna()
        summ[method] = {
            "n_rated": len(sub),
            "npmi_mean": round(float(sub["npmi"].mean()), 4),
            "npmi_wmean": round(float(
                (sub["npmi"] * sub["size"]).sum() / sub["size"].sum()), 4),
            "cohesion_mean": round(float(sub["cohesion"].mean()), 4),
            "pearson_coh_npmi": round(float(np.corrcoef(
                sub["cohesion"], sub["npmi"])[0, 1]), 4)}
    pool = df.dropna()
    summ["pooled_pearson_coh_npmi"] = round(float(np.corrcoef(
        pool["cohesion"], pool["npmi"])[0, 1]), 4)
    print(json.dumps(summ, indent=2, sort_keys=True), flush=True)
    with open(os.path.join(a.out, "results_phaseb_coh_summary.json"),
              "w") as f:
        json.dump(summ, f, indent=2, sort_keys=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for method in sorted(set(df["method"])):
        sub = df[df["method"] == method].dropna()
        ax.scatter(sub["cohesion"], sub["npmi"], s=8, alpha=0.6,
                   label="%s (r=%.2f)" % (
                       method, summ[method]["pearson_coh_npmi"]))
    ax.set_xlabel("geometrische Kohäsion (intra-Kosinus)")
    ax.set_ylabel("Wort-Kohärenz NPMI (unabhaengig)")
    ax.set_title("Externe Validitaet: Geometrie vs. Woerter (r=%s pooled)" %
                 summ["pooled_pearson_coh_npmi"])
    ax.legend(markerscale=2, fontsize="small")
    fig.savefig(os.path.join(a.plots, "phaseb_coh.png"), dpi=120)
    plt.close(fig)


def part_samples(a, base, winners):
    rng = np.random.default_rng(7)
    units, key = [], {}
    uids = ["u%02d" % (i + 1) for i in range(6 * len(
        [w for w in winners.values() if w is not None]))]
    rng.shuffle(uids)
    k = 0
    for method, w in winners.items():
        if w is None:
            continue
        lab = np.load(os.path.join(a.out, "labels_pb_%s.npy" % method))
        clusters = sorted(set(lab) - {-1}, key=lambda c: (lab == c).sum())
        thirds = np.array_split(clusters, 3)  # klein/mittel/gross
        for stratum in thirds:
            pick = rng.choice(list(stratum), min(2, len(stratum)),
                              replace=False)
            for c in pick:
                idx = np.where(lab == c)[0]
                ex = rng.choice(idx, min(8, len(idx)), replace=False)
                uid = uids[k]
                k += 1
                units.append({
                    "unit": uid, "cluster_size": int(len(idx)),
                    "examples": [{
                        "identifier": str(base["ids"][int(i)]),
                        "link": base["links"][int(i)],
                        "excerpt": " ".join(str(
                            base["summaries"][int(i)]).split())[:400]}
                        for i in ex]})
                key[uid] = {"method": method, "cluster": int(c)}
    with open(os.path.join(a.out, "samples_phaseb_units.json"), "w") as f:
        json.dump(units, f, indent=1)
    with open(os.path.join(a.out, "samples_phaseb_key.json"), "w") as f:
        json.dump({"SEALED": "erst nach Rating lesen!", "key": key}, f,
                  indent=1)
    print("samples: %d units (blind), key versiegelt" % len(units), flush=True)


def part_plots(a, winners):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    bake = pd.read_csv(os.path.join(a.out, "results_phaseb_bake.csv"))
    ok = bake[bake["accepted"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = [m for m in GRID if winners.get(m)]
    means = [ok[ok["method"] == m]["adjusted"].max() for m in methods]
    ax.bar(methods, means, color=["tab:blue", "tab:orange", "tab:green",
                                 "tab:red"][:len(methods)])
    for m, v in zip(methods, means):
        ax.text(m, v + 0.001, "%.4f" % v, ha="center", fontsize=9)
    ax.set_ylabel("bestes adjusted (Voll-N)")
    ax.set_title("Phase B: Bake-off der Methoden (fixe Einbettung d=11)")
    fig.savefig(os.path.join(a.plots, "phaseb_bake.png"), dpi=120)
    plt.close(fig)

    stab = json.load(open(os.path.join(a.out, "results_phaseb_stab.json")))
    sub = pd.read_csv(os.path.join(a.out, "results_phaseb_sub.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    w = 0.25
    ax.bar(x - w, [stab[m]["ari_seeds"] for m in methods], w,
           label="ARI Seeds")
    for j, frac in enumerate(SUB_FRACS):
        vals = [sub[(sub["method"] == m) & (sub["frac"] == frac)][
            "ari_vs_full"].iloc[0] for m in methods]
        ax.bar(x + j * w, vals, w, label="ARI sub-%d%%" % int(100 * frac))
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("ARI")
    ax.set_title("Phase B: Seed- und Subsample-Stabilitaet der Winner")
    ax.legend(fontsize="small")
    fig.savefig(os.path.join(a.plots, "phaseb_stab.png"), dpi=120)
    plt.close(fig)
    print("plots ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cache", default=os.path.join(HERE, "umap_cache_doe"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--plots", default=os.path.join(HERE, "plots"))
    ap.add_argument("--part", default="all",
                    choices=["all", "bake", "stab", "sub", "coh", "samples",
                             "plots"])
    ap.add_argument("--methods", default="hdbscan,dbscan,leiden,agglo")
    a = ap.parse_args()

    import cupy as cp

    from data import load_full, truncate_norm

    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.plots, exist_ok=True)
    base = load_full(a.db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu, Xn_gpu = cp.asarray(X), cp.asarray(Xn)
    print("phaseb: N=%d methods=%s" % (X.shape[0], a.methods), flush=True)

    parts = ([a.part] if a.part != "all" else
             ["bake", "stab", "sub", "coh", "samples", "plots"])
    winners = None
    if "bake" in parts:
        Xr = umap_fit(X_gpu, 42, a.cache, "d11")
        Xe = cp.asnumpy(Xr)
        winners = part_bake(a, Xn_gpu, Xr, Xe, a.methods.split(","))
        del Xr
        cp.get_default_memory_pool().free_all_blocks()
    else:
        winners = json.load(open(os.path.join(
            a.out, "results_phaseb_winners.json")))
    if "stab" in parts:
        part_stab(a, X_gpu, Xn_gpu, winners)
    if "sub" in parts:
        part_sub(a, X, X_gpu, Xn_gpu, winners)
    if "coh" in parts:
        part_coh(a, base, Xn, winners)
    if "samples" in parts:
        part_samples(a, base, winners)
    if "plots" in parts:
        part_plots(a, winners)


if __name__ == "__main__":
    main()
