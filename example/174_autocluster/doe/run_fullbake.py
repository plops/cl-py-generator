"""Full-Bake-off: Methodenvergleich auf 3 Einbettungen (d=8/11/16, GPU).

Phase B liess HDBSCAN/DBSCAN/Leiden/Agglo auf EINER Einbettung (d=11)
antreten — der Leiden-Kollaps auf anderen Seeds zeigte aber, dass das
Ranking embedding-abhaengig sein kann. Dieser Treiber wiederholt den
Bake-off (gleiches GRID wie Phase B) auf drei Einbettungen mit
fixem nn=45/md=0.09 (Winner-Geometrie, nur d variiert) und prueft die
Winner-Stabilitaet auf je 2 weiteren Seeds.

Frischer Cache-Tag (fullbake_*): die d=11-Schiene ist damit eine echte
Replikation des Phase-B-Bakes (UMAP wird neu gerechnet, nicht geladen).
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "summaries_compact_20260924.db")
DEFAULT_CACHE = os.path.join(HERE, "umap_cache_fullbake")
DEFAULT_OUT = os.path.join(HERE, "results")
EMBS = [(8, 45, 0.09), (11, 45, 0.09), (16, 45, 0.09)]
STAB_SEEDS = [1337, 2026]


def r4(v):
    return round(float(v), 4) if isinstance(v, float) else v


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--plots", default=os.path.join(HERE, "plots"))
    ap.add_argument("--methods", default="hdbscan,dbscan,leiden,agglo")
    a = ap.parse_args()

    import cupy as cp

    import run_phaseb as pb
    from data import load_full, truncate_norm
    from design import pairwise_ari

    os.makedirs(a.cache, exist_ok=True)
    base = load_full(a.db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu, Xn_gpu = cp.asarray(X), cp.asarray(Xn)
    methods = a.methods.split(",")
    print("fullbake: N=%d embs=%s methods=%s" % (
        X.shape[0], [e[0] for e in EMBS], methods), flush=True)

    bake_rows, stab_rows, winners = [], [], {}
    for d, nn, md in EMBS:
        pb.EMB = {"d": d, "n_neighbors": nn, "min_dist": md}
        tag = "fullbake_d%02d" % d
        Xr = pb.umap_fit(X_gpu, 42, a.cache, tag)
        Xe = cp.asnumpy(Xr)
        key = "d%02d" % d
        winners[key] = {}
        for method in methods:
            best = None
            for p in pb.GRID[method]:
                lab = pb.cluster_labels(method, Xe, Xr, p)
                res = pb.score_on_full(Xn_gpu, lab)
                bake_rows.append({"emb": key, "method": method,
                                  "param": json.dumps(p, sort_keys=True),
                                  **{k: r4(v) for k, v in res.items()
                                     if k != "n_scored"}})
                if res.get("accepted") and (best is None or
                                            res["adjusted"] > best[0]["adjusted"]):
                    best = (res, p, np.asarray(lab))
            winners[key][method] = (
                {"param": best[1],
                 **{k: r4(v) for k, v in best[0].items()
                    if k != "n_scored"}} if best else None)
            print("WINNER %s %s: %s" % (key, method, winners[key][method]),
                  flush=True)
            if best is not None:
                np.save(os.path.join(
                    a.out, "labels_fb_%s_%s.npy" % (key, method)), best[2])
        del Xr
        cp.get_default_memory_pool().free_all_blocks()

        # Winner-Stabilitaet: gleiche Geometrie, 2 weitere Seeds
        for method in methods:
            w = winners[key][method]
            if w is None:
                continue
            labs, sc = [np.load(os.path.join(
                a.out, "labels_fb_%s_%s.npy" % (key, method)))], [w["adjusted"]]
            for s in STAB_SEEDS:
                Xr = pb.umap_fit(X_gpu, s, a.cache, tag)
                Xe = cp.asnumpy(Xr)
                lab = pb.cluster_labels(method, Xe, Xr, w["param"], seed=s)
                res = pb.score_on_full(Xn_gpu, lab)
                labs.append(np.asarray(lab))
                sc.append(res.get("adjusted"))
                stab_rows.append({"emb": key, "method": method, "seed": s,
                                  **{k: r4(v) for k, v in res.items()
                                     if k != "n_scored"}})
                print("stab %s %s seed=%d adj=%s" % (
                    key, method, s, res.get("adjusted")), flush=True)
                del Xr
                cp.get_default_memory_pool().free_all_blocks()
            w["ari_seeds"] = round(pairwise_ari(labs), 4)
            w["stab_min"] = round(float(np.nanmin(sc)), 4)
            w["stab_max"] = round(float(np.nanmax(sc)), 4)

    write_csv(os.path.join(a.out, "results_fullbake_bake.csv"), bake_rows)
    write_csv(os.path.join(a.out, "results_fullbake_stab.csv"), stab_rows)
    with open(os.path.join(a.out, "results_fullbake_winners.json"), "w") as f:
        json.dump(winners, f, indent=2, sort_keys=True)

    # Plot: Winner-Balken je Einbettung (+ Stab-Spanne)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(methods))
    width = 0.25
    for j, (d, _, _) in enumerate(EMBS):
        key = "d%02d" % d
        vals = [(winners[key][m] or {}).get("adjusted", float("nan"))
                for m in methods]
        lo = [(winners[key][m] or {}).get("stab_min", float("nan"))
              for m in methods]
        hi = [(winners[key][m] or {}).get("stab_max", float("nan"))
              for m in methods]
        ax.bar(xs + (j - 1) * width, vals, width, label="d=%d" % d)
        for x, v, l, h in zip(xs + (j - 1) * width, vals, lo, hi):
            if v == v and l == l:
                ax.plot([x, x], [l, h], color="black", linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("bestes adjusted (+ Seed-Spanne)")
    ax.set_title("Full-Bake-off: Methodenranking je Einbettung (nn=45, md=0.09)")
    ax.legend()
    fig.savefig(os.path.join(a.plots, "fullbake_bake.png"), dpi=120)
    plt.close(fig)
    print("plots ok", flush=True)


if __name__ == "__main__":
    main()
