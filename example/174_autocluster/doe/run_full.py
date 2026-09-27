"""Full-Sweep-Treiber: Sobol-128 x 5 Seeds auf erweiterter Box (GPU).

Vollstaendiger Space-Filling-Sweep ueber FULL_BOUNDS
(d[2,24] x nn[10,80] x md[0,0.4] x mcs[5,60] x ms[3,30]),
k=3072-Schnittmenge, R=5 Seed-Replikate. Eigener Cache (umap_cache_full),
inkrementelle CSV + Resume: bereits fertige (run_id, seed)-Paare werden
per gecachter Einbettung + Re-Clustering (HDBSCAN deterministisch)
label-identisch rekonstruiert statt neu gefittet.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "summaries_compact_20260924.db")
DEFAULT_CACHE = os.path.join(HERE, "umap_cache_full")
DEFAULT_OUT = os.path.join(HERE, "results")
FULL_SEEDS = [42, 1337, 2026, 7, 99]


def r4(v):
    return round(float(v), 4) if isinstance(v, float) else v


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["run_id"]), int(row["seed"])))
    return done


def recover_from_cache(Xn_gpu, k, cfg, seed, cache):
    """Labels aus gecachter Einbettung rekonstruieren (Resume, ohne UMAP).

    HDBSCAN ist bei fester Einbettung deterministisch, daher sind die
    Labels identisch zum Original-Run. Gibt None zurueck, wenn kein
    Cache-Eintrag existiert (dann Voll-Run noetig).
    """
    import cupy as cp

    from experiment import cache_key, run_hdbscan
    from validate import score_config

    path = os.path.join(cache, cache_key(k, cfg, seed) + ".npy")
    if not os.path.exists(path):
        return None
    Xr = cp.asarray(np.load(path))
    labels = run_hdbscan(Xr, cfg)
    res = score_config(Xn_gpu, np.asarray(labels))
    del Xr
    cp.get_default_memory_pool().free_all_blocks()
    return res, np.asarray(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--seeds", default=",".join(str(s) for s in FULL_SEEDS))
    ap.add_argument("--design-seed", type=int, default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    import cupy as cp

    from data import load_full, truncate_norm
    from design import (FULL_DESIGN_SEED, aggregate_point, generate_sobol_design,
                        save_design)
    from experiment import run_one

    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    base = load_full(a.db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu, Xn_gpu = cp.asarray(X), cp.asarray(Xn)

    design = generate_sobol_design(a.n_points,
                                   seed=FULL_DESIGN_SEED
                                   if a.design_seed is None else a.design_seed)
    save_design(design, os.path.join(a.out, "design_full.csv"))

    raw_path = os.path.join(a.out, "results_full.csv")
    done = load_done(raw_path)
    print("full: N=%d points=%d seeds=%s resume=%d/%d" % (
        X.shape[0], len(design), seeds, len(done),
        len(design) * len(seeds)), flush=True)

    fieldnames = (["run_id", "seed", "secs", "recovered"] +
                  list(design[0].keys()) +
                  ["accepted", "n_clusters", "noise_ratio", "silhouette",
                   "adjusted"])
    fresh = not os.path.exists(raw_path)
    fout = open(raw_path, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if fresh:
        w.writeheader()

    agg = []
    t_all = time.time()
    for i, cfg in enumerate(design):
        per_seed = []
        for s in seeds:
            rec = None
            if (i, s) in done:
                rec = recover_from_cache(Xn_gpu, 3072, cfg, s, a.cache)
            if rec is not None:
                res, labels = rec
                secs, recovered = 0.0, 1
            else:
                t0 = time.time()
                res, labels = run_one(X_gpu, Xn_gpu, 3072, cfg, s, a.cache)
                secs, recovered = round(time.time() - t0, 1), 0
                w.writerow({"run_id": i, "seed": s, "secs": secs,
                            "recovered": recovered, **cfg,
                            **{k: r4(v) for k, v in res.items()
                               if k != "n_scored"}})
            per_seed.append((cfg, res, labels))
        fout.flush()
        ag = aggregate_point(per_seed)
        if ag is not None:
            agg.append({"run_id": i,
                        **{k: r4(v) for k, v in ag.items()}})
        acc = sum(1 for _, r, _ in per_seed if r.get("accepted"))
        print("point %d/%d accepted=%d/%d %s (%.0fs elapsed)" % (
            i + 1, len(design), acc, len(seeds),
            ("mean=%.4f std=%.4f sn=%.2f ari=%.3f" % (
                ag["mean_adj"], ag["std_adj"], ag["sn_ratio"],
                ag["ari_stability"]) if ag else "REJECTED"),
            time.time() - t_all), flush=True)
    fout.close()

    agg_path = os.path.join(a.out, "results_full_agg.csv")
    if agg:
        with open(agg_path, "w", newline="") as f:
            ww = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            ww.writeheader()
            ww.writerows(agg)
    print("wrote %s (%d rows), agg %d/%d (%.0fs total)" % (
        raw_path, len(design) * len(seeds), len(agg), len(design),
        time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
