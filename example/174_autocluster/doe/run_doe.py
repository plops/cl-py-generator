"""DoE-Treiber: Jitter → Screening → Breiten-Ablation → DBSCAN-Kalibrierung.

Phasen (review.md Teil 3):
  jitter    Best-Config × 10 Seeds: wie groß ist das nn_descent-Rauschen?
  screening LHS-Plan (default 60 Punkte) × 3 Seeds auf k=3072-Schnittmenge
  width     12 geteilte LHS-Punkte × {128,768,3072} × 3 Seeds (fair, gleiche Rows)
  dbscan    faire eps-Kalibrierung pro d (statt linearer Heuristik)
  all       alles nacheinander

Alle Phasen nutzen die GPU (cuML/cuPy) und schreiben CSVs nach doe/results.
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
DEFAULT_CACHE = os.path.join(HERE, "umap_cache_doe")
DEFAULT_OUT = os.path.join(HERE, "results")

BEST_CFG = {"d": 12, "n_neighbors": 30, "min_dist": 0.1,
            "min_cluster_size": 15, "min_samples": 15}
JITTER_SEEDS = [42, 1337, 2026, 7, 99, 11, 123, 777, 9001, 31415]
EPS_GRID = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


def r4(v):
    return round(float(v), 4) if isinstance(v, float) else v


def phase_jitter(db, cache, out, seeds):
    """Best-Config über viele Seeds: Jitter-Quantifizierung + ARI."""
    import cupy as cp

    from data import load_full, truncate_norm
    from design import pairwise_ari
    from experiment import run_one

    base = load_full(db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu, Xn_gpu = cp.asarray(X), cp.asarray(Xn)
    print("jitter: N=%d k=3072 cfg=%s" % (X.shape[0], BEST_CFG), flush=True)
    rows, labels_all, scores = [], [], []
    for s in seeds:
        t0 = time.time()
        res, labels = run_one(X_gpu, Xn_gpu, 3072, BEST_CFG, s, cache)
        rows.append({"seed": s, "secs": round(time.time() - t0, 1),
                     **{k: r4(v) for k, v in res.items() if k != "n_scored"}})
        print("seed=%d adj=%s clusters=%s noise=%s" % (
            s, res.get("adjusted"), res.get("n_clusters"),
            res.get("noise_ratio")), flush=True)
        if res.get("accepted"):
            scores.append(res["adjusted"])
            labels_all.append(np.asarray(labels))
    write_csv(os.path.join(out, "results_doe_jitter.csv"), rows)
    if len(scores) >= 2:
        print("jitter: mean=%.4f std=%.4f ARI=%.4f (n=%d)" % (
            np.mean(scores), np.std(scores),
            pairwise_ari(labels_all), len(scores)), flush=True)


def phase_screening(db, cache, out, n_points, seeds, design_seed=42):
    """LHS-Screening auf der k=3072-Schnittmenge, R=len(seeds) Replikate."""
    import cupy as cp

    from data import load_full, truncate_norm
    from design import aggregate_point, generate_lhs_design, save_design
    from experiment import run_one

    base = load_full(db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu, Xn_gpu = cp.asarray(X), cp.asarray(Xn)
    design = generate_lhs_design(n_points, design_seed)
    save_design(design, os.path.join(out, "design_screening.csv"))
    print("screening: N=%d points=%d seeds=%s" % (
        X.shape[0], n_points, seeds), flush=True)
    rows, agg = [], []
    for i, cfg in enumerate(design):
        per_seed = []
        for s in seeds:
            t0 = time.time()
            res, labels = run_one(X_gpu, Xn_gpu, 3072, cfg, s, cache)
            per_seed.append((cfg, res, labels))
            rows.append({"run_id": i, "seed": s,
                         "secs": round(time.time() - t0, 1), **cfg,
                         **{k: r4(v) for k, v in res.items()
                            if k != "n_scored"}})
        a = aggregate_point(per_seed)
        if a is not None:
            agg.append({"run_id": i, **{k: r4(v) for k, v in a.items()}})
        acc = sum(1 for _, r, _ in per_seed if r.get("accepted"))
        print("point %d/%d accepted=%d %s" % (i + 1, n_points, acc,
              ("mean=%.4f std=%.4f sn=%.2f ari=%.3f" % (
                  a["mean_adj"], a["std_adj"], a["sn_ratio"],
                  a["ari_stability"]) if a else "REJECTED")),
              flush=True)
    write_csv(os.path.join(out, "results_doe_screening.csv"), rows)
    if agg:
        write_csv(os.path.join(out, "results_doe_screening_agg.csv"), agg)


def phase_width(db, cache, out, n_points, seeds, widths=(128, 768, 3072)):
    """Faire Breiten-Ablation: gleiche LHS-Punkte + gleiche Rows für alle k."""
    import cupy as cp

    from data import load_aligned
    from design import aggregate_point, generate_lhs_design, save_design
    from experiment import run_one

    base, per_width = load_aligned(db, widths)
    gpu = {k: (cp.asarray(v["X"]), cp.asarray(v["X_norm"]))
           for k, v in per_width.items()}
    design = generate_lhs_design(n_points, seed=7)
    save_design(design, os.path.join(out, "design_width.csv"))
    print("width: N=%d points=%d widths=%s seeds=%s" % (
        base["X"].shape[0], n_points, widths, seeds), flush=True)
    rows, agg = [], []
    for i, cfg in enumerate(design):
        for k in widths:
            X_gpu, Xn_gpu = gpu[k]
            per_seed = []
            for s in seeds:
                res, labels = run_one(X_gpu, Xn_gpu, k, cfg, s, cache)
                per_seed.append((cfg, res, labels))
                rows.append({"run_id": i, "k": k, "seed": s, **cfg,
                             **{kk: r4(v) for kk, v in res.items()
                                if kk != "n_scored"}})
            a = aggregate_point(per_seed)
            if a is not None:
                agg.append({"run_id": i, "k": k,
                            **{kk: r4(v) for kk, v in a.items()
                               if kk not in ("d", "n_neighbors", "min_dist",
                                             "min_cluster_size",
                                             "min_samples")},
                            **cfg})
            print("point %d k=%d %s" % (i, k, "REJECTED" if a is None else
                  "mean=%.4f std=%.4f" % (a["mean_adj"], a["std_adj"])),
                  flush=True)
    write_csv(os.path.join(out, "results_doe_width.csv"), rows)
    if agg:
        write_csv(os.path.join(out, "results_doe_width_agg.csv"), agg)


def phase_dbscan(db, cache, out, dims=(4, 8, 12, 16), seed=42):
    """Faire DBSCAN-eps-Kalibrierung: eps-Grid pro d statt Heuristik.

    UMAP-Einbettung fix (nn=30, md=0.1, k=3072), dann eps-Grid + Scoring
    im Originalraum — derselbe Maßstab wie für HDBSCAN.
    """
    import cupy as cp

    from data import load_full, truncate_norm
    from experiment import run_dbscan_eps, run_umap_seed
    from validate import score_config

    base = load_full(db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu, Xn_gpu = cp.asarray(X), cp.asarray(Xn)
    rows = []
    for d in dims:
        cfg = {"d": d, "n_neighbors": 30, "min_dist": 0.1}
        path = os.path.join(cache, "dbscan_k3072_d%02d_s%d.npy" % (d, seed))
        if os.path.exists(path):
            Xr = cp.asarray(np.load(path))
        else:
            Xr = run_umap_seed(X_gpu, cfg, seed)
            cp.save(path, Xr)
        for eps in EPS_GRID:
            labels = run_dbscan_eps(Xr, eps)
            res = score_config(Xn_gpu, labels)
            rows.append({"d": d, "eps": eps,
                         **{k: r4(v) for k, v in res.items()
                            if k != "n_scored"}})
            print("d=%d eps=%.2f adj=%s clusters=%s noise=%s" % (
                d, eps, res.get("adjusted"), res.get("n_clusters"),
                res.get("noise_ratio")), flush=True)
        del Xr
        cp.get_default_memory_pool().free_all_blocks()
    write_csv(os.path.join(out, "results_doe_dbscan.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["jitter", "screening", "width", "dbscan", "all"])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-screening", type=int, default=60)
    ap.add_argument("--n-width", type=int, default=12)
    ap.add_argument("--seeds", default="42,1337,2026")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.phase in ("jitter", "all"):
        phase_jitter(a.db, a.cache, a.out, JITTER_SEEDS)
    if a.phase in ("screening", "all"):
        phase_screening(a.db, a.cache, a.out, a.n_screening, seeds)
    if a.phase in ("width", "all"):
        phase_width(a.db, a.cache, a.out, a.n_width, seeds)
    if a.phase in ("dbscan", "all"):
        phase_dbscan(a.db, a.cache, a.out)


if __name__ == "__main__":
    main()
