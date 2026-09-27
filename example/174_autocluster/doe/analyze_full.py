"""Full-Sweep-Auswertung (CPU): RSM/ANOVA, Optima, Alt-vs-neu, Plots."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from design import FULL_RESPONSE_FORMULA, fit_response_surface

    res = os.path.join(HERE, "results")
    agg = pd.read_csv(os.path.join(res, "results_full_agg.csv"))
    raw = pd.read_csv(os.path.join(res, "results_full.csv"))
    print("=== Full-Sweep: %d aggregierte Punkte (%d Runs, %d akzeptiert) ==="
          % (len(agg), len(raw), int(raw["accepted"].sum())), flush=True)

    model, anova = fit_response_surface(agg, formula=FULL_RESPONSE_FORMULA)
    print("R²=%.3f adj-R²=%.3f n=%d" % (
        model.rsquared, model.rsquared_adj, len(agg)), flush=True)
    print("--- ANOVA (Typ II, nach F sortiert) ---", flush=True)
    print(anova.sort_values("F", ascending=False).to_string(), flush=True)

    top_sn = agg.sort_values("sn_ratio", ascending=False).head(5)
    top_mean = agg.sort_values("mean_adj", ascending=False).head(5)
    print("--- Top-5 robust (S/N) ---", flush=True)
    print(top_sn.to_string(index=False), flush=True)
    print("--- Top-5 Peak (mean) ---", flush=True)
    print(top_mean.to_string(index=False), flush=True)
    robust = agg.loc[agg["sn_ratio"].idxmax()].to_dict()
    peak = agg.loc[agg["mean_adj"].idxmax()].to_dict()

    old = json.load(open(os.path.join(res, "best_doe.json")))["robust_optimum"]
    print("ALT robust: mean=%.4f (d=%g nn=%g md=%g mcs=%g ms=%g)" % (
        old["mean_adj"], old["d"], old["n_neighbors"], old["min_dist"],
        old["min_cluster_size"], old["min_samples"]), flush=True)
    print("NEU robust: mean=%.4f (+%+.4f) ari=%.3f ncl=%.0f" % (
        robust["mean_adj"], robust["mean_adj"] - old["mean_adj"],
        robust["ari_stability"], robust["mean_clusters"]), flush=True)
    print("NEU peak:   mean=%.4f (+%+.4f) ari=%.3f ncl=%.0f" % (
        peak["mean_adj"], peak["mean_adj"] - old["mean_adj"],
        peak["ari_stability"], peak["mean_clusters"]), flush=True)

    with open(os.path.join(res, "best_full.json"), "w") as f:
        json.dump({"robust_optimum": robust, "peak_optimum": peak,
                   "old_robust_mean": old["mean_adj"],
                   "rsquared": float(model.rsquared),
                   "rsquared_adj": float(model.rsquared_adj)},
                  f, indent=2, sort_keys=True, default=float)

    plots = os.path.join(HERE, "plots")
    os.makedirs(plots, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(agg["mean_adj"], agg["sn_ratio"], s=12, alpha=0.7)
    ax.scatter([robust["mean_adj"]], [robust["sn_ratio"]], s=80,
               marker="*", color="red", label="robust")
    ax.scatter([peak["mean_adj"]], [peak["sn_ratio"]], s=80,
               marker="^", color="green", label="peak")
    ax.axvline(old["mean_adj"], color="gray", linestyle="--",
               label="alt 0.1343")
    ax.set_xlabel("mean_adj")
    ax.set_ylabel("Taguchi-S/N")
    ax.set_title("Full-Sweep: robust vs. peak vs. alt")
    ax.legend()
    fig.savefig(os.path.join(plots, "full_sn_vs_mean.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    sc = ax.scatter(agg["min_cluster_size"], agg["mean_adj"],
                    c=agg["d"], s=14, alpha=0.8, cmap="viridis")
    fig.colorbar(sc, label="d")
    ax.axhline(old["mean_adj"], color="gray", linestyle="--")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("mean_adj")
    ax.set_title("Full-Sweep: mcs-Effekt (Farbe = d)")
    fig.savefig(os.path.join(plots, "full_mcs_vs_mean.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(agg["d"] + np.random.default_rng(1).uniform(
        -0.25, 0.25, len(agg)), agg["mean_adj"], s=14, alpha=0.7)
    ax.axhline(old["mean_adj"], color="gray", linestyle="--")
    ax.set_xlabel("UMAP d")
    ax.set_ylabel("mean_adj")
    ax.set_title("Full-Sweep: d-Effekt (erweitert 2-24)")
    fig.savefig(os.path.join(plots, "full_d_vs_mean.png"), dpi=120)
    plt.close(fig)
    print("plots ok", flush=True)


if __name__ == "__main__":
    main()
