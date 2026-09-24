"""Build cluster_data.json: canonical HDBSCAN labels + 2D coords + samples + neighbors."""

import json
import os

import numpy as np

CACHE = "umap_cache_k3072"
KEY12 = "k3072_d12_nn030_md0p1.npy"
KEY2 = "k3072_d02_nn030_md0p1.npy"
N_SAMPLES = 8
N_NEIGHBORS = 3


def main():
    import cupy as cp

    from loader import load_embeddings
    from cluster_search import cluster_hdbscan

    r = load_embeddings("summaries_compact_20260924.db", 3072)
    X12 = np.load(os.path.join(CACHE, KEY12))
    p2 = os.path.join(CACHE, KEY2)
    if os.path.exists(p2):
        X2 = np.load(p2)
    else:
        from sweep_umap import run_umap
        got, _ = run_umap(cp.asarray(r["X"]), 2, 30, 0.1)
        X2 = cp.asnumpy(got)
        np.save(p2, X2)

    labels = np.asarray(cluster_hdbscan(cp.asarray(X12)))
    ids = np.array(r["ids"])
    np.savetxt("plots/labels_canonical.csv",
               np.column_stack([ids, labels]), fmt="%d",
               header="identifier,cluster", comments="")

    clusters = sorted(set(labels) - {-1})
    cents = {c: X12[labels == c].mean(axis=0) for c in clusters}
    out = {"params": {"db": "summaries_compact_20260924.db", "width": 3072,
                      "d": 12, "nn": 30, "md": 0.1, "method": "hdbscan"},
           "clusters": []}
    for c in clusters:
        m = np.where(labels == c)[0]
        dists = sorted(((float(np.linalg.norm(cents[c] - cents[o])), int(o))
                        for o in clusters if o != c))[:N_NEIGHBORS]
        out["clusters"].append({
            "id": int(c), "size": int(m.size),
            "neighbors": [o for _, o in dists],
            "samples": [{"identifier": int(ids[i]),
                         "link": r["links"][i],
                         "summary": (r["summaries"][i] or "")[:600]}
                        for i in m[:N_SAMPLES]],
        })
    out["noise_size"] = int((labels == -1).sum())
    json.dump(out, open("cluster_data.json", "w"), ensure_ascii=False)
    print("clusters=%d noise=%d wrote cluster_data.json" % (
        len(clusters), out["noise_size"]))


if __name__ == "__main__":
    main()
