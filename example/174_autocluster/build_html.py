"""Merge subagent titles + build zoomable HTML cluster map (plotly, offline)."""

import glob
import json
import os
import re

import numpy as np


def extract_titles():
    merged = {}
    base = "/root/.local/share/muse/sessions/2026/09/24/01a0d2c0-a6be-7160-b1ee-6ed6ffd7c3c8/subagent"
    for path in glob.glob(os.path.join(base, "*/session.jsonl")):
        try:
            text = open(path, encoding="utf-8").read().replace('\\"', '"')
        except OSError:
            continue
        best = {}
        for m in re.finditer(r'\{"\d+"\s*:\s*\{"title"', text):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
            except ValueError:
                continue
            if isinstance(obj, dict) and obj and all(
                    isinstance(v, dict) and "title" in v
                    for v in obj.values()) and len(obj) > len(best):
                best = obj
        merged.update(best)
    json.dump(merged, open("cluster_titles.json", "w"), ensure_ascii=False)
    print("titles: %d clusters" % len(merged))
    return merged


def build_html(titles):
    import plotly.graph_objects as go

    data = json.load(open("cluster_data.json"))
    X2 = np.load("umap_cache_k3072/k3072_d02_nn030_md0p1.npy")
    lab = np.loadtxt("plots/labels_canonical.csv", skiprows=1, dtype=int)
    ids, labels = lab[:, 0], lab[:, 1]

    from loader import load_embeddings
    r = load_embeddings("summaries_compact_20260924.db", 3072)
    summ = {i: ((r["summaries"][k] or "")[:220], r["links"][k])
            for k, i in enumerate(r["ids"])}

    fig = go.Figure()
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        t = titles.get(str(int(c)), {})
        name = "%d %s (%d)" % (c, t.get("title", "?"), m.sum())
        hover = ["<b>%s</b><br>%s<br><a href='%s'>video</a><br>%s" % (
            t.get("title", c), ident, summ.get(ident, ("", ""))[1],
            summ.get(ident, ("", ""))[0]) for ident in ids[m]]
        fig.add_trace(go.Scattergl(x=X2[m, 0], y=X2[m, 1], mode="markers",
                                   name=name, text=hover, hoverinfo="text",
                                   marker=dict(size=3, opacity=0.7)))
    n = labels == -1
    fig.add_trace(go.Scattergl(x=X2[n, 0], y=X2[n, 1], mode="markers",
                               name="Noise (%d)" % n.sum(),
                               hoverinfo="skip",
                               marker=dict(size=2, color="lightgrey")))
    fig.update_layout(title="Video-Summary-Cluster (k=3072, UMAP d12, HDBSCAN)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(itemsizing="constant"),
                      dragmode="zoom", hovermode="closest")
    fig.write_html("plots/clusters.html", include_plotlyjs=True)
    print("wrote plots/clusters.html (%d traces)" % len(fig.data))


if __name__ == "__main__":
    build_html(extract_titles())
