"""face_dataset_validate.py — End-to-End-Validierung an einem gelabelten
Gesichts-Datensatz (LFW) mit DENSELBEN Modellen/Preproc wie die Rust-Pipeline.

Warum: `face_validate.py` prueft nur die Selbstkonsistenz der bereits
gespeicherten DB. Diese Datei lädt einen UNABHAENGIGEN Datensatz mit
Ground-Truth-Identitaeten (LFW people), jagt jedes Bild durch SCRFD (det_500m)
+ ArcFace (w600k_mbf) mit exakt der Rust-Vorverarbeitung (5-Punkt-Alignment auf
das ArcFace-Template, (x-127.5)/127.5 RGB-planar, L2-Norm) und prueft:

  * Same-/Diff-Person-Cosine-Trennung (Cohen's d) — validiert Embeddings.
  * Verifikations-ROC/Accuracy auf LFW-artigen Paaren — externer Sanity-Check.
  * UMAP(2D)+HDBSCAN mit den gefundenen Parametern — validiert die Viz-Kette.

Nutzung:
    python face_dataset_validate.py --models /pfad/models \
        --min-faces 20 --nn 15 --md 0.0 --d 16 --mcs 3
"""

import argparse
import os
import time

import numpy as np

# ArcFace-Template (identisch zu 03_alignment.rs::ARCFACE_DST).
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
CROP = 112


def estimate_similarity(src):
    """Least-Squares-Aehnlichkeitstrafo src->Template (wie estimate_similarity
    in Rust): x'=a*x-b*y+tx, y'=b*x+a*y+ty. Rueckgabe 2x3 Vorwaertsmatrix."""
    ata = np.zeros((4, 4))
    atb = np.zeros(4)
    for s, d in zip(src, ARCFACE_DST):
        x, y = float(s[0]), float(s[1])
        for r, b in (([x, -y, 1.0, 0.0], d[0]), ([y, x, 0.0, 1.0], d[1])):
            r = np.array(r)
            ata += np.outer(r, r)
            atb += r * b
    p = np.linalg.solve(ata, atb)
    a, b, tx, ty = p
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def align(img_rgb, landmarks):
    """Warpt img (H,W,3 uint8 RGB) auf 112x112 anhand 5 Landmarks."""
    import cv2

    m = estimate_similarity(landmarks)
    return cv2.warpAffine(img_rgb, m, (CROP, CROP), flags=cv2.INTER_LINEAR,
                          borderValue=0)


def preprocess_112(crop_rgb):
    """(x-127.5)/127.5 RGB-planar NCHW, wie preprocess_112 in Rust."""
    x = crop_rgb.astype(np.float32)
    x = (x - 127.5) / 127.5
    x = np.transpose(x, (2, 0, 1))  # HWC->CHW
    return x[None, ...].astype(np.float32)


class Scrfd:
    """SCRFD-Detektor (det_500m), 1:1 zum Rust-Decoding in 04_scrfd_detector.rs.

    9 Outputs (scores/bbox/kps fuer strides 8/16/32), NUM_ANCHORS=2,
    Center=(x*stride,y*stride) row-major x2 dupliziert, Regressionen *stride,
    Preproc (x-127.5)/128.0 RGB-planar. Input fest 640x640 (letterbox oben-links,
    det_scale zum Ruecktransform der Landmarks). Nur groesstes Gesicht."""

    INW = 640
    STRIDES = [8, 16, 32]
    NUM_ANCHORS = 2

    def __init__(self, path, conf=0.5):
        import onnxruntime as ort

        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.s.get_inputs()[0].name
        self.conf = conf

    @staticmethod
    def _anchor_centers(stride, gh, gw, num_anchors):
        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        c = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32) * stride
        return np.repeat(c, num_anchors, axis=0)

    def detect(self, img_rgb, min_side=160):
        import cv2

        # Native Aufloesung, ggf. hochskalieren, dann auf Vielfaches von 32
        # padden (das Modell hat dynamische Input-Groesse). Letterboxing in
        # ein 640er-Feld verkleinert das Gesicht zu stark -> keine Detektion.
        h0, w0 = img_rgb.shape[:2]
        up = max(1.0, min_side / min(h0, w0))
        if up > 1.0:
            img_rgb = cv2.resize(img_rgb, (int(round(w0 * up)), int(round(h0 * up))))
        h, w = img_rgb.shape[:2]
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        canvas = np.zeros((pad_h, pad_w, 3), np.uint8)
        canvas[:h, :w] = img_rgb
        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32).copy()
        outs = self.s.run(None, {self.iname: blob})

        cand = []
        for i, stride in enumerate(self.STRIDES):
            scores = np.asarray(outs[i]).reshape(-1)
            bbox = np.asarray(outs[i + 3]).reshape(-1, 4)
            kps = np.asarray(outs[i + 6]).reshape(-1, 10)
            gh, gw = pad_h // stride, pad_w // stride
            centers = self._anchor_centers(stride, gh, gw, self.NUM_ANCHORS)
            n = min(len(scores), len(centers), len(bbox), len(kps))
            for j in range(n):
                if scores[j] < self.conf:
                    continue
                cx, cy = centers[j]
                x1 = cx - bbox[j, 0] * stride
                y1 = cy - bbox[j, 1] * stride
                x2 = cx + bbox[j, 2] * stride
                y2 = cy + bbox[j, 3] * stride
                pts = np.empty((5, 2), np.float32)
                for p in range(5):
                    pts[p, 0] = (cx + kps[j, 2 * p] * stride) / up
                    pts[p, 1] = (cy + kps[j, 2 * p + 1] * stride) / up
                area = (x2 - x1) * (y2 - y1)
                cand.append((float(scores[j]), area, pts))
        if not cand:
            return None
        cand.sort(key=lambda t: t[1], reverse=True)  # groesste Box
        return cand[0][2]


class Arcface:
    def __init__(self, path):
        import onnxruntime as ort

        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.s.get_inputs()[0].name

    def embed(self, crop_rgb):
        out = self.s.run(None, {self.iname: preprocess_112(crop_rgb)})[0].reshape(-1)
        n = np.linalg.norm(out)
        return out / n if n > 1e-9 else out


def build_embeddings(models_dir, min_faces):
    from sklearn.datasets import fetch_lfw_people

    print("lade LFW (min_faces_per_person=%d)..." % min_faces, flush=True)
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True,
                           resize=1.0, funneled=True)
    imgs = (lfw.images * 255).astype(np.uint8)  # (N,H,W,3) RGB
    y = lfw.target
    print("LFW: %d Bilder, %d Personen" % (len(imgs), len(set(y))), flush=True)

    scrfd = Scrfd(os.path.join(models_dir, "det_500m.onnx"))
    arc = Arcface(os.path.join(models_dir, "w600k_mbf.onnx"))

    embs, labs = [], []
    t0 = time.time()
    n_nodet = 0
    for i in range(len(imgs)):
        lm = scrfd.detect(imgs[i])
        if lm is None:
            n_nodet += 1
            continue
        crop = align(imgs[i], lm)
        embs.append(arc.embed(crop))
        labs.append(int(y[i]))
        if (i + 1) % 200 == 0:
            print("  %d/%d (%.1fs, no-det=%d)"
                  % (i + 1, len(imgs), time.time() - t0, n_nodet), flush=True)
    print("fertig: %d Embeddings, %d ohne Detektion (%.1fs)"
          % (len(embs), n_nodet, time.time() - t0))
    return np.array(embs, np.float32), np.array(labs, np.int32)


def sep_stats(X, labels, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    order = np.argsort(labels, kind="stable")
    labs = labels[order]
    same = []
    i = 0
    while i < n:
        j = i
        while j < n and labs[j] == labs[i]:
            j += 1
        idxs = order[i:j]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                same.append(float(X[idxs[a]] @ X[idxs[b]]))
        i = j
    same = np.array(same)
    m = max(len(same) * 4, 20000)
    ia, ib = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = labels[ia] != labels[ib]
    diff = np.einsum("ij,ij->i", X[ia[keep]], X[ib[keep]])
    d = (same.mean() - diff.mean()) / np.sqrt((same.var() + diff.var()) / 2)
    print("\n== LFW Intra/Inter (unabhaengiger Datensatz) ==")
    print("same n=%d mean=%.4f std=%.4f" % (len(same), same.mean(), same.std()))
    print("diff n=%d mean=%.4f std=%.4f" % (len(diff), diff.mean(), diff.std()))
    print("Cohen's d = %.3f" % d)
    # Verifikations-AUC.
    from sklearn.metrics import roc_auc_score

    yv = np.r_[np.ones(len(same)), np.zeros(len(diff))]
    sv = np.r_[same, diff]
    print("Verifikations-ROC-AUC = %.4f" % roc_auc_score(yv, sv))
    return d


def cluster_check(X, labels, nn, md, d, mcs, plotdir):
    import cupy as cp
    from cuml.cluster.hdbscan import HDBSCAN
    from cuml.manifold.umap import UMAP
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    Xg = cp.asarray(X)
    Y2 = UMAP(n_components=2, n_neighbors=nn, min_dist=md, metric="cosine",
              build_algo="nn_descent", random_state=42).fit_transform(Xg)
    Y2 = Y2.get()
    Yd = Y2 if d == 2 else UMAP(n_components=d, n_neighbors=nn, min_dist=md,
                                metric="cosine", build_algo="nn_descent",
                                random_state=42).fit_transform(Xg).get()
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=mcs,
                  metric="euclidean").fit_predict(Yd)
    lab = lab.get() if hasattr(lab, "get") else np.asarray(lab)
    valid = lab != -1
    ari = adjusted_rand_score(labels[valid], lab[valid])
    ami = adjusted_mutual_info_score(labels[valid], lab[valid])
    print("\n== LFW UMAP(%dD)+HDBSCAN (nn=%d md=%.2f mcs=%d) ==" % (d, nn, md, mcs))
    print("Cluster=%d  Rauschen=%.1f%%  ARI=%.3f  AMI=%.3f  (Personen=%d)"
          % (len(set(lab[valid])), 100 * (1 - valid.mean()), ari, ami,
             len(set(labels))))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(plotdir, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
        ax[0].scatter(Y2[:, 0], Y2[:, 1], s=6, c=labels, cmap="tab20", alpha=0.6)
        ax[0].set_title("LFW 2D UMAP, gefaerbt nach ECHTER Person")
        c = lab.astype(float)
        c[c < 0] = np.nan
        ax[1].scatter(Y2[:, 0], Y2[:, 1], s=6, c=c, cmap="tab20", alpha=0.6)
        ax[1].set_title("gefaerbt nach HDBSCAN-Cluster (ARI=%.3f)" % ari)
        p = os.path.join(plotdir, "lfw_validate.png")
        fig.savefig(p, bbox_inches="tight")
        print("Plot: %s" % p)
    except Exception as e:  # noqa: BLE001
        print("Plot uebersprungen: %s" % e)
    return ari, ami


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--min-faces", type=int, default=20)
    ap.add_argument("--nn", type=int, default=15)
    ap.add_argument("--md", type=float, default=0.0)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--mcs", type=int, default=3)
    ap.add_argument("--plotdir", default="plots_face")
    ap.add_argument("--cache", default="lfw_emb.npz")
    a = ap.parse_args()

    if os.path.exists(a.cache):
        print("nutze Cache %s" % a.cache)
        z = np.load(a.cache)
        X, labels = z["X"], z["labels"]
    else:
        X, labels = build_embeddings(a.models, a.min_faces)
        np.savez(a.cache, X=X, labels=labels)
    print("Embeddings: %s, Personen=%d" % (X.shape, len(set(labels))))
    sep_stats(X, labels)
    cluster_check(X, labels, a.nn, a.md, a.d, a.mcs, a.plotdir)


if __name__ == "__main__":
    main()
