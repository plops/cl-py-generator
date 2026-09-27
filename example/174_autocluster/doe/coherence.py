"""NPMI-Themenkohaerenz: externe Validierung jenseits der Vektorgeometrie.

Wort-Kookkurrenzen in den (deutschen) Summaries statt Embedding-Abstaende:
pro Cluster Top-10-Woerter, mittlere paarweise NPMI = Kohaerenz.
Handgerollt ohne gensim: deterministisch, getestet, CPU-only.
Konvention: nie kookkurrierende Paare zaehlen -1 (ehrliche Strafe).
"""

import re

import numpy as np

TOKEN_RE = re.compile(r"[a-zäöüß]{3,}")

STOPWORDS_DE = frozenset(
    "der die das und ist ein eine einer eines einem einen dem den des "
    "zu von mit für auf sich nicht auch als wie oder aber wenn dann "
    "man es er sie ihr ihm ihn uns euer eure eurem euren ihrer ihrem "
    "ihren im am beim vom zum zur über unter durch gegen ohne um "
    "wird werden wurde wurden hat haben hatte hatten kann können muss "
    "müssen soll sollen will wollen darf dürfen mag mögen dies diese "
    "dieser dieses diesen diesem jahr jahren video videos youtube kanal "
    "folge teil the and for with from that this have has are was were "
    "euch dich sich mich mein meine meiner meinen meinem meines dein "
    "deine deiner deinen deinem deines sein seine seiner seinen seinem "
    "seines unser unsere euer eure kein keine keiner keinen keinem keinen "
    "alle allem allen aller alles andere anderen anderem anderen sehr "
    "mehr wieder schon noch nur auch bereits etwa sowie zwischen nach "
    "vor bei seit während weil dass damit wohingegen jedoch also denn "
    "doch nun hier dort wo wohin daher darum deshalb trotzdem obwohl "
    "sowie sowohl entweder weder sondern vielmehr prozent"
    .split())


def tokenize(text):
    """Kleinbuchstaben, ≥3 Buchstaben (inkl. Umlaute/ß), ohne Stopwoerter."""
    return [t for t in TOKEN_RE.findall((text or "").lower())
            if t not in STOPWORDS_DE]


class NpmiCorpus:
    """Binaere Dok-Term-Matrix + NPMI-Kohaerenz (deterministisch)."""

    def __init__(self, min_df=5):
        self.min_df = min_df

    def fit(self, texts):
        from scipy.sparse import csr_matrix

        docs = [tokenize(t) for t in texts]
        df = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.vocab = sorted(w for w, c in df.items() if c >= self.min_df)
        self.w2i = {w: j for j, w in enumerate(self.vocab)}
        rows, cols = [], []
        for i, d in enumerate(docs):
            for w in set(d) & set(self.vocab):
                rows.append(i)
                cols.append(self.w2i[w])
        self.bow = csr_matrix(
            (np.ones(len(rows), np.int32), (rows, cols)),
            shape=(len(docs), len(self.vocab)))
        self.n_docs = len(docs)
        self.df = np.asarray(self.bow.sum(axis=0)).ravel().astype(float)
        return self

    def top_words(self, doc_ids, topn=10):
        """Top-Woerter nach In-Cluster-Haeufigkeit (Count desc, Wort asc)."""
        from collections import Counter

        c = Counter()
        for i in doc_ids:
            row = self.bow.getrow(int(i))
            c.update({self.vocab[j]: 1 for j in row.indices})
        ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ranked[:topn]]

    def coherence(self, doc_ids, topn=10):
        """Mittlere paarweise NPMI der Top-Woerter; nan bei <2 Woertern."""
        words = self.top_words(doc_ids, topn)
        if len(words) < 2:
            return float("nan")
        idx = [self.w2i[w] for w in words]
        co = (self.bow[:, idx].T @ self.bow[:, idx]).toarray().astype(float)
        p1 = self.df[idx] / self.n_docs
        vals = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                p12 = co[a, b] / self.n_docs
                if p12 <= 0:
                    vals.append(-1.0)
                else:
                    vals.append(float(np.log(p12 / (p1[a] * p1[b]))
                                        / (-np.log(p12))))
        return float(np.mean(vals))


def cohesion(Xn, idx, max_n=300, seed=0):
    """Mittlere paarweise Kosinus-Aehnlichkeit im Cluster (≤max_n Sample)."""
    idx = np.asarray(sorted(set(int(i) for i in idx)))
    if len(idx) < 2:
        return float("nan")
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(
            idx, max_n, replace=False))
    s = np.asarray(Xn)[idx] @ np.asarray(Xn)[idx].T
    iu = np.triu_indices(len(idx), 1)
    return float(s[iu].mean())
