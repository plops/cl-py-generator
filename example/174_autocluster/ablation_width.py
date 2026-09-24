"""Width ablation: same pilot pipeline at k=768 vs k=3072 (prompt requirement).

Quantifies whether full 3072 Matryoshka dims beat the 768 prefix.
"""

import argparse

from run_pilot import run_pilot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[768, 3072])
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    for w in a.widths:
        run_pilot(width=w, cache="umap_cache_k%d" % w,
                  out="results_width%d.csv" % w, db=a.db)


if __name__ == "__main__":
    main()
