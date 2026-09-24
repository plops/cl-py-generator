"""Pure-logic tests: scoring formula, eps scale, label stats (no GPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_umap import adjusted_score, eps_for_dim, label_stats


def test_adjusted_score_perfect_clusters_no_noise():
    assert adjusted_score(1.0, 0.0) == 1.0


def test_adjusted_score_penalizes_noise():
    assert adjusted_score(0.5, 0.2) == 0.4


def test_eps_grows_with_dim():
    assert eps_for_dim(16) > eps_for_dim(4) > eps_for_dim(2)


def test_label_stats_counts_noise_and_clusters():
    st = label_stats(np.array([0, 0, 1, 1, -1, -1]))
    assert st["n_clusters"] == 2
    assert abs(st["noise_ratio"] - 1 / 3) < 1e-9


def test_label_stats_all_noise():
    st = label_stats(np.array([-1, -1]))
    assert st["n_clusters"] == 0 and st["noise_ratio"] == 1.0
