"""Titel-Store-Tests: Signaturen, Jaccard-Matching, Update-Planung (CPU)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "doe"))

from titles import (KEEP_JACCARD, job_sig, load_store, make_store,  # noqa: E402
                    match_clusters, member_sig, plan_update, save_store)


def assign_of(mapping):
    return {i: c for c, ids in mapping.items() for i in ids}


def test_member_sig_order_free_and_sensitive():
    assert member_sig([3, 1, 2]) == member_sig([1, 2, 3])
    assert member_sig([1, 2, 3]) != member_sig([1, 2, 4])


def test_job_sig_order_free():
    assert job_sig([5, 1], [9, 2]) == job_sig([1, 5], [2, 9])
    assert job_sig([1, 5], [2, 9]) != job_sig([1, 5], [2, 10])


def test_match_permuted_labels():
    old = assign_of({0: [1, 2, 3, 4], 1: [5, 6, 7]})
    new = assign_of({7: [1, 2, 3, 4], 3: [5, 6, 7]})
    m = match_clusters(old, new)
    assert m[7]["old"] == 0 and m[7]["jaccard"] == 1.0
    assert m[3]["old"] == 1 and m[3]["jaccard"] == 1.0


def test_match_split_cluster():
    old = assign_of({0: list(range(100))})
    new = assign_of({0: list(range(50)), 1: list(range(50, 100))})
    m = match_clusters(old, new)
    assert m[0]["old"] == 0 and m[0]["jaccard"] == 0.5
    assert m[1]["old"] == 0 and m[1]["jaccard"] == 0.5


def test_match_ignores_noise_and_new_points():
    old = assign_of({0: [1, 2, 3], -1: [4]})
    new = assign_of({5: [1, 2, 3, 100, 101], -1: [4]})
    m = match_clusters(old, new)
    assert m[5]["old"] == 0
    assert abs(m[5]["jaccard"] - 3 / 5) < 1e-9  # 3 overlap / 5 union


def test_match_no_overlap_gives_none():
    m = match_clusters(assign_of({0: [1, 2]}), assign_of({9: [3, 4]}))
    assert m[9]["old"] is None and m[9]["jaccard"] == 0.0


def test_plan_update_keep_vs_retitle():
    store = make_store({"t": 1}, assign_of({0: list(range(100)),
                                            1: list(range(100, 200))}),
                       {0: "A", 1: "B"}, {}, {})
    same = assign_of({5: list(range(100)), 6: list(range(100, 200))})
    keep, ret = plan_update(store, same)
    assert sorted(keep) == [5, 6] and ret == []
    assert keep[5]["title"] == "A" and keep[5]["jaccard"] == 1.0
    # Kleiner Drift (J=95/105) bleibt keep, neuer Split retitelt.
    drifted = assign_of({5: list(range(95)), 6: list(range(100, 200)),
                         7: list(range(95, 100))})
    keep2, ret2 = plan_update(store, drifted)
    assert sorted(keep2) == [5, 6], (keep2, ret2)
    assert [r["new"] for r in ret2] == [7]
    assert KEEP_JACCARD == 0.7


def test_store_roundtrip(tmp_path):
    store = make_store({"m": "hdbscan"}, assign_of({0: [1, 2], 1: [3]}),
                       {0: "Aa", 1: "Bb"}, {0: [1]}, {0: [1], 1: [0]})
    p = str(tmp_path / "s.json")
    save_store(store, p)
    back = load_store(p)
    assert back["titles"]["0"]["members"] == [1, 2]
    assert back["titles"]["0"]["title"] == "Aa"
    assert back["version"] == 1
