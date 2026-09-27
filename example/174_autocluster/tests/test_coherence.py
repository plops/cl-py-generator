"""Kohaerenz-Tests: NPMI trennt Themen von Zufall, Kohäsion deterministisch (CPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "doe"))

from coherence import NpmiCorpus, cohesion, tokenize  # noqa: E402

SPORT = ["fussball", "torwart", "stadion", "trainer", "meisterschaft",
         "spieltag", "abseits", "elfmeter", "bundesliga", "torschuss"]
KOCHEN = ["rezept", "zutaten", "backofen", "teig", "gewuerze", "kochtopf",
          "schmoren", "vorspeise", "dessert", "kuechenmesser"]


def docs_one_topic(vocab, n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(vocab, 12)) for _ in range(n)]


def test_tokenize_drops_stopwords_and_short():
    toks = tokenize("Der Torwart hält den Elfmeter in der Bundesliga!")
    assert "torwart" in toks and "elfmeter" in toks
    assert "der" not in toks and "den" not in toks and "in" not in toks


def test_npmi_separated_topics_beat_random():
    rng = np.random.default_rng(1)
    sport = docs_one_topic(SPORT, 60, seed=1)
    kochen = docs_one_topic(KOCHEN, 60, seed=2)
    mixed = [" ".join(rng.choice(SPORT + KOCHEN, 12)) for _ in range(60)]
    corp = NpmiCorpus(min_df=3).fit(sport + kochen + mixed)
    c_sport = corp.coherence(list(range(0, 60)))
    c_koch = corp.coherence(list(range(60, 120)))
    c_mix = corp.coherence(list(range(120, 180)))
    assert np.isfinite(c_sport) and np.isfinite(c_mix)
    assert c_sport > c_mix + 0.15, (c_sport, c_mix)
    assert c_koch > c_mix + 0.15, (c_koch, c_mix)


def test_npmi_deterministic_and_bounded():
    docs = docs_one_topic(SPORT, 40, seed=3)
    corp = NpmiCorpus(min_df=2).fit(docs)
    a = corp.coherence(list(range(40)))
    b = corp.coherence(list(range(40)))
    assert a == b and -1.0 <= a <= 1.0


def test_npmi_needs_two_words():
    corp = NpmiCorpus(min_df=1).fit(["fussball fussball fussball"])
    assert np.isnan(corp.coherence([0]))


def test_cohesion_identical_is_one():
    X = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    assert cohesion(X, list(range(10))) == 1.0


def test_cohesion_separated_beats_random():
    rng = np.random.default_rng(4)
    tight = rng.normal(size=(50, 20)) * 0.1 + 5.0
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)
    loose = rng.normal(size=(50, 20))
    loose /= np.linalg.norm(loose, axis=1, keepdims=True)
    assert cohesion(tight, list(range(50))) > 0.9
    assert cohesion(loose, list(range(50))) < 0.3


def test_cohesion_singleton_is_nan():
    assert np.isnan(cohesion(np.eye(3), [1]))
