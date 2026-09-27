# PHASEA_de.md — Runde 2, Phase A: CCD-Bestaetigung × Dedup (DE)

Datum: 2026-09-27 · Code: `doe/run_phasea.py` (+ `dedup_map`, `generate_ccd_design`,
`rsm_argmax`) · Tests: +4 in `tests/test_doe.py` · GPU: RTX A4000

## 1. Ziel

Zwei Fragen aus Runde 1 / dem Fidelity-Follow-up schliessen:

1. **Liegt das Optimum unterhalb von mcs=10?** (Runden-1-Optimum mcs=11 an
   der alten Untergrenze 10 → CCD-Verfeinerung mcs[5,19] × ms[5,15] ×
   nn[24,60], d=11/md=0.09/k=3072 fixiert, 20 Punkte × 5 Seeds.)
2. **Helfen oder schaden die 8 % Dubletten?** (Dedup{an,aus} als Block;
   Scoring stets rueckprojiziert auf volle N=16.692 → fairer Vergleich:
   Dedup wirkt nur auf UMAP+HDBSCAN-Fitting.)

206 Fits total (200 CCD + 6 Konfirmierung mit frischen Seeds).

## 2. Ergebnisse

### 2.1 Dedup schadet massiv und einstimmig: Δ = −0,028

ANOVA (gepoolt, R²=0,975, n=38): Dedup-Haupteffekt F=860, p<1e-19 —
mit Abstand dominant. Alle 19 gepaarten Configs schlechter mit Dedup
(0,132 → 0,104 im Mittel), frische Seeds bestaetigen (0,134 vs. 0,107).
Noise-Level identisch (~0,35) — der Verlust ist schlechtere Separation,
nicht mehr Noise. Konservativ gemessen: rueckprojizierte Zwillinge teilen
sich stets das Cluster (gratis-hohe Silhouette), trotzdem −0,028.

### 2.2 Dedup destabilisiert 5-fach (Seed-Streuung)

Mittlere Seed-Std: ohne Dedup 0,0013 (wie Runde 1), mit Dedup 0,0072
(max 0,015). Deutung: Dubletten sind fixierte Null-Distanz-Anker, die
UMAP-Graph und HDBSCAN stabilisieren; ohne sie schlaegt nn_descent-Jitter
voll durch. ARI aehnlich (0,66 vs. 0,64) — Labels aehnlich stabil, Scores nicht.

### 2.3 Einzige signifikante Interaktion: dedup × nn (F=27, p<1e-4)

nn-Stufenmittel: ohne Dedup flach (24→0,131, 42→0,133, 60→0,131), mit Dedup
steil fallend (24→0,109, 42→0,105, 60→0,094). Ohne Anker-Dubletten waescht
grosses nn die Struktur aus → optimales nn je Block verschieden (50 vs. 24).
Die vermutete dedup×mcs-Interaktion wurde **widerlegt** (p=0,33).

### 2.4 Zoom-Box ist flach: mcs/ms/nn alle insignifikant (p>0,15)

Im verfeinerten Bereich [5,19]×[5,15]×[24,60] sind alle Configs statistisch
gleichauf; der RSM-Argmax an mcs=5 ist Rausch-Chasing, kein Befund.
mcs=5 fragmentiert auf ~400–465 Cluster bei ARI≈0,52 — ohne signifikanten
Score-Gewinn. **Empfehlung: mcs=11 (Runden-1-Optimum) behalten**, nicht jagen.

## 3. Conclusion

Phase A beantwortet beide Fragen negativ-konstruktiv: (1) Unterhalb mcs=10
liegt kein signifikant besseres Optimum — die Box ist flach, mcs=11 bleibt.
(2) Dedup ist kontraproduktiv (−0,028 Score, 5× mehr Jitter) — die Pipeline
soll Dubletten behalten; sie sind stabilisierende Anker, kein Schmutz.
Gefaellig daneben: nn nur ohne Dubletten kritisch (klein halten). Naechster
Schritt bleibt Phase B (Bake-off inkl. Leiden + LLM-Kohaerenz-Validierung),
denn die Score-Fragmentierungs-Debatte (217 vs. 465 Cluster) entscheidet nur
externe Themen-Qualitaet.

## 4. Artefakte

- `doe/results/results_phasea_{raw,agg,confirm}.csv`, `design_phasea.csv`,
  `best_phasea.json` (Argmax + Konfirmierung je Block).
- `doe/plots/phasea_{mcs_slice,dedup_pair,count}.png`.
- Abgelehnte Config (beide Bloecke): pt=3 (mcs=5, ms=15, nn=60) — max ms +
  max nn → Noise > 40 %, plausibel.
- Cache `doe/umap_cache_doe/pa_*.npy` (per `.gitignore` lokal).
