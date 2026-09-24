# ENV_de.md — Umgebung (RTX A4000, CUDA-UMD 13.4, Ubuntu-26-Container)

Deutsche Übersetzung von `ENV.md`.

Projektlokale Env: `.venv/` (uv). Neu anlegen mit:

```
uv venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://pypi.nvidia.com
```

Gepinnt am 2026-09-24 (verifiziert: `import cuml, cupy` + GPU-Alloc ok):

```
numpy==2.4.6
pandas==3.0.3
matplotlib==3.11.2
scikit-learn==1.9.1
pytest==9.1.1
cuml-cu12==26.08.00
cupy-cuda12x==14.2.0
```

Hinweise:

- `cuml-cu12`/`cupy-cuda12x` stammen von `https://pypi.nvidia.com` (CUDA-12-Wheels
  laufen unter dem CUDA-13.4-UMD-Treiber auf der A4000 problemlos).
- cuML warnt: `build_algo='nn_descent'` ist nicht deterministisch; für
  bitreproduzierbare Runs `build_algo='brute_force_knn'` nutzen (langsamer, exakt).
- DB immer read-only öffnen (`mode=ro`); nie in
  `/workspace/src/rs-summarizer/summaries.db` schreiben.
