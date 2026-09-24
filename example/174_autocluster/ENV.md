# ENV.md — Environment (RTX A4000, CUDA-UMD 13.4, Ubuntu 26 container)

Project-local env: `.venv/` (uv). Recreate with:

```
uv venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://pypi.nvidia.com
```

Pinned 2026-09-24 (verified working, `import cuml, cupy` + GPU alloc ok):

```
numpy==2.4.6
pandas==3.0.3
matplotlib==3.11.2
scikit-learn==1.9.1
pytest==9.1.1
cuml-cu12==26.08.00
cupy-cuda12x==14.2.0
```

Notes:

- `cuml-cu12`/`cupy-cuda12x` come from `https://pypi.nvidia.com` (CUDA-12 wheels
  run fine under the CUDA-13.4 UMD driver on the A4000).
- cuML warns `build_algo='nn_descent' is not deterministic`; for bit-reproducible
  runs use `build_algo='brute_force_knn'` (slower, exact).
- DB is opened read-only (`mode=ro`); never write to
  `/workspace/src/rs-summarizer/summaries.db`.
