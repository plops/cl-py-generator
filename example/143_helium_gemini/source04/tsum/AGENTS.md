# Repository Guidelines

## Project Structure & Module Organization
This repository is a small FastHTML app for summarizing YouTube transcripts with Gemini.
Key files and directories:
- `p04_host.py` is the main FastHTML + Uvicorn entrypoint and orchestration layer.
- `s01_*.py` through `s04_*.py` are focused utility modules (validation, VTT parsing, formatting).
- `t01_*.py` through `t04_*.py` are lightweight, assert-based test scripts.
- `doc/` contains architecture notes (`doc/architecture.md`).
- `data/` and `exports/` hold local inputs/outputs.
- Static assets live at the repo root (`htmx.min.js`, `pico.min.css`, `script.js`, `favicon.*`).

## Build, Test, and Development Commands
- `uv sync` installs dependencies from `pyproject.toml`/`uv.lock`.
- `GEMINI_API_KEY=... uv run start` runs the app via the `start` script (Uvicorn on port 5001).
- `GEMINI_API_KEY=... uv run uvicorn p04_host:app --port 5001 --host 0.0.0.0` is an explicit alternative.
- `python t01_validate_youtube_url.py` (and other `t##_*.py`) runs the assert-based checks.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, snake_case for functions and variables.
- Script naming is ordinal and role-based: `p##_` for main pipeline, `s##_` for helpers, `t##_` for tests.
- Keep modules small and single-purpose; match existing patterns for logging and validation.

## Testing Guidelines
- Tests are simple scripts with top-level `assert` statements; no pytest harness.
- Add or update `t##_*.py` scripts when modifying `s##_*.py` behavior.
- Prefer deterministic inputs (e.g., fixtures in `data/`) to keep tests reliable.

## Commit & Pull Request Guidelines
- Recent history uses Conventional Commits with scopes (e.g., `feat(args): add ROI configuration`).
- Keep subjects short and imperative; include a scope when it adds clarity.
- PRs should include a concise summary, how you tested, and screenshots for UI changes.

## Configuration & Secrets
- Set `GEMINI_API_KEY` before running the app; requests will fail without it.
- Logs are written to `transcript_summarizer.log`; include relevant excerpts when reporting issues.
