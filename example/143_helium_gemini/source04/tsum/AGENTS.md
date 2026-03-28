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

Use the following emacs command to occasionally format the codes (and always before commits):
```
 emacs --batch -l ~/.emacs gen04.lisp       --eval "(package-initialize)"       --eval "(require 'slime)"       --eval "(slime-setup '(slime-cl-indent))"       --eval "(setq lisp-indent-function 'common-lisp-indent-function)"       --eval "(indent-region (point-min) (point-max))"       -f save-buffer
```

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

## S-Expression Transpiler (cl-py-generator)

### Overview
This project uses `cl-py-generator` to transpile Common Lisp S-expressions into Python code. The main transpiler source is in `gen04.lisp`, which generates the main application file `p04_host.py`.

### Key Files
- **`gen04.lisp`**: Main Lisp source file containing S-expressions that generate Python code
- **`p04_host.py`**: Generated Python file (do not edit directly - changes will be lost)
- **`SUPPORTED_FORMS.md`**: Complete reference of supported S-expression forms and their Python output

### Running the Transpiler
```bash
cd /path/to/cl-py-generator/example/143_helium_gemini
sbcl --non-interactive --load gen04.lisp
```

### Supported Instructions
- **Main Repository README**: `/home/kiel/stage/cl-py-generator/README.md`
- **Transpiler Documentation**: `/home/kiel/stage/cl-py-generator/SUPPORTED_FORMS.md`
- **Test Suite**: Run `./run-tests.sh` from the cl-py-generator root directory
- **Generate Documentation**: Run `./generate-docs.sh` from the cl-py-generator root directory

### Common S-Expression Patterns
- **Function Definitions**: `(def function-name (params) ...)`
- **Variable Assignment**: `(setf variable-name value)`
- **Dictionary Creation**: `(dict ((key1 val1) (key2 val2))` or `(dictionary :key1 val1 :key2 val2)`
- **List Creation**: `(list item1 item2 item3)`
- **Conditional Logic**: `(when condition ...)` or `(if condition then-form else-form)`
- **Imports**: `(imports (module1 module2))` or `(imports-from (module1 symbol1 symbol2))`

### Important Notes
- Always regenerate `p04_host.py` from `gen04.lisp` after making changes
- Use `cl-py-generator:in` for Python's `in` operator
- Use `aref` for Python list/dict access with string keys
- The transpiler follows Lisp conventions with hyphenated names (e.g., `cl-py-generator:in`)
