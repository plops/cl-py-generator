# Repository Guidelines

## Project Structure & Module Organization
- Core Common Lisp sources live at the repo root, notably `py.lisp` (Python emitter), `pipe.lisp` (pipeline helpers), and `package.lisp`/`cl-py-generator.asd` (package/system definitions).
- Tests are defined in `transpiler-tests.lisp`; the differential tests for the parenthesis elision live in `paren-tests.lisp`. The `tests.lisp` file is ad-hoc scratch usage, not the main suite.
- Example outputs and experiments are under `example/` (many numbered subprojects).
- Tooling and hooks live in `tools/` (notably `lisp-format` and a pre-commit script).
- `SUPPORTED_FORMS.md` is generated from tests; do not edit it by hand.

## Build, Test, and Development Commands
- `./run-tests.sh` runs SBCL, loads `transpiler-tests.lisp`, and executes `run-transpiler-tests`.
- When running SBCL tests or tools programmatically, always pass the `--disable-debugger` command-line argument. This prevents SBCL from hanging in the interactive debugger on error. Refer to the [lisp-dev](file:///home/kiel/stage/cl-py-generator/.agents/skills/lisp-dev/SKILL.md) skill for detailed options.
- `./run-paren-tests.sh` runs the differential tests of the parenthesis elision: every expression is emitted with and without `:omit-redundant-parentheses` and both variants are evaluated by `python3`, so a dropped parenthesis shows up as a different value.
- `./generate-docs.sh` regenerates `SUPPORTED_FORMS.md` from the test cases.
- Both scripts assume Quicklisp and the repo path `~/quicklisp/local-projects/cl-py-generator`; adjust if your local setup differs.

## Coding Style & Naming Conventions
- Use idiomatic Common Lisp naming (lowercase with hyphens, e.g., `emit-py`, `run-transpiler-tests`).
- Keep s-expression indentation consistent with existing files; prefer formatting via `git lisp-format` (see `tools/pre-commit`).
- Avoid editing generated artifacts directly; regenerate them from source tests instead.

## Testing Guidelines
- Add new forms to `*test-cases*` in `transpiler-tests.lisp` with `:name`, `:description`, `:lisp`, `:python`, and `:tags`. The list is already quoted, so write `:tags (:core :call)` — an extra quote breaks tag filtering and doc grouping.
- Use `:exec-test t` and `:expected-output` when validating runtime behavior.
- The test runner formats Python using `ruff format` and executes with `python3`; ensure both are available.
- **Parentheses**: the fully parenthesized mode (`:omit-redundant-parentheses nil`) is the oracle of `paren-tests.lisp`; keep it correct. Never decide about parentheses from the head of a form alone — `(- x)` prints a unary minus, `(+ x)` prints just `x`; use `effective-operator`. Every operator named in `*infix-operators*`, `*prefix-operators*`, `*chaining-operators*` or `*associative-operators*` must appear in `*precedence*` (`check-operator-tables` enforces this at load time).
- **Float Precision**: `print-sufficient-digits-f64` relies on the ANSI read/print consistency guarantee: it binds `*read-default-float-format*` to the type of the number being printed and uses `prin1-to-string`, then normalizes the exponent marker (`d`/`s`/`f`/`l`) to `e` so the literal is valid Python. Never print floats with `~G`/a fixed digit count here — a mismatch between `*read-default-float-format*` and the printed float type causes precision loss (double literals were previously truncated to 12 significant digits) or infinite print-reconstruct loops.

## Commit & Pull Request Guidelines
- Recent history shows short, descriptive messages (e.g., “pose plotter”) and occasional Conventional Commits (`feat(scope): ...`). Keep messages concise and specific.
- PRs should include a clear summary, the tests run (or a note if not run), and regenerated docs when emitter behavior changes.
