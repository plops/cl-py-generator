# Repository Guidelines

## Project Structure & Module Organization
- Core Common Lisp sources live at the repo root, notably `py.lisp` (Python emitter), `pipe.lisp` (pipeline helpers), and `package.lisp`/`cl-py-generator.asd` (package/system definitions).
- Tests are defined in `transpiler-tests.lisp`. The `tests.lisp` file is ad-hoc scratch usage, not the main suite.
- Example outputs and experiments are under `example/` (many numbered subprojects).
- Tooling and hooks live in `tools/` (notably `lisp-format` and a pre-commit script).
- `SUPPORTED_FORMS.md` is generated from tests; do not edit it by hand.

## Build, Test, and Development Commands
- `./run-tests.sh` runs SBCL, loads `transpiler-tests.lisp`, and executes `run-transpiler-tests`.
- When running SBCL tests or tools programmatically, always pass the `--disable-debugger` command-line argument. This prevents SBCL from hanging in the interactive debugger on error. Refer to the [lisp-dev](file:///home/kiel/stage/cl-py-generator/.agents/skills/lisp-dev/SKILL.md) skill for detailed options.
- `./generate-docs.sh` regenerates `SUPPORTED_FORMS.md` from the test cases.
- Both scripts assume Quicklisp and the repo path `~/quicklisp/local-projects/cl-py-generator`; adjust if your local setup differs.

## Coding Style & Naming Conventions
- Use idiomatic Common Lisp naming (lowercase with hyphens, e.g., `emit-py`, `run-transpiler-tests`).
- Keep s-expression indentation consistent with existing files; prefer formatting via `git lisp-format` (see `tools/pre-commit`).
- Avoid editing generated artifacts directly; regenerate them from source tests instead.

## Testing Guidelines
- Add new forms to `*test-cases*` in `transpiler-tests.lisp` with `:name`, `:description`, `:lisp`, `:python`, and `:tags`.
- Use `:exec-test t` and `:expected-output` when validating runtime behavior.
- The test runner formats Python using `ruff format` and executes with `python3`; ensure both are available.
- **Float Reader Precision Alert**: When formatting and parsing floating-point representations in Lisp (`read-from-string`), be aware of `*read-default-float-format*`. A mismatch between the default read format (`single-float`) and the printed float type (e.g. double-float) will cause precision mismatches and infinite print-reconstruct loops. Always bind `*read-default-float-format*` to match the target float type.

## Commit & Pull Request Guidelines
- Recent history shows short, descriptive messages (e.g., “pose plotter”) and occasional Conventional Commits (`feat(scope): ...`). Keep messages concise and specific.
- PRs should include a clear summary, the tests run (or a note if not run), and regenerated docs when emitter behavior changes.
