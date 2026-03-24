## Repo summary: `cl-py-generator`

This repository is a small Common Lisp system that provides a DSL and emitter for generating Python code from S-expressions. You describe Python programs as structured Lisp forms (often produced by macros), and the library emits syntactically-correct Python source (and also Jupyter notebooks).

### What you get

- **`emit-py`**: converts supported S-expression forms into Python text.
- **`write-source`**: writes generated Python to `<name>.py` (with a hash-table to avoid rewriting identical output).
- **`write-notebook`**: builds a `.ipynb` JSON file from `(markdown ...)` and `(python ...)` cells, then formats it via `jq`.

### Key files

#### `cl-py-generator.asd` (ASDF system definition)
- Defines one ASDF system: `cl-py-generator`
- Depends on: `alexandria`, `jonathan`, `external-program`
- Components loaded in order (`:serial t`): `package.lisp`, `py.lisp`, and conditionally `pipe.lisp` on SBCL.

#### `package.lisp` (public API + DSL surface)
- Defines the `:cl-py-generator` package.
- Exports:
  - core API: `emit-py`, `write-source`, `write-notebook`
  - the DSL “node names” that `emit-py` recognizes: literals/containers (`list`, `tuple`, `dict`, ...), statements (`def`, `class`, `if`, `for`, `while`, `with`, `try`, ...), operators (`:+`, `:==`, ...), and helpers (`dot`, `aref`, `slice`, `comments`, ...).

#### `py.lisp` (implementation)
- Implements:
  - **Emitter**: `emit-py` as a large dispatcher on the head symbol of each form.
  - **Function parsing / typing**: `consume-declare` + `parse-defun` read leading `(declare ...)` forms and emit Python type annotations when available.
  - **File output**: `write-source` emits code and writes only if output content has changed since last write.
  - **Notebook output**: `write-notebook` builds notebook JSON (`jonathan:to-json`) and pretty-prints it with `jq`.

#### `pipe.lisp` (SBCL-only interactive helper)
- Starts a background `python3` process and streams output.
- Provides `run` to send generated code (via `emit-py`) into the running Python process.

#### `SUPPORTED_FORMS.md` (auto-generated behavior/spec)
- Documentation generated from test cases, showing:
  - the input S-expression
  - the expected generated Python (after formatting)
- Functions as both user reference and regression spec.

#### `transpiler-tests.lisp` (tests + doc generator)
- Defines a table-driven test suite (`*test-cases*`).
- Normalizes expected/actual code using `ruff format` before comparison.
- Optionally performs execution tests by running the generated Python with `python3`.
- Can generate `SUPPORTED_FORMS.md` via `generate-documentation`.

### How code generation works (high level)
- `emit-py` is effectively a pretty-printer: it pattern-matches on known DSL heads (e.g. `if`, `for`, `dict`, `dot`) and emits Python syntax with `format`.
- Block forms emit a header (`...:`) and then indent the body via `(do ...)`/`(indent ...)`.
- `def` routes through `parse-defun` to handle lambda-lists and optional type annotations derived from `(declare ...)`.

### If you want a “repo map” summary too
Add `git ls-files` output (or a top-level `find . -maxdepth 2 -type f`) and I’ll summarize the directory layout and how examples/tests are organized across the repo.
