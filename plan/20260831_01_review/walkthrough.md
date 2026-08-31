# Code Review `py.lisp` / `transpiler-tests.lisp` — 2026-08-31

Reviewer: agent for wol pumba (wolpumba@gmail.com)
Environment: Ubuntu container, SBCL 2.6.0.debian, quicklisp, `ruff` + `python3`
from `/workspace/.venv`, `jq`, `uvx` in `/usr/bin`.

Baseline before the review: `./run-tests.sh` reported 135/135 passing.
After the review: **140/140 passing**, docs regenerated, `py.lisp` compiles
without a single warning (it previously produced one full `WARNING` and two
`STYLE-WARNING`s).

---

## 1. Findings

Sorted by impact. Everything marked *fixed* is implemented in this commit.

### 1.1 `write-source` was unusable outside the author's machine (fixed)

```lisp
(sb-ext:run-program "/snap/bin/uvx" (list "ruff" "format" (namestring fn)))
```

The formatter path was hard coded to a snap install. In this container the call
signalled `Couldn't execute "/snap/bin/uvx": No such file or directory` —
i.e. the main entry point of the library aborted *after* writing the file, so
any `gen*.lisp` that emits more than one file died halfway through.

Fix: `format-python-file` + `python-format-command` look up the formatter at
runtime (`ruff` in `PATH`, then `uvx ruff format`, result cached), use
`uiop:run-program` instead of `sb-ext:` (works on non-SBCL implementations),
and never signal: a missing or failing formatter produces a warning and the
unformatted file stays on disk. The new special variable
`*python-format-command*` (exported) overrides the command, `:none` disables
formatting.

The dead `#+nil` branches for `autopep8`, `yapf` and a hard coded
`/home/martin/.local/bin/black` were removed.

### 1.2 Generated notebooks were rejected by `nbformat` (fixed)

`write-notebook` produced

```json
{ "CELLS": [ { "cell_type": "markdown", ... } ], "METADATA": {...}, "NBFORMAT": 4 }
```

Upper case top level keys, lower case cell keys. Jupyter/`nbformat` refuse
this file:

```
ValidationError: Notebook could not be converted from version 1 to version 2
because it's missing a key: cells
```

Root cause — this was the most surprising finding of the review: `jonathan`
prints keyword keys through the printer, so their case depends on
`(readtable-case *readtable*)`, **and** it constant-folds literal plist keys at
compile time. `py.lisp` switches the global readtable to `:invert` at load
time, so the keys of the literal outer plist were baked in during
`compile-file` (readtable still `:upcase` → `"CELLS"`), whereas the cell
plists, built at runtime under `:invert`, came out lower case. The result
depended on whether the file was compiled or loaded as source, and on the
state of the image — a genuinely non-deterministic output.

Fix: the notebook structure is now built as **alists with explicit string
keys** and encoded with `:from :alist`, which is independent of the readtable
and of constant folding. Verified with the real schema:

```
$ python3 -c "import nbformat; nbformat.validate(nbformat.read('/tmp/verify_nb.ipynb', as_version=4))"
nbformat validate: OK
```

The cell body extraction moved into its own function `notebook-cell-source`,
which now signals an error for unknown cell types instead of silently emitting
`null`.

Two related cleanups in the same function:
* the `case` keys were written `` `markdown ``/`` `python `` (backquoted!). They
  read as `(quasiquote markdown)`, i.e. a key *list*, so both clauses also
  matched the symbol `quasiquote` — SBCL reported
  `Duplicate key SB-INT:QUASIQUOTE in CASE form`. Now plain symbols.
* the cell temp file was hard coded to `/dev/shm/cell`; it now uses
  `uiop:temporary-directory`.

### 1.3 Silent precision loss for double float literals (fixed)

`print-sufficient-digits-f64` increased the digit count until the relative
error was below `1d-12`, so **doubles were emitted with only ~12 significant
digits** although the docstring claimed "until the same bit pattern is
obtained". It also emitted the padding of `~,vG`:

| lisp value | old output | new output |
|---|---|---|
| `(/ 1d0 3)` | `0.333333333333` (wrong, does not round-trip) | `0.3333333333333333` |
| `1.2345678901234567d5` | `123456.7890123` | `123456.78901234567` |
| `1d0` | `1.` + 4 trailing blanks | `1.0` |
| `1d30` | `1.0e+30` | `1.0e30` |

Fix: rely on the ANSI read/print consistency guarantee — bind
`*read-default-float-format*` to the type of the number, use
`prin1-to-string`, and normalize a leftover exponent marker (`d s f l`) to
`e`. Shortest round-trip representation, no padding, valid Python.

New regression tests: `float-round-trip` and the execution test
`functional-float-precision`, which asserts in Python that
`0.3333333333333333 == 1 / 3` — this fails with the old emitter.

No committed example output changes because of this (no generated `.py` in
`example/` contains a >12-digit literal), so the fix only affects future
generation.

### 1.4 `(declare (capture x))` crashed with an unbound variable (fixed)

`consume-declare` pushed onto a variable `captures` that was never bound:

```
; caught WARNING: undefined variable: CL-PY-GENERATOR::CAPTURES
...
ERROR: The variable captures is unbound.
```

Nothing in the repository uses the `capture` declaration, which is why it was
never noticed. The variable is now bound and the collected names are stored in
the environment under the key `captures`; the docstring mentions it.

### 1.5 Operators silently dropped arguments (fixed)

```
(/ a b c)  => a/b     ; c disappeared
(** a b c) => a**b
(// a b c) => a//b
(% a b c)  => a%b
(/)        => "/"     ; garbage
```

Fix: `/`, `//` and `%` now chain (left associative, which is what both Common
Lisp and Python mean), `**` signals an error for anything other than two
arguments because Python's `**` is right associative and guessing would be
wrong. Degenerate arities signal errors instead of emitting garbage.
Tests added: `div-chained`, `floor-div-chained`.

`(+)` and `(*)` with zero arguments still emit an empty string; not changed
because the fix is not obvious (CL would say `0`/`1`) and no example relies on
either.

### 1.6 `break` instead of `error` in the emitter (fixed)

Four error paths used `break`. `break` signals a plain `simple-condition`, so
it is **not** caught by `handler-case ... (error ...)`, and under
`--disable-debugger` (which `AGENTS.md` mandates for scripted runs) it kills
the process with a backtrace:

```
3: (SB-INT:%BREAK BREAK "multiple return values unsupported: ~a" (INT INT))
unhandled condition in --disable-debugger mode, quitting
```

All four are now `error`, so a generator script can catch and report them.

### 1.7 `:tags` was quoted twice — tag filtering and doc grouping were dead (fixed)

`*test-cases*` is one big quoted list, but every entry wrote
`:tags '(:core :call)`. Therefore `(getf tc :tags)` returned
`(QUOTE (:CORE :CALL))`:

* `(run-transpiler-tests :tags '(:import))` selected **0** tests.
* `generate-documentation` grouped by `(first tags)` = `QUOTE`, so the whole of
  `SUPPORTED_FORMS.md` consisted of a single section named `## quote Forms`.

Fix: removed the inner quote in all 135 entries. `SUPPORTED_FORMS.md` now has
proper `control-flow / core / import / operator` sections, and
`(run-transpiler-tests :tags (list :import))` runs the 5 import tests.

### 1.8 A failing test aborted the whole test run (fixed)

```lisp
;; Skip to the next test case in the dolist loop
(return)
```

`(return)` returns from the implicit `nil` block of `dolist`, i.e. it left the
loop entirely: the first transpilation failure skipped **all remaining tests**
and still printed a summary. Wrapped the body in a named `block one-test` and
use `return-from`.

### 1.9 `lookup-associativity` declared `op` ignored and then read it (fixed)

`STYLE-WARNING: reading an ignored variable: OP`. Declaration removed.

---

## 2. Documentation updates

* `README.md`: fixed the mangled first heading (`Th# cl-py-generator`); fixed
  the quick start, which passed `"output.py"` to `write-source` and would have
  produced `~/output.py.py` in the user's home directory; documented the
  external tool requirements (`ruff`/`uvx`, `jq`, `python3`) and
  `*python-format-command*`.
* `.agents/skills/cl-py-generator/SKILL.md`: the operator table still described
  the pre-`omit-redundant-parentheses` output (`(+ a b)` → `((a) + (b))`), and
  the `cond`/`while`/ternary examples showed `if ( a > b ):`. Updated to the
  actual default output, added the arity rules, a paragraph on number/float
  emission, the comprehension-filter idiom (see below) and the `:tags`
  pitfall.
* `AGENTS.md`: replaced the float note (which described the removed
  `read-from-string` loop) with the current mechanism, and added the `:tags`
  rule.
* `README_SUMMARY.md`: dependency list and formatter behaviour.
* `SUPPORTED_FORMS.md`: regenerated (now grouped by tag, 140 forms).

### Undocumented feature found: comprehension filters

`(? cond expr)` with the else branch omitted emits `expr if cond`, which is
useless as a statement but is exactly a comprehension filter when `expr` is a
`for-generator`. Two examples use it (`example/162_genai/gen02.lisp`,
`example/143_helium_gemini/gen04.lisp`), and it was documented nowhere:

```lisp
(list (? (> x 0) (for-generator (x xs) x)))   ; => [x for x in xs if x > 0]
```

Added as test `comprehension-filter` and to the skill documentation.

---

## 3. Verification

```
./run-tests.sh        -> Total 140, Passed 140, Failed 0
./generate-docs.sh    -> SUPPORTED_FORMS.md with 4 tag sections
compile-file py.lisp  -> no warnings (was: 1 WARNING, 2 STYLE-WARNINGs)
```

Manual checks in a fresh image (`/tmp/verify.lisp`):

* `write-source` writes and formats a file and returns its pathname (used to
  signal an error here);
* `*python-format-command*` auto-detects `("ruff" "format")`, `:none` disables
  formatting silently;
* `write-notebook` output validates against `nbformat`;
* `(** a b c)`, `(% a)`, `(paren* + a b)` and `(declare (values int int))` are
  now catchable `error`s;
* the generated file prints `True` for the float round-trip check;
* regenerating a real example (`example/166_mbti/gen01.lisp`) works end to end.

---

## 4. Unexpected findings not acted upon

* **`example/166_mbti/p01_mbti.py` is stale.** Regenerating it from
  `gen01.lisp` changes `# 4   INFJ    1.5   1.2     1.6C` to `... 1.6`; the
  stray `C` does not exist in the generator source, so the committed file was
  hand-edited at some point. Reverted to keep this commit focused — worth a
  separate "regenerate all examples" pass.
* **Two test cases are compared unformatted.** `space-basic` (`alpha beta`) and
  `for-generator-basic` (`i * 2 for i in range(3)`) are not valid Python
  modules, so `ruff format` fails on them and `run-ruff-format` falls back to
  the raw string (visible as four `Ruff format failed` warnings per run). The
  comparison is still correct, just not normalized.
* **`emit-py`'s `:str` argument is half-implemented.** Only the number branches
  write to it (`(format str ...)`), every other branch returns a fresh string,
  so passing a stream loses everything but numbers. No caller in the repository
  uses `:str`. Left alone, but it should either be removed or honoured
  everywhere.
* **Dead state.** `*warn-breaking*` is only read inside `#+nil` code;
  `*env-functions*`/`*env-macros*` are set by `:clear-env` but never read, so
  `:clear-env` has no effect at all. Harmless, but misleading.
* **`(setf (readtable-case *readtable*) :invert)` mutates the global
  readtable** as a load side effect. Every `gen*.lisp` depends on it, so it
  cannot simply be removed, but it is what caused finding 1.2 and it will
  surprise anyone loading this system next to other libraries.
* **`in`/`is`/`not-in`/`is-not` always emit surrounding parentheses**, even with
  `omit-redundant-parentheses` and even though they are in the precedence
  table. Cosmetic only (ruff keeps them), so left as is.
* `pipe.lisp` uses SBCL internals (`sb-impl::process-%status`,
  `sb-impl::process-pty`) and is loaded only on SBCL; it still works but will
  break on any SBCL that renames those internals.
* `tools/pre-commit` is not installed as `.git/hooks/pre-commit` in this
  checkout, so `git lisp-format` did not run. `py.lisp` was deliberately not
  reformatted wholesale — that would bury the review in whitespace churn.
