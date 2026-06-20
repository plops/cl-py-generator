---
name: lisp-dev
description: Guidelines and documentation for developing, testing, and running Common Lisp code non-interactively using SBCL.
---

# Lisp Development Skill

This skill contains instructions and guidelines for running Common Lisp code, executing test suites, and generating documentation using Steel Bank Common Lisp (SBCL) in a non-interactive/batch environment.

## Running SBCL Non-Interactively

When running Lisp scripts, tests, or builds in automated systems (like CI/CD pipelines, pre-commit hooks, or background tasks), you must prevent SBCL from dropping into the interactive debugger when an unhandled error occurs.

### Preventing Interactive Debugger Hangs
By default, SBCL enters an interactive debugger prompt (`*` or `db>`) upon encountering an error, waiting for user input. This causes background tasks and automated runners to hang.

To disable the debugger and force SBCL to print a backtrace and exit immediately with a non-zero code on error, use the `--disable-debugger` option:

```bash
sbcl --disable-debugger \
     --load file.lisp \
     --eval '(form)' \
     --quit
```

### Key SBCL Flags:
- `--disable-debugger`: Disables the interactive debugger. If an error occurs, a backtrace is printed and SBCL exits with code 1.
- `--non-interactive`: (SBCL 2.0+) Sets up SBCL to run in a non-interactive batch mode, though `--disable-debugger` is still needed to guarantee exits on all errors.
- `--noinform`: Suppresses the startup banner message.
- `--load <file>`: Loads a Lisp file before starting the REPL or evaluating eval forms.
- `--eval <form>`: Evaluates a Lisp s-expression form.
- `--quit`: Exits SBCL when all options and eval forms have been processed.

---

## Workspace Workflows

### 1. Running the Test Suite
The repository includes a script `./run-tests.sh` to run the transpiler tests.

To run tests:
```bash
./run-tests.sh
```

To run a subset of tests using tags, or customize test execution, run SBCL directly:
```bash
sbcl --disable-debugger \
     --load ~/quicklisp/local-projects/cl-py-generator/transpiler-tests.lisp \
     --eval "(cl-py-generator/tests::run-transpiler-tests :tags '(:core))" \
     --quit
```

### 2. Generating Supported Forms Documentation
To regenerate `SUPPORTED_FORMS.md` from the current test cases:
```bash
./generate-docs.sh
```

---

## Coding Style & Standards
- Keep s-expression indentation consistent with existing files.
- Prefer formatting via `git lisp-format` (see `tools/pre-commit`).
- Always run the test suite and documentation generator before committing or submitting a PR.
