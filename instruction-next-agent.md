# Instruction for Next Agent: Transpiler Pattern Analysis and Test Expansion

## Context & Objectives
You are tasked with continuing the task of identifying transpiler input language use patterns in `example/*/gen*.lisp` files that are not yet covered in [transpiler-tests.lisp](file:///home/kiel/stage/cl-py-generator/transpiler-tests.lisp).
- Currently, examples from index 103 to 171 have been analyzed.
- In the index 1 to 102 range, the following 15 folders were analyzed and sampled:
  - `example/01_plot`
  - `example/02_qt`
  - `example/03_cl`
  - `example/05_trellis_qt`
  - `example/27_thinkpad_fanspeed`
  - `example/36_cadquery`
  - `example/44_zernike`
  - `example/46_opticspy`
  - `example/56_myhdl`
  - `example/60_py_webull`
  - `example/68_arith`
  - `example/74_gr_plot`
  - `example/80_fulltext`
  - `example/84_lte`
  - `example/88_plotly`
- All other folders within the **index 1 to 102** range remain **UNCHECKED**.
- **INSTRUCTION FOR NEXT AGENT**: Do not take shortcuts by scanning or skipping folders. You must systematically inspect the remaining folders index-by-index to identify unusual patterns or variant usages of existing forms (e.g., edge cases of existing DSL nodes, complex nested expressions, or combinations of arguments) that might not be adequately tested.
- Locate 5 to 7 undocumented or untested transpiler constructs/patterns.

## Target Repositories & Files
1. [transpiler-tests.lisp](file:///home/kiel/stage/cl-py-generator/transpiler-tests.lisp): Append new test cases here under `*test-cases*`.
2. [example-to-test.md](file:///home/kiel/stage/cl-py-generator/example-to-test.md): Add new mappings from the identified pattern to the source file name.
3. [package.lisp](file:///home/kiel/stage/cl-py-generator/package.lisp): If any newly tested DSL keyword needs to be exported from the `:cl-py-generator` package, add and export it.

## Step-by-Step Execution Plan

### 1. Research & Analysis
- Browse example folders starting from index 102 downwards (e.g. `example/102_fisher`, `example/101_tex_layout`, etc.).
- Look inside their `gen*.lisp` files for structures that do not match the existing list of test cases in [transpiler-tests.lisp](file:///home/kiel/stage/cl-py-generator/transpiler-tests.lisp).
  - Target areas of interest:
    - Complex arithmetic, array slices, or matrix operations.
    - Multi-line templates or raw string manipulation.
    - Special decorators, inline functions, or custom macros.
    - Package dependencies or custom package exports.
    - Usages that rely on awkward workarounds (such as raw string code injection or parenthesized statements like `raise(...)` or `assert(...)`) that would benefit from introducing a new, first-class S-expression form in `py.lisp`. Add any such suggestions to [suggestions-new-forms.md](file:///home/kiel/stage/cl-py-generator/suggestions-new-forms.md).

### 2. Implementation
- For each identified pattern (aim for 5 to 7 patterns), add a test case in `transpiler-tests.lisp`.
- Add a new row to the table in `example-to-test.md` with:
  - Pattern Name
  - Lisp Form
  - Expected Python
  - Reference File Path

### 3. Verification
- Run the test suite:
  ```bash
  ./run-tests.sh
  ```
- Regenerate the docs from the tests:
  ```bash
  ./generate-docs.sh
  ```
- Verify that `SUPPORTED_FORMS.md` is updated.

### 4. Git Commits
- Commit changes using Conventional Commits guidelines with a thorough description of the added constructs. E.g.:
  ```bash
  git commit -m "test(core): add unit tests for <pattern1>, <pattern2>" -m "Add unit tests and documentation mappings for newly identified transpiler constructs (<pattern1>, <pattern2>)."
  ```

### 5. Update this file

Change the index of the remaining examples, so that the next agent knows which example folders have been checked already.

## Important Alerts
> [!IMPORTANT]
> **Symbol Package Matching Alert**:
> The transpiler dispatcher uses exact symbol equality (`eq`) to match DSL keywords (like `slice`, `paren*`, etc.).
> - Make sure the symbols in your test cases are either exported by `:cl-py-generator` or package-qualified (e.g. `cl-py-generator::paren*`).
> - Unexported symbols read in another package will fallback to generic Python function calls (e.g., `symbol-name()`).

> [!TIP]
> **Ruff Format Padding**:
> When writing expected python strings, trailing commas inside tuples (e.g., `(a, b,)`) dictate formatting style. If Ruff format fails with indentation diffs, ensure parentheses match and add trailing commas to expected python tuples as necessary.
