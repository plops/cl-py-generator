---
name: cl-py-generator
description: Provides documentation and guides code generation using the cl-py-generator Common Lisp to Python transpiler. Use when writing, modifying, or testing Common Lisp forms to be transpiled into Python.
---

# CL-Py-Generator Transpiler

This skill documents the Lisp-like S-Expression DSL used by the `cl-py-generator` transpiler to emit Python code. Use this documentation to write correct Lisp forms, extend the transpiler, or understand generated Python outputs.

## When to use this skill

- When implementing new features or writing new Lisp generator code in `example/` or at the workspace root.
- When adding or modifying test cases in [transpiler-tests.lisp](file:///home/kiel/stage/cl-py-generator/transpiler-tests.lisp).
- When debugging generated Python files or updating the core emitter in [py.lisp](file:///home/kiel/stage/cl-py-generator/py.lisp).

## Language Architecture

The transpiler takes Lisp S-Expressions representing a Python AST and compiles them into Python code strings.
The package name is `:cl-py-generator`.
The primary public entrypoints are:
- `emit-py`: Emits Python code from a Lisp form.
- `write-source`: Transpiles code and writes the result to a `.py` file.
- `write-notebook`: Writes the code to a Jupyter Notebook `.ipynb` file.

The transpiler uses an `:invert` readtable-case, meaning that lower-case code is preserved and maps naturally to lower-case Python identifiers.

---

## DSL Reference & Mapping Guide

Below is a complete reference of the Lisp forms supported by the transpiler and their generated Python syntax.

### 1. Variables & Assignments
- Direct assignment: `(= a b)` &rarr; `a = b`
- Multiple assignments: `(setf a 1 b 2)` &rarr; `a = 1\nb = 2`
- Increment / Decrement:
  - `(incf a 2)` &rarr; `a += 2`
  - `(decf a 3)` &rarr; `a -= 3`

### 2. Basic Operators
All operators wrap their operands in parentheses to preserve operator precedence.
- **Arithmetic**:
  - `(+ a b)` &rarr; `((a) + (b))`
  - `(- a b)` &rarr; `((a) - (b))`
  - `(* a b c)` &rarr; `((a) * (b) * (c))`
  - `(/ a b)` &rarr; `((a) / (b))`
  - `(// a b)` &rarr; `((a) // (b))` (floor division)
  - `(% a b)` &rarr; `((a) % (b))` (modulo)
  - `(** a b)` &rarr; `((a) ** (b))` (exponentiation)
  - `(@ a b)` &rarr; `((a) @ (b))` (matrix multiplication)
- **Bitwise**:
  - `(& a b)` or `(logand a b)` &rarr; `((a) & (b))`
  - `(^ a b)` or `(logxor a b)` &rarr; `((a) ^ (b))`
  - `(logior a b)` or `cl-py-generator::|\||` &rarr; `((a) | (b))`
  - `(<< a b)` &rarr; `((a) << (b))`
  - `(>> a b)` &rarr; `((a) >> (b))`
- **Comparison**:
  - `(== a b)` &rarr; `((a) == (b))`
  - `(!= a b)` &rarr; `((a) != (b))`
  - `(< a b)` &rarr; `((a) < (b))`
  - `(<= a b)` &rarr; `((a) <= (b))`
  - `(> a b)` &rarr; `((a) > (b))`
  - `(>= a b)` &rarr; `((a) >= (b))`
  - `(in a b)` &rarr; `(a in b)`
  - `(not-in a b)` &rarr; `(a not in b)`
  - `(is a b)` &rarr; `(a is b)`
  - `(is-not a b)` &rarr; `(a is not b)`
- **Logical**:
  - `(and a b)` &rarr; `((a) and (b))`
  - `(or a b)` &rarr; `((a) or (b))`

### 3. Collections & Accessors
- **List Literal**: `(list 1 2)` &rarr; `[1, 2]`
- **Tuple Literal**: `(tuple 1 2)` &rarr; `(1, 2,)`
- **Paren Literal (comma separated)**: `(paren a b)` &rarr; `(a, b)`
- **Ntuple Literal (naked tuple)**: `(ntuple a b c)` &rarr; `a, b, c`
- **Set Literal (curly)**: `(curly 1 2)` &rarr; `{1, 2}`
- **Dict Literal**: `(dict ((string "a") 1) ((string "b") 2))` &rarr; `{"a": 1, "b": 2}`
- **Dictionary Constructor**: `(dictionary :a 1 :b 2)` &rarr; `dict(a=1, b=2)`
- **Array / Slice Indexing**:
  - Index access: `(aref arr 1)` &rarr; `arr[1]`
  - Multi-index access: `(aref arr i j)` &rarr; `arr[i, j]`
  - Slice: `(aref arr (slice 1 5 2))` &rarr; `arr[1:5:2]`
  - Open slice start/end (use `nil`):
    - `(slice nil 3)` &rarr; `:3`
    - `(slice 1 nil)` &rarr; `1:`
- **Member Access (dot)**: `(dot obj attr)` &rarr; `obj.attr`

### 4. Control Flow
- **Conditional**:
  - If/Else: `(if condition true-stmt false-stmt)` &rarr; `if condition:\n    true-stmt\nelse:\n    false-stmt`
  - When: `(when condition body*)` &rarr; `if condition:\n    body*`
  - Unless: `(unless condition body*)` &rarr; `if not condition:\n    body*`
  - Ternary: `(? condition true-expr false-expr)` &rarr; `(true-expr) if (condition) else (false-expr)`
  - Cond (multi-branch):
    ```lisp
    (cond ((> a b) (return a))
          ((< a b) (return b))
          (t (return 0)))
    ```
    Emits:
    ```python
    if ( a > b ):
        return a
    elif ( a < b ):
        return b
    else:
        return 0
    ```
- **Loops**:
  - While: `(while (< a b) (setf a (+ a 1)))` &rarr; `while (a < b):\n    a = a + 1`
  - For: `(for (i (range 3)) (print i))` &rarr; `for i in range(3):\n    print(i)`
  - List Comprehension Generator: `(for-generator (i (range 3)) (* i 2))` &rarr; `i * 2 for i in range(3)`
- **Try/Except/Finally**:
  ```lisp
  (try (setf a 1)
       (Exception (setf a 2))
       ((as ValueError e) (setf a 3))
       (else (setf a 4))
       (finally (setf a 5)))
  ```
  Emits:
  ```python
  try:
      a = 1
  except Exception:
      a = 2
  except ValueError as e:
      a = 3
  else:
      a = 4
  finally:
      a = 5
  ```

### 5. Functions & Classes
- **Function Definitions**:
  `def` compiles to a python function. Parameters can be regular or keyword (`&key`).
  ```lisp
  (def foo (x &key (y 1))
    (declare (type int x)
             (type float y)
             (values str))
    (return (str x)))
  ```
  Emits:
  ```python
  def foo(x: int, y: float = 1) -> str:
      return str(x)
  ```
  *Note:* The `declare` form at the start of the function body is processed to extract type hints.
  - `(declare (type <type> <var>))` sets a parameter type.
  - `(declare (values <type>))` sets the return type annotation.
- **Lambdas**:
  `(lambda (x) (+ x 1))` &rarr; `lambda x: x + 1`
- **Class Definitions**:
  `(class Foo (ParentClass) body*)` &rarr; `class Foo(ParentClass):\n    body*`
- **Return Statement**:
  - `(return expr)` or `(return_ expr)` &rarr; `return expr`

### 6. Imports
- Single: `(import sys)` &rarr; `import sys`
- Alias: `(import (np numpy))` &rarr; `import numpy as np`
- Multiple: `(imports (sys (np numpy)))` &rarr; `import sys\nimport numpy as np`
- From-Import: `(import-from math sin cos)` &rarr; `from math import sin, cos`
- Multiple From-Imports:
  ```lisp
  (imports-from (math sin cos)
                (pathlib Path))
  ```
  Emits:
  ```python
  from math import sin, cos
  from pathlib import Path
  ```

### 7. Literals & Code Structure
- **Strings**:
  - Double quotes: `(string "hello")` &rarr; `"hello"`
  - Byte string: `(string-b "data")` &rarr; `b"data"`
  - Triple-quoted: `(string3 "text")` &rarr; `"""text"""`
  - Triple-quoted raw: `(rstring3 "raw")` &rarr; `r"""raw"""`
  - F-string: `(fstring "{x}")` &rarr; `f"{x}"`
  - F-string triple-quoted: `(fstring3 "{x}")` &rarr; `f"""{x}"""`
- **Comments**:
  - Single line: `(comment "note")` &rarr; `# note`
  - Multi line: `(comments "line1" "line2")` &rarr; `# line1\n# line2`
- **Grouping**:
  - `do`: Combines forms sequentially and indents them (adds block indentation level).
  - `do0`: Combines forms sequentially at the current indentation level.
  - `space`: Combines forms separated by space (e.g. `(space a b)` &rarr; `a b`).
  - `symbol`: Replaces hyphens with colons (e.g. `(symbol foo-bar)` &rarr; `foo:bar`).
  - `cell` / `export`: Prefixes the forms with Jupyter cell export comments (`# export` or `# |export`).

---

## Testing & Documentation Workflows

When altering the transpiler codebase or checking how forms render:

1. **Running Tests**:
   - Run the script `./run-tests.sh` to load `transpiler-tests.lisp` and execute `run-transpiler-tests`.
   - The test runner automatically formats code using `ruff format` and checks execution correctness using `python3`.

2. **Adding Test Cases**:
   - Add new cases to `*test-cases*` in `transpiler-tests.lisp` with `:name`, `:description`, `:lisp`, `:python`, and optionally `:exec-test t` and `:expected-output`.

3. **Regenerating Docs**:
   - Run `./generate-docs.sh` to update `SUPPORTED_FORMS.md` with the latest mappings from the test suite. Do not edit `SUPPORTED_FORMS.md` by hand.
