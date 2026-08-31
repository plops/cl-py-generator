# cl-py-generator

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/plops/cl-py-generator)

A Common Lisp library that transpiles S-expressions into Python code. Write Python using Lisp syntax and leverage the power of Lisp macros for Python code generation.

## Why cl-py-generator?

- **Lisp macros for Python**: Use Common Lisp's powerful macro system to generate Python code
- **Type-safe code generation**: Generate correct Python code through S-expression transformation
- **Extensive library support**: Over 170 working examples covering NumPy, PyTorch, JAX, Django, FastAPI, Qt, and more
- **Well-tested**: Comprehensive test suite with documented behavior for all supported forms

## Quick Start

```lisp
(ql:quickload "cl-py-generator")

;; note: write-source appends ".py" and, for a relative name, writes into the
;; third argument (default: your home directory)
(cl-py-generator:write-source
  "output"
  '(do0
     (imports ((np numpy)))
     (def calculate-mean (data)
       (return (np.mean data)))
     (setf result (calculate-mean (list 1 2 3 4 5)))
     (print result))
  #p"/tmp/")
```

This generates clean, readable Python:

```python
import numpy as np

def calculate_mean(data):
    return np.mean(data)

result = calculate_mean([1, 2, 3, 4, 5])
print(result)
```

## Documentation

### Supported Forms

The library supports a comprehensive set of Python constructs through S-expressions. See [SUPPORTED_FORMS.md](SUPPORTED_FORMS.md) for auto-generated documentation with examples for all supported forms. 

Key features include:
- Data structures: lists, tuples, dictionaries, sets
- Control flow: if/when/unless, for/while loops, comprehensions
- Functions and classes: def, lambda, class definitions with inheritance
- Operators: arithmetic, comparison, logical, bitwise
- Python-specific: context managers (with), try/except, decorators, f-strings
- Import statements and module management 

### API

The main exported functions are:

- `emit-py` - Convert S-expressions to Python code
- `write-source` - Write generated Python code to a file (`(write-source name code &optional dir ignore-hash)`; `.py` is appended to `name`)
- `write-notebook` - Generate Jupyter notebook files

### External tools

Generated files are pretty printed with an external formatter. `write-source`
looks for `ruff` in `PATH` and falls back to `uvx ruff format`; if neither is
available the code is written unformatted and a warning is signalled. Bind
`cl-py-generator:*python-format-command*` to a command list (e.g.
`(list "black" "-q")`) to choose a different formatter, or to `:none` to switch
formatting off. `write-notebook` additionally uses `jq` to pretty print the
notebook JSON, and the test suite needs `ruff` and `python3`.

## Examples

The repository contains over 170 real-world examples demonstrating the library's capabilities:

**Machine Learning & Data Science:**
- FastAI, PyTorch, JAX with automatic differentiation
- NumPy, CuPy, Numba for numerical computing
- OpenCV, MediaPipe for computer vision
- Matplotlib, Plotly for visualization

**Web Development:**
- Django, Flask, FastHTML web frameworks
- Browser automation with Playwright and Helium
- RESTful APIs and web scraping

**GUI Applications:**
- Qt (PySide), Kivy, wxPython, Tkinter
- 3D visualization with VTK and Open3D

**Hardware & Systems:**
- CUDA, ROCm for GPU computing
- MyHDL, Migen for hardware description
- Serial communication and embedded systems

**And many more!** Browse the [example/](example/) directory for complete working projects. 

## Installation

```lisp
(ql:quickload "cl-py-generator")
```

Or clone this repository and load it directly:

```lisp
(load "package.lisp")
(load "py.lisp")
```

## Design Philosophy

cl-py-generator treats Python generation as a transpilation problem. Instead of string manipulation, it uses structured S-expressions that map cleanly to Python's AST, ensuring syntactically correct output.

The library is particularly useful for:
- Generating repetitive Python code
- Creating code generation pipelines
- Maintaining Python projects where you want macro-level abstractions
- Prototyping Python APIs with Lisp's REPL workflow

## Similar Projects

- [Hy](https://github.com/hylang/hy) - A Lisp dialect that runs on Python
- [Basilisp](https://github.com/basilisp-lang/basilisp) - A Clojure-compatible Lisp for Python
- [waxeye](https://waxc.netlify.app/) - Another S-expression to Python transpiler 

   

