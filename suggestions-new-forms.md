# Proposed New S-Expression Forms for `cl-py-generator`

The following S-Expression forms would improve code readability, AST semantics, and help avoid syntax workarounds (like raw string insertion or helper function fallbacks).

---

## 1. Exception Raising (`raise_`)

### Current Approach (Fallback to Function Call)
Currently, writing `(raise (ValueError "msg"))` compiles to `raise(ValueError("msg"))`, which wraps the `raise` statement in function-call parentheses.
```lisp
;; Lisp Input
(raise (ValueError (string "invalid value")))
```
```python
# Emitted Python
raise(ValueError("invalid value"))
```

### Proposed Form: `raise_`
Introduce a dedicated statement form that prints `raise` followed by the exception expression without surrounding the `raise` keyword in parentheses.
```lisp
;; Proposed Lisp Input
(raise_ (ValueError (string "invalid value")))
```
```python
# Expected Python Output
raise ValueError("invalid value")
```

---

## 2. Dedicated Decorator Construct (`@` / `decorator`)

### Current Approach (Siblings in `do0` block)
Currently, decorators are written as sibling expressions in a `do0` block, which is a structural hack.
```lisp
;; Lisp Input
(do0
  (@rt.route (string "/"))
  (def index ()
    (return (string "Hello"))))
```
```python
# Emitted Python
@rt.route("/")
def index():
    return "Hello"
```

### Proposed Form: `decorator` or `decorate`
Introduce a semantic structure linking decorators explicitly to the functions or classes they decorate.
```lisp
;; Proposed Lisp Input
(decorator (@rt.route (string "/"))
  (def index ()
    (return (string "Hello"))))
```
```python
# Expected Python Output
@rt.route("/")
def index():
    return "Hello"
```

---

## 3. Dedicated Comprehension Forms (`list-comp`, `dict-comp`, `set-comp`)

### Current Approach (Overloaded `slice` and `curly`)
Currently, comprehensions reuse `for-generator` wrapped in `list` / `curly`, and repurpose `slice` to represent `:` inside dictionaries.
```lisp
;; Lisp Input
(curly (for-generator ((ntuple i s) (enumerate chars)) (slice s (+ i 1))))
```
```python
# Emitted Python
{s: i + 1 for i, s in enumerate(chars)}
```

### Proposed Forms
Provide explicit, semantic comprehension forms.
```lisp
;; Proposed Lisp Input (Dict Comprehension)
(dict-comp ((ntuple i s) (enumerate chars))
  s
  (+ i 1))

;; Proposed Lisp Input (List Comprehension)
(list-comp (x (range 5))
  (* x 2))
```
```python
# Expected Python Output
{s: i + 1 for i, s in enumerate(chars)}
[x * 2 for x in range(5)]
```

---

## 4. Assertion Statement (`assert_` / `py-assert`)

### Current Approach
The `assert` keyword is commented out in `py.lisp` because it conflicts with the Common Lisp `assert` macro if used without qualification.
```lisp
;; Lisp Input (Function call fallback)
(assert (== x y))
```
```python
# Emitted Python
assert(x == y)
```

### Proposed Form: `assert_`
Provide a shadow-safe dedicated assertion form that formats as a statement.
```lisp
;; Proposed Lisp Input
(assert_ (== x y) (string "x and y must be equal"))
```
```python
# Expected Python Output
assert x == y, "x and y must be equal"
```
