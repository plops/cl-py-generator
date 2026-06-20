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

---

## 5. Global Statement (`global_` / `py-global`)

### Current Approach (Raw String Code Injection)
Currently, `global` statements are written using raw string injection:
```lisp
;; Lisp Input
"global f_num"
```
```python
# Emitted Python
global f_num
```

### Proposed Form: `global_`
Provide a dedicated statement form that prints `global` followed by space-separated variable names.
```lisp
;; Proposed Lisp Input
(global_ f_num)
```
```python
# Expected Python Output
global f_num
```

---

## 6. Nonlocal Statement (`nonlocal_` / `py-nonlocal`)

### Current Approach (Raw String Code Injection)
Currently, `nonlocal` statements are written using raw string injection:
```lisp
;; Lisp Input
"nonlocal x"
```
```python
# Emitted Python
nonlocal x
```

### Proposed Form: `nonlocal_`
Provide a dedicated statement form that prints `nonlocal` followed by space-separated variable names.
```lisp
;; Proposed Lisp Input
(nonlocal_ x)
```
```python
# Expected Python Output
nonlocal x
```

---

## 7. Asynchronous Function/Statements (`async-def` / `async_` and `await_` / `py-await`)

### Current Approach (Workaround using `space`)
Asynchronous functions and awaits currently rely on joining `async` and `await` with spaces as a prefix:
```lisp
;; Lisp Input
(space async (def post (comment)
               (return (space await (self.cli.request ...)))))
```
```python
# Emitted Python
async def post(comment):
    return await self.cli.request(...)
```

### Proposed Forms: `async-def` and `await_`
Introduce semantic forms for asynchronous functions and await expressions.
```lisp
;; Proposed Lisp Input
(async-def post (comment)
  (return (await_ (self.cli.request ...))))
```
```python
# Expected Python Output
async def post(comment):
    return await self.cli.request(...)
```

---

## 8. Delete Statement (`del_` / `py-del`)

### Current Approach (Fallback to Function Call)
Writing `(del x)` falls back to function-call syntax `del(x)`.
```lisp
;; Lisp Input
(del (aref _job_store uid))
```
```python
# Emitted Python
del(_job_store[uid])
```

### Proposed Form: `del_`
Provide a dedicated statement form that prints `del` followed by space-separated expressions.
```lisp
;; Proposed Lisp Input
(del_ (aref _job_store uid))
```
```python
# Expected Python Output
del _job_store[uid]
```

---

## 9. First-Class Class Attribute Type Annotations

### Current Approach (Raw String Code Injection)
Class variable type annotations (often required by `@dataclass` or schema definitions) are currently written as raw string literals.
```lisp
;; Lisp Input
(class GenerationConfig ()
  "prompt_text:str"
  (setf "model:str" (string "gemini-flash-latest")))
```
```python
# Emitted Python
class GenerationConfig:
    prompt_text: str
    model: str = "gemini-flash-latest"
```

### Proposed Form: `typed-var` or type declarations in class
Allow declaring typed attributes in classes natively.
```lisp
;; Proposed Lisp Input
(class GenerationConfig ()
  (typed-var prompt_text str)
  (setf (typed-var model str) (string "gemini-flash-latest")))
```
```python
# Expected Python Output
class GenerationConfig:
    prompt_text: str
    model: str = "gemini-flash-latest"
```

---

## 10. Yield Expressions (`yield_` / `py-yield` and `yield-from`)

### Current Approach (Fallback to Function Call)
Writing `(yield x)` falls back to function-call syntax `yield(x)`.
```lisp
;; Lisp Input
(yield (dictionary :type (string "thought") :text part.text))
```
```python
# Emitted Python
yield({"type": "thought", "text": part.text})
```

### Proposed Forms: `yield_` and `yield-from`
Support yield as a keyword/expression statement without parenthesizing its argument.
```lisp
;; Proposed Lisp Input
(yield_ (dictionary :type (string "thought") :text part.text))
(yield-from iterable)
```
```python
# Expected Python Output
yield {"type": "thought", "text": part.text}
yield from iterable
```
