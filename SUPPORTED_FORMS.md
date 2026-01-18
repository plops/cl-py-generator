# Supported S-Expression Forms
This documentation is auto-generated from the test suite. Do not edit manually.

## quote Forms
### `(+ 1 2)`
Tests the '+' operator with two integer arguments.

**Lisp S-Expression:**
```lisp
(+ 1 2)
```

**Generated Python (after formatting):**
```python
((1) + (2))

```

### `(cl-py-generator:def cl-py-generator/tests::foo (cl-py-generator/tests::x)
      (return cl-py-generator/tests::x))`
Tests a simple function definition with one argument and a return statement.

**Lisp S-Expression:**
```lisp
(cl-py-generator:def cl-py-generator/tests::foo (cl-py-generator/tests::x)
 (return cl-py-generator/tests::x))
```

**Generated Python (after formatting):**
```python
def foo(x):
    return x

```

### `(setf cl-py-generator/tests::a 1
           cl-py-generator/tests::b 2)`
Tests 'setf' for assigning multiple variables in sequence.

**Lisp S-Expression:**
```lisp
(setf cl-py-generator/tests::a 1
      cl-py-generator/tests::b 2)
```

**Generated Python (after formatting):**
```python
a = 1
b = 2

```

### `(list 1 2)`
Tests list literal emission.

**Lisp S-Expression:**
```lisp
(list 1 2)
```

**Generated Python (after formatting):**
```python
[1, 2]

```

### `(cl-py-generator:tuple 1 2 3)`
Tests tuple literal emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:tuple 1 2 3)
```

**Generated Python (after formatting):**
```python
(
    1,
    2,
    3,
)

```

### `(cl-py-generator:paren cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests paren emission for comma-separated values.

**Lisp S-Expression:**
```lisp
(cl-py-generator:paren cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
(a, b)

```

### `(cl-py-generator:ntuple cl-py-generator/tests::a cl-py-generator/tests::b
      cl-py-generator/tests::c)`
Tests ntuple emission without surrounding parentheses.

**Lisp S-Expression:**
```lisp
(cl-py-generator:ntuple cl-py-generator/tests::a cl-py-generator/tests::b
 cl-py-generator/tests::c)
```

**Generated Python (after formatting):**
```python
a, b, c

```

### `(cl-py-generator:curly 1 2 3)`
Tests curly emission for set literals.

**Lisp S-Expression:**
```lisp
(cl-py-generator:curly 1 2 3)
```

**Generated Python (after formatting):**
```python
{1, 2, 3}

```

### `(cl-py-generator:dict ((string "a") 1) ((string "b") 2))`
Tests dict literal emission with explicit key/value pairs.

**Lisp S-Expression:**
```lisp
(cl-py-generator:dict ((string "a") 1) ((string "b") 2))
```

**Generated Python (after formatting):**
```python
{("a"): (1), ("b"): (2)}

```

### `(cl-py-generator:dictionary :a 1 :b 2)`
Tests keyword-based dictionary constructor emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:dictionary :a 1 :b 2)
```

**Generated Python (after formatting):**
```python
dict(a=1, b=2)

```

### `(incf cl-py-generator/tests::a 2)`
Tests incf emission with explicit increment.

**Lisp S-Expression:**
```lisp
(incf cl-py-generator/tests::a 2)
```

**Generated Python (after formatting):**
```python
a += 2

```

### `(decf cl-py-generator/tests::a 3)`
Tests decf emission with explicit decrement.

**Lisp S-Expression:**
```lisp
(decf cl-py-generator/tests::a 3)
```

**Generated Python (after formatting):**
```python
a -= 3

```

### `(aref cl-py-generator/tests::arr 1)`
Tests array reference emission.

**Lisp S-Expression:**
```lisp
(aref cl-py-generator/tests::arr 1)
```

**Generated Python (after formatting):**
```python
arr[1]

```

### `(aref cl-py-generator/tests::arr (cl-py-generator:slice 1 2))`
Tests slice emission inside indexing.

**Lisp S-Expression:**
```lisp
(aref cl-py-generator/tests::arr (cl-py-generator:slice 1 2))
```

**Generated Python (after formatting):**
```python
arr[1:2]

```

### `(cl-py-generator:dot cl-py-generator/tests::obj
      cl-py-generator/tests::attr)`
Tests dot form emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:dot cl-py-generator/tests::obj cl-py-generator/tests::attr)
```

**Generated Python (after formatting):**
```python
obj.attr

```

### `(and cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests logical and emission.

**Lisp S-Expression:**
```lisp
(and cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) and (b))

```

### `(or cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests logical or emission.

**Lisp S-Expression:**
```lisp
(or cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) or (b))

```

### `(cl-py-generator:== cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests equality comparison emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:== cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) == (b))

```

### `(cl-py-generator:!= cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests inequality comparison emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:!= cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) != (b))

```

### `(< cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests less-than comparison emission.

**Lisp S-Expression:**
```lisp
(< cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) < (b))

```

### `(<= cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests less-than-or-equal comparison emission.

**Lisp S-Expression:**
```lisp
(<= cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) <= (b))

```

### `(> cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests greater-than comparison emission.

**Lisp S-Expression:**
```lisp
(> cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) > (b))

```

### `(>= cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests greater-than-or-equal comparison emission.

**Lisp S-Expression:**
```lisp
(>= cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) >= (b))

```

### `(cl-py-generator:in cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests membership comparison emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:in cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
(a in b)

```

### `(cl-py-generator:not-in cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests negative membership comparison emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:not-in cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
(a not in b)

```

### `(cl-py-generator:is cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests identity comparison emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:is cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
(a is b)

```

### `(cl-py-generator:is-not cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests negative identity comparison emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:is-not cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
(a is not b)

```

### `(cl-py-generator:% cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests modulo operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:% cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) % (b))

```

### `(- cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests subtraction operator emission.

**Lisp S-Expression:**
```lisp
(- cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) - (b))

```

### `(* cl-py-generator/tests::a cl-py-generator/tests::b
        cl-py-generator/tests::c)`
Tests multiplication operator emission.

**Lisp S-Expression:**
```lisp
(* cl-py-generator/tests::a cl-py-generator/tests::b cl-py-generator/tests::c)
```

**Generated Python (after formatting):**
```python
((a) * (b) * (c))

```

### `(/ cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests division operator emission.

**Lisp S-Expression:**
```lisp
(/ cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) / (b))

```

### `(cl-py-generator:@ cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests matrix-multiplication operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:@ cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) @ (b))

```

### `(// cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests floor division operator emission.

**Lisp S-Expression:**
```lisp
(// cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) // (b))

```

### `(** cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests exponentiation operator emission.

**Lisp S-Expression:**
```lisp
(** cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) ** (b))

```

### `(cl-py-generator:<< cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests left shift operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:<< cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) << (b))

```

### `(cl-py-generator:>> cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests right shift operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:>> cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) >> (b))

```

### `(cl-py-generator:& cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests bitwise and operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:& cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) & (b))

```

### `(cl-py-generator:^ cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests bitwise xor operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:^ cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
((a) ^ (b))

```

### `(logior cl-py-generator/tests::a cl-py-generator/tests::b
             cl-py-generator/tests::c)`
Tests bitwise or operator emission.

**Lisp S-Expression:**
```lisp
(logior cl-py-generator/tests::a cl-py-generator/tests::b
        cl-py-generator/tests::c)
```

**Generated Python (after formatting):**
```python
((a) | (b) | (c))

```

### `(string "hi")`
Tests string literal emission.

**Lisp S-Expression:**
```lisp
(string "hi")
```

**Generated Python (after formatting):**
```python
"hi"

```

### `(cl-py-generator:string-b "data")`
Tests byte string literal emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:string-b "data")
```

**Generated Python (after formatting):**
```python
b"data"

```

### `(cl-py-generator:string3 "block")`
Tests triple-quoted string literal emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:string3 "block")
```

**Generated Python (after formatting):**
```python
"""block"""

```

### `(cl-py-generator:rstring3 "raw")`
Tests raw triple-quoted string literal emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:rstring3 "raw")
```

**Generated Python (after formatting):**
```python
r"""raw"""

```

### `(cl-py-generator:fstring "{x}")`
Tests f-string emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:fstring "{x}")
```

**Generated Python (after formatting):**
```python
f"{x}"

```

### `(cl-py-generator:fstring3 "{x}")`
Tests triple-quoted f-string emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:fstring3 "{x}")
```

**Generated Python (after formatting):**
```python
f"""{x}"""

```

### `(cl-py-generator:comment "note")`
Tests single line comment emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:comment "note")
```

**Generated Python (after formatting):**
```python
# note

```

### `(cl-py-generator:comments "line1" "line2")`
Tests multi-line comment emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:comments "line1" "line2")
```

**Generated Python (after formatting):**
```python
# line1
# line2

```

### `(symbol cl-py-generator/tests::foo-bar)`
Tests symbol emission with hyphen to colon conversion.

**Lisp S-Expression:**
```lisp
(symbol cl-py-generator/tests::foo-bar)
```

**Generated Python (after formatting):**
```python
foo: bar

```

### `(lambda (cl-py-generator/tests::x) (+ cl-py-generator/tests::x 1))`
Tests lambda emission with a single expression body.

**Lisp S-Expression:**
```lisp
(lambda (cl-py-generator/tests::x) (+ cl-py-generator/tests::x 1))
```

**Generated Python (after formatting):**
```python
lambda x: ((x) + (1))

```

### `(if (cl-py-generator:== cl-py-generator/tests::a cl-py-generator/tests::b)
         (return 1)
         (return 2))`
Tests if/else emission.

**Lisp S-Expression:**
```lisp
(if (cl-py-generator:== cl-py-generator/tests::a cl-py-generator/tests::b)
    (return 1)
    (return 2))
```

**Generated Python (after formatting):**
```python
if (a) == (b):
    return 1
else:
    return 2

```

### `(when (> cl-py-generator/tests::a cl-py-generator/tests::b)
       (return cl-py-generator/tests::a))`
Tests when emission.

**Lisp S-Expression:**
```lisp
(when (> cl-py-generator/tests::a cl-py-generator/tests::b)
  (return cl-py-generator/tests::a))
```

**Generated Python (after formatting):**
```python
if (a) > (b):
    return a

```

### `(unless (> cl-py-generator/tests::a cl-py-generator/tests::b)
       (return cl-py-generator/tests::b))`
Tests unless emission.

**Lisp S-Expression:**
```lisp
(unless (> cl-py-generator/tests::a cl-py-generator/tests::b)
  (return cl-py-generator/tests::b))
```

**Generated Python (after formatting):**
```python
if not ((a) > (b)):
    return b

```

### `(cl-py-generator:while
      (< cl-py-generator/tests::a cl-py-generator/tests::b)
      (setf cl-py-generator/tests::a (+ cl-py-generator/tests::a 1)))`
Tests while loop emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:while (< cl-py-generator/tests::a cl-py-generator/tests::b)
 (setf cl-py-generator/tests::a (+ cl-py-generator/tests::a 1)))
```

**Generated Python (after formatting):**
```python
while (a) < (b):
    a = (a) + (1)

```

### `(cl-py-generator:for
      (cl-py-generator/tests::i (cl-py-generator/tests::range 3))
      (print cl-py-generator/tests::i))`
Tests for loop emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:for
 (cl-py-generator/tests::i (cl-py-generator/tests::range 3))
 (print cl-py-generator/tests::i))
```

**Generated Python (after formatting):**
```python
for i in range(3):
    print(i)

```

### `(cl-py-generator:for-generator
      (cl-py-generator/tests::i (cl-py-generator/tests::range 3))
      (* cl-py-generator/tests::i 2))`
Tests for-generator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:for-generator
 (cl-py-generator/tests::i (cl-py-generator/tests::range 3))
 (* cl-py-generator/tests::i 2))
```

**Generated Python (after formatting):**
```python
((i)*(2)) for i in range(3)
```

### `(class cl-py-generator/tests::Foo nil
      (cl-py-generator:def cl-py-generator/tests::__init__
       (cl-py-generator/tests::self cl-py-generator/tests::x)
       (setf (cl-py-generator:dot cl-py-generator/tests::self
              cl-py-generator/tests::x)
               cl-py-generator/tests::x)))`
Tests class emission with a simple initializer.

**Lisp S-Expression:**
```lisp
(class cl-py-generator/tests::Foo nil
 (cl-py-generator:def cl-py-generator/tests::__init__
  (cl-py-generator/tests::self cl-py-generator/tests::x)
  (setf (cl-py-generator:dot cl-py-generator/tests::self
         cl-py-generator/tests::x)
          cl-py-generator/tests::x)))
```

**Generated Python (after formatting):**
```python
class Foo:
    def __init__(self, x):
        self.x = x

```

### `(import cl-py-generator/tests::sys)`
Tests single import emission.

**Lisp S-Expression:**
```lisp
(import cl-py-generator/tests::sys)
```

**Generated Python (after formatting):**
```python
import sys

```

### `(cl-py-generator:imports
      (cl-py-generator/tests::sys
       (cl-py-generator/tests::np cl-py-generator/tests::numpy)
       (cl-py-generator/tests::plt cl-py-generator/tests::matplotlib.pyplot)))`
Tests multiple import emissions with aliases.

**Lisp S-Expression:**
```lisp
(cl-py-generator:imports
 (cl-py-generator/tests::sys
  (cl-py-generator/tests::np cl-py-generator/tests::numpy)
  (cl-py-generator/tests::plt cl-py-generator/tests::matplotlib.pyplot)))
```

**Generated Python (after formatting):**
```python
import sys
import numpy as np
import matplotlib.pyplot as plt

```

### `(cl-py-generator:import-from cl-py-generator/tests::math sin cos)`
Tests from-import emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:import-from cl-py-generator/tests::math sin cos)
```

**Generated Python (after formatting):**
```python
from math import sin, cos

```

### `(cl-py-generator:imports-from (cl-py-generator/tests::math sin cos)
      (cl-py-generator/tests::pathlib cl-py-generator/tests::Path))`
Tests multiple from-import emissions.

**Lisp S-Expression:**
```lisp
(cl-py-generator:imports-from (cl-py-generator/tests::math sin cos)
 (cl-py-generator/tests::pathlib cl-py-generator/tests::Path))
```

**Generated Python (after formatting):**
```python
from math import sin, cos
from pathlib import Path

```

### `(cl-py-generator:with (open cl-py-generator/tests::|"F.TXT"|)
      (setf cl-py-generator/tests::data
              (cl-py-generator:dot cl-py-generator/tests::f read)))`
Tests with-statement emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:with (open cl-py-generator/tests::|"F.TXT"|)
 (setf cl-py-generator/tests::data
         (cl-py-generator:dot cl-py-generator/tests::f read)))
```

**Generated Python (after formatting):**
```python
with open("f.txt"):
    data = f.read

```

### `(cl-py-generator:try (setf cl-py-generator/tests::a 1)
      (cl-py-generator/tests::Exception (setf cl-py-generator/tests::a 2))
      (cl-py-generator:else (setf cl-py-generator/tests::a 3))
      (cl-py-generator:finally (setf cl-py-generator/tests::a 4)))`
Tests try/except/else/finally emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:try (setf cl-py-generator/tests::a 1)
 (cl-py-generator/tests::Exception (setf cl-py-generator/tests::a 2))
 (cl-py-generator:else (setf cl-py-generator/tests::a 3))
 (cl-py-generator:finally (setf cl-py-generator/tests::a 4)))
```

**Generated Python (after formatting):**
```python
try:
    a = 1
except Exception:
    a = 2
else:
    a = 3
finally:
    a = 4

```

### `(cond
      ((> cl-py-generator/tests::a cl-py-generator/tests::b)
       (return cl-py-generator/tests::a))
      ((< cl-py-generator/tests::a cl-py-generator/tests::b)
       (return cl-py-generator/tests::b))
      (t (return 0)))`
Tests cond emission.

**Lisp S-Expression:**
```lisp
(cond
 ((> cl-py-generator/tests::a cl-py-generator/tests::b)
  (return cl-py-generator/tests::a))
 ((< cl-py-generator/tests::a cl-py-generator/tests::b)
  (return cl-py-generator/tests::b))
 (t (return 0)))
```

**Generated Python (after formatting):**
```python
if (a) > (b):
    return a
elif (a) < (b):
    return b
else:
    return 0

```

### `(cl-py-generator:? (> cl-py-generator/tests::a cl-py-generator/tests::b)
      cl-py-generator/tests::a cl-py-generator/tests::b)`
Tests ternary emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator:? (> cl-py-generator/tests::a cl-py-generator/tests::b)
 cl-py-generator/tests::a cl-py-generator/tests::b)
```

**Generated Python (after formatting):**
```python
(a) if ((a) > (b)) else (b)

```

### `(cl-py-generator:def cl-py-generator/tests::foo nil
      (cl-py-generator:return_ (cl-py-generator/tests::x)))`
Tests return_ emission inside a function.

**Lisp S-Expression:**
```lisp
(cl-py-generator:def cl-py-generator/tests::foo nil
 (cl-py-generator:return_ (cl-py-generator/tests::x)))
```

**Generated Python (after formatting):**
```python
def foo():
    return x

```

### `(space cl-py-generator/tests::alpha cl-py-generator/tests::beta)`
Tests space emission.

**Lisp S-Expression:**
```lisp
(space cl-py-generator/tests::alpha cl-py-generator/tests::beta)
```

**Generated Python (after formatting):**
```python
alpha beta
```

### `(print (+ 5 8))`
Verifies that the generated code for '+' executes correctly.

**Lisp S-Expression:**
```lisp
(print (+ 5 8))
```

**Generated Python (after formatting):**
```python
print(((5) + (8)))

```

