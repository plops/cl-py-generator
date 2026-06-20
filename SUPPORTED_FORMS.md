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
1 + 2

```

### `(def foo (x) (return x))`
Tests a simple function definition with one argument and a return statement.

**Lisp S-Expression:**
```lisp
(def foo (x) (return x))
```

**Generated Python (after formatting):**
```python
def foo(x):
    return x

```

### `(setf a 1
           b 2)`
Tests 'setf' for assigning multiple variables in sequence.

**Lisp S-Expression:**
```lisp
(setf a 1
      b 2)
```

**Generated Python (after formatting):**
```python
a = 1
b = 2

```

### `(= a 1)`
Tests direct assignment emission.

**Lisp S-Expression:**
```lisp
(= a 1)
```

**Generated Python (after formatting):**
```python
a = 1

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

### `(tuple 1 2 3)`
Tests tuple literal emission.

**Lisp S-Expression:**
```lisp
(tuple 1 2 3)
```

**Generated Python (after formatting):**
```python
(
    1,
    2,
    3,
)

```

### `(paren a b)`
Tests paren emission for comma-separated values.

**Lisp S-Expression:**
```lisp
(paren a b)
```

**Generated Python (after formatting):**
```python
(a, b)

```

### `(ntuple a b c)`
Tests ntuple emission without surrounding parentheses.

**Lisp S-Expression:**
```lisp
(ntuple a b c)
```

**Generated Python (after formatting):**
```python
a, b, c

```

### `(curly 1 2 3)`
Tests curly emission for set literals.

**Lisp S-Expression:**
```lisp
(curly 1 2 3)
```

**Generated Python (after formatting):**
```python
{1, 2, 3}

```

### `(dict ((string "a") 1) ((string "b") 2))`
Tests dict literal emission with explicit key/value pairs.

**Lisp S-Expression:**
```lisp
(dict ((string "a") 1) ((string "b") 2))
```

**Generated Python (after formatting):**
```python
{("a"): (1), ("b"): (2)}

```

### `(dictionary :a 1 :b 2)`
Tests keyword-based dictionary constructor emission.

**Lisp S-Expression:**
```lisp
(dictionary :a 1 :b 2)
```

**Generated Python (after formatting):**
```python
dict(a=1, b=2)

```

### `(incf a 2)`
Tests incf emission with explicit increment.

**Lisp S-Expression:**
```lisp
(incf a 2)
```

**Generated Python (after formatting):**
```python
a += 2

```

### `(decf a 3)`
Tests decf emission with explicit decrement.

**Lisp S-Expression:**
```lisp
(decf a 3)
```

**Generated Python (after formatting):**
```python
a -= 3

```

### `(aref arr 1)`
Tests array reference emission.

**Lisp S-Expression:**
```lisp
(aref arr 1)
```

**Generated Python (after formatting):**
```python
arr[1]

```

### `(aref arr (slice 1 2))`
Tests slice emission inside indexing.

**Lisp S-Expression:**
```lisp
(aref arr (slice 1 2))
```

**Generated Python (after formatting):**
```python
arr[1:2]

```

### `(aref arr (slice 1 5 2))`
Tests slice emission with a step.

**Lisp S-Expression:**
```lisp
(aref arr (slice 1 5 2))
```

**Generated Python (after formatting):**
```python
arr[1:5:2]

```

### `(aref arr (slice nil 3))`
Tests slice emission with an open start.

**Lisp S-Expression:**
```lisp
(aref arr (slice nil 3))
```

**Generated Python (after formatting):**
```python
arr[:3]

```

### `(aref arr (slice 1 nil))`
Tests slice emission with an open end.

**Lisp S-Expression:**
```lisp
(aref arr (slice 1 nil))
```

**Generated Python (after formatting):**
```python
arr[1:]

```

### `(aref arr i j)`
Tests multi-index emission.

**Lisp S-Expression:**
```lisp
(aref arr i j)
```

**Generated Python (after formatting):**
```python
arr[i, j]

```

### `(dot obj attr)`
Tests dot form emission.

**Lisp S-Expression:**
```lisp
(dot obj attr)
```

**Generated Python (after formatting):**
```python
obj.attr

```

### `(try (setf a 1) ((as Exception e) (setf a 2)))`
Tests as-form emission in except clauses.

**Lisp S-Expression:**
```lisp
(try (setf a 1) ((as Exception e) (setf a 2)))
```

**Generated Python (after formatting):**
```python
try:
    a = 1
except Exception as e:
    a = 2

```

### `(and a b)`
Tests logical and emission.

**Lisp S-Expression:**
```lisp
(and a b)
```

**Generated Python (after formatting):**
```python
a and b

```

### `(or a b)`
Tests logical or emission.

**Lisp S-Expression:**
```lisp
(or a b)
```

**Generated Python (after formatting):**
```python
a or b

```

### `(== a b)`
Tests equality comparison emission.

**Lisp S-Expression:**
```lisp
(== a b)
```

**Generated Python (after formatting):**
```python
a == b

```

### `(!= a b)`
Tests inequality comparison emission.

**Lisp S-Expression:**
```lisp
(!= a b)
```

**Generated Python (after formatting):**
```python
a != b

```

### `(< a b)`
Tests less-than comparison emission.

**Lisp S-Expression:**
```lisp
(< a b)
```

**Generated Python (after formatting):**
```python
a < b

```

### `(<= a b)`
Tests less-than-or-equal comparison emission.

**Lisp S-Expression:**
```lisp
(<= a b)
```

**Generated Python (after formatting):**
```python
a <= b

```

### `(> a b)`
Tests greater-than comparison emission.

**Lisp S-Expression:**
```lisp
(> a b)
```

**Generated Python (after formatting):**
```python
a > b

```

### `(>= a b)`
Tests greater-than-or-equal comparison emission.

**Lisp S-Expression:**
```lisp
(>= a b)
```

**Generated Python (after formatting):**
```python
a >= b

```

### `(in a b)`
Tests membership comparison emission.

**Lisp S-Expression:**
```lisp
(in a b)
```

**Generated Python (after formatting):**
```python
(a in b)

```

### `(not-in a b)`
Tests negative membership comparison emission.

**Lisp S-Expression:**
```lisp
(not-in a b)
```

**Generated Python (after formatting):**
```python
(a not in b)

```

### `(is a b)`
Tests identity comparison emission.

**Lisp S-Expression:**
```lisp
(is a b)
```

**Generated Python (after formatting):**
```python
(a is b)

```

### `(is-not a b)`
Tests negative identity comparison emission.

**Lisp S-Expression:**
```lisp
(is-not a b)
```

**Generated Python (after formatting):**
```python
(a is not b)

```

### `(% a b)`
Tests modulo operator emission.

**Lisp S-Expression:**
```lisp
(% a b)
```

**Generated Python (after formatting):**
```python
a % b

```

### `(- a b)`
Tests subtraction operator emission.

**Lisp S-Expression:**
```lisp
(- a b)
```

**Generated Python (after formatting):**
```python
a - b

```

### `(* a b c)`
Tests multiplication operator emission.

**Lisp S-Expression:**
```lisp
(* a b c)
```

**Generated Python (after formatting):**
```python
a * b * c

```

### `(/ a b)`
Tests division operator emission.

**Lisp S-Expression:**
```lisp
(/ a b)
```

**Generated Python (after formatting):**
```python
a / b

```

### `(@ a b)`
Tests matrix-multiplication operator emission.

**Lisp S-Expression:**
```lisp
(@ a b)
```

**Generated Python (after formatting):**
```python
a @ b

```

### `(// a b)`
Tests floor division operator emission.

**Lisp S-Expression:**
```lisp
(// a b)
```

**Generated Python (after formatting):**
```python
a // b

```

### `(** a b)`
Tests exponentiation operator emission.

**Lisp S-Expression:**
```lisp
(** a b)
```

**Generated Python (after formatting):**
```python
a**b

```

### `(<< a b)`
Tests left shift operator emission.

**Lisp S-Expression:**
```lisp
(<< a b)
```

**Generated Python (after formatting):**
```python
a << b

```

### `(>> a b)`
Tests right shift operator emission.

**Lisp S-Expression:**
```lisp
(>> a b)
```

**Generated Python (after formatting):**
```python
a >> b

```

### `(& a b)`
Tests bitwise and operator emission.

**Lisp S-Expression:**
```lisp
(& a b)
```

**Generated Python (after formatting):**
```python
a & b

```

### `(logand a b c)`
Tests logand emission.

**Lisp S-Expression:**
```lisp
(logand a b c)
```

**Generated Python (after formatting):**
```python
a & b & c

```

### `(^ a b)`
Tests bitwise xor operator emission.

**Lisp S-Expression:**
```lisp
(^ a b)
```

**Generated Python (after formatting):**
```python
a ^ b

```

### `(logxor a b)`
Tests logxor emission.

**Lisp S-Expression:**
```lisp
(logxor a b)
```

**Generated Python (after formatting):**
```python
a ^ b

```

### `(cl-py-generator::|\|| a b)`
Tests bitwise or operator emission.

**Lisp S-Expression:**
```lisp
(cl-py-generator::|\|| a b)
```

**Generated Python (after formatting):**
```python
a | b

```

### `(logior a b c)`
Tests bitwise or operator emission.

**Lisp S-Expression:**
```lisp
(logior a b c)
```

**Generated Python (after formatting):**
```python
a | b | c

```

### `(foo 1 :bar 2 :baz 3)`
Tests keyword argument emission in function calls.

**Lisp S-Expression:**
```lisp
(foo 1 :bar 2 :baz 3)
```

**Generated Python (after formatting):**
```python
foo(1, bar=2, baz=3)

```

### `(foo :bar 2)`
Tests keyword-only call emission.

**Lisp S-Expression:**
```lisp
(foo :bar 2)
```

**Generated Python (after formatting):**
```python
foo(bar=2)

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

### `(string-b "data")`
Tests byte string literal emission.

**Lisp S-Expression:**
```lisp
(string-b "data")
```

**Generated Python (after formatting):**
```python
b"data"

```

### `(string3 "block")`
Tests triple-quoted string literal emission.

**Lisp S-Expression:**
```lisp
(string3 "block")
```

**Generated Python (after formatting):**
```python
"""block"""

```

### `(rstring3 "raw")`
Tests raw triple-quoted string literal emission.

**Lisp S-Expression:**
```lisp
(rstring3 "raw")
```

**Generated Python (after formatting):**
```python
r"""raw"""

```

### `(fstring "{x}")`
Tests f-string emission.

**Lisp S-Expression:**
```lisp
(fstring "{x}")
```

**Generated Python (after formatting):**
```python
f"{x}"

```

### `(fstring3 "{x}")`
Tests triple-quoted f-string emission.

**Lisp S-Expression:**
```lisp
(fstring3 "{x}")
```

**Generated Python (after formatting):**
```python
f"""{x}"""

```

### `(comment "note")`
Tests single line comment emission.

**Lisp S-Expression:**
```lisp
(comment "note")
```

**Generated Python (after formatting):**
```python
# note

```

### `(comments "line1" "line2")`
Tests multi-line comment emission.

**Lisp S-Expression:**
```lisp
(comments "line1" "line2")
```

**Generated Python (after formatting):**
```python
# line1
# line2

```

### `(symbol foo-bar)`
Tests symbol emission with hyphen to colon conversion.

**Lisp S-Expression:**
```lisp
(symbol foo-bar)
```

**Generated Python (after formatting):**
```python
foo: bar

```

### `(lambda (x) (+ x 1))`
Tests lambda emission with a single expression body.

**Lisp S-Expression:**
```lisp
(lambda (x) (+ x 1))
```

**Generated Python (after formatting):**
```python
lambda x: x + 1

```

### `(if (== a b)
         (return 1)
         (return 2))`
Tests if/else emission.

**Lisp S-Expression:**
```lisp
(if (== a b)
    (return 1)
    (return 2))
```

**Generated Python (after formatting):**
```python
if a == b:
    return 1
else:
    return 2

```

### `(when (> a b) (return a))`
Tests when emission.

**Lisp S-Expression:**
```lisp
(when (> a b) (return a))
```

**Generated Python (after formatting):**
```python
if a > b:
    return a

```

### `(unless (> a b) (return b))`
Tests unless emission.

**Lisp S-Expression:**
```lisp
(unless (> a b) (return b))
```

**Generated Python (after formatting):**
```python
if not a > b:
    return b

```

### `(while (< a b) (setf a (+ a 1)))`
Tests while loop emission.

**Lisp S-Expression:**
```lisp
(while (< a b) (setf a (+ a 1)))
```

**Generated Python (after formatting):**
```python
while a < b:
    a = a + 1

```

### `(for (i (range 3)) (print i))`
Tests for loop emission.

**Lisp S-Expression:**
```lisp
(for (i (range 3)) (print i))
```

**Generated Python (after formatting):**
```python
for i in range(3):
    print(i)

```

### `(for-generator (i (range 3)) (* i 2))`
Tests for-generator emission.

**Lisp S-Expression:**
```lisp
(for-generator (i (range 3)) (* i 2))
```

**Generated Python (after formatting):**
```python
i*2 for i in range(3)
```

### `(class Foo nil (def __init__ (self x) (setf (dot self x) x)))`
Tests class emission with a simple initializer.

**Lisp S-Expression:**
```lisp
(class Foo nil (def __init__ (self x) (setf (dot self x) x)))
```

**Generated Python (after formatting):**
```python
class Foo:
    def __init__(self, x):
        self.x = x

```

### `(class Child (Base) (def __init__ (self) (return 1)))`
Tests class emission with a parent class.

**Lisp S-Expression:**
```lisp
(class Child (Base) (def __init__ (self) (return 1)))
```

**Generated Python (after formatting):**
```python
class Child(Base):
    def __init__(self):
        return 1

```

### `(import sys)`
Tests single import emission.

**Lisp S-Expression:**
```lisp
(import sys)
```

**Generated Python (after formatting):**
```python
import sys

```

### `(import (np numpy))`
Tests import emission with alias.

**Lisp S-Expression:**
```lisp
(import (np numpy))
```

**Generated Python (after formatting):**
```python
import numpy as np

```

### `(imports (sys (np numpy) (plt matplotlib.pyplot)))`
Tests multiple import emissions with aliases.

**Lisp S-Expression:**
```lisp
(imports (sys (np numpy) (plt matplotlib.pyplot)))
```

**Generated Python (after formatting):**
```python
import sys
import numpy as np
import matplotlib.pyplot as plt

```

### `(import-from math sin cos)`
Tests from-import emission.

**Lisp S-Expression:**
```lisp
(import-from math sin cos)
```

**Generated Python (after formatting):**
```python
from math import sin, cos

```

### `(imports-from (math sin cos) (pathlib Path))`
Tests multiple from-import emissions.

**Lisp S-Expression:**
```lisp
(imports-from (math sin cos) (pathlib Path))
```

**Generated Python (after formatting):**
```python
from math import sin, cos
from pathlib import Path

```

### `(with (open |"F.TXT"|) (setf data (dot f read)))`
Tests with-statement emission.

**Lisp S-Expression:**
```lisp
(with (open |"F.TXT"|) (setf data (dot f read)))
```

**Generated Python (after formatting):**
```python
with open("f.txt"):
    data = f.read

```

### `(with (as (open |"F.TXT"|) f) (setf data (dot f read)))`
Tests with-statement emission using 'as'.

**Lisp S-Expression:**
```lisp
(with (as (open |"F.TXT"|) f) (setf data (dot f read)))
```

**Generated Python (after formatting):**
```python
with open("f.txt") as f:
    data = f.read

```

### `(do0 (setf a 1) (setf b 2))`
Tests do0 emission without extra indentation.

**Lisp S-Expression:**
```lisp
(do0 (setf a 1) (setf b 2))
```

**Generated Python (after formatting):**
```python
a = 1
b = 2

```

### `(cell (setf a 1))`
Tests cell emission with export comment.

**Lisp S-Expression:**
```lisp
(cell (setf a 1))
```

**Generated Python (after formatting):**
```python
# export
a = 1

```

### `(try (setf a 1) (Exception (setf a 2)) (else (setf a 3))
      (finally (setf a 4)))`
Tests try/except/else/finally emission.

**Lisp S-Expression:**
```lisp
(try (setf a 1) (Exception (setf a 2)) (else (setf a 3)) (finally (setf a 4)))
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

### `(cond ((> a b) (return a)) ((< a b) (return b)) (t (return 0)))`
Tests cond emission.

**Lisp S-Expression:**
```lisp
(cond ((> a b) (return a)) ((< a b) (return b)) (t (return 0)))
```

**Generated Python (after formatting):**
```python
if a > b:
    return a
elif a < b:
    return b
else:
    return 0

```

### `(? (> a b) a b)`
Tests ternary emission.

**Lisp S-Expression:**
```lisp
(? (> a b) a b)
```

**Generated Python (after formatting):**
```python
a if a > b else b

```

### `(def foo nil (return_ (x)))`
Tests return_ emission inside a function.

**Lisp S-Expression:**
```lisp
(def foo nil (return_ (x)))
```

**Generated Python (after formatting):**
```python
def foo():
    return x

```

### `(space alpha beta)`
Tests space emission.

**Lisp S-Expression:**
```lisp
(space alpha beta)
```

**Generated Python (after formatting):**
```python
alpha beta
```

### `(setf (aref u2 0 0) v)`
Tests assigning to a specific index via aref inside setf.

**Lisp S-Expression:**
```lisp
(setf (aref u2 0 0) v)
```

**Generated Python (after formatting):**
```python
u2[0, 0] = v

```

### `(def simulate (E y &key (t_max 1.0))
      (declare (type float E)
               (type list y)
               (type float t_max)
               (values list))
      (return y))`
Tests function definitions with parameter and return type declarations.

**Lisp S-Expression:**
```lisp
(def simulate (E y &key (t_max 1.0))
 (declare (type float E)
          (type list y)
          (type float t_max)
          (values list))
 (return y))
```

**Generated Python (after formatting):**
```python
def simulate(E: float, y: list, t_max: float = 1.0) -> list:
    return y

```

### `(space async (def time_generator nil (return 1)))`
Tests async function definitions using space construct.

**Lisp S-Expression:**
```lisp
(space async (def time_generator nil (return 1)))
```

**Generated Python (after formatting):**
```python
async def time_generator():
    return 1

```

### `(do0 (@rt (string "/")) (def get (request) (return 1)))`
Tests function decorators using the symbol fallback starting with @.

**Lisp S-Expression:**
```lisp
(do0 (@rt (string "/")) (def get (request) (return 1)))
```

**Generated Python (after formatting):**
```python
@rt("/")
def get(request):
    return 1

```

### `(curly (for-generator ((ntuple i s) (enumerate chars)) (slice s (+ i 1))))`
Tests dictionary comprehension using curly, for-generator and slice.

**Lisp S-Expression:**
```lisp
(curly (for-generator ((ntuple i s) (enumerate chars)) (slice s (+ i 1))))
```

**Generated Python (after formatting):**
```python
{s: i + 1 for i, s in enumerate(chars)}

```

### `(list (for-generator (r responses) r))`
Tests list comprehension using list and for-generator.

**Lisp S-Expression:**
```lisp
(list (for-generator (r responses) r))
```

**Generated Python (after formatting):**
```python
[r for r in responses]

```

### `(try (setf a 1) ("Exception as e" (print e)))`
Tests try/except block using string directly for except clause.

**Lisp S-Expression:**
```lisp
(try (setf a 1) ("Exception as e" (print e)))
```

**Generated Python (after formatting):**
```python
try:
    a = 1
except Exception as e:
    print(e)

```

### `(func **tub.input)`
Tests keyword argument unpacking in function calls.

**Lisp S-Expression:**
```lisp
(func **tub.input)
```

**Generated Python (after formatting):**
```python
func(**tub.input)

```

### `(def foo (self *args **kwargs) (return 1))`
Tests function definitions with *args and **kwargs unpacking parameters.

**Lisp S-Expression:**
```lisp
(def foo (self *args **kwargs) (return 1))
```

**Generated Python (after formatting):**
```python
def foo(self, *args, **kwargs):
    return 1

```

### `(foo)`
Tests function/constructor call with no arguments.

**Lisp S-Expression:**
```lisp
(foo)
```

**Generated Python (after formatting):**
```python
foo()

```

### `(tuple 2.2 2.2d0)`
Tests single-float and double-float number representations.

**Lisp S-Expression:**
```lisp
(tuple 2.2 2.2d0)
```

**Generated Python (after formatting):**
```python
(
    2.2,
    2.2,
)

```

### `(print (+ 5 8))`
Verifies that the generated code for '+' executes correctly.

**Lisp S-Expression:**
```lisp
(print (+ 5 8))
```

**Generated Python (after formatting):**
```python
print(5 + 8)

```

### `(tuple (- x) (/ x))`
Tests unary minus and unary division operators.

**Lisp S-Expression:**
```lisp
(tuple (- x) (/ x))
```

**Generated Python (after formatting):**
```python
(
    -x,
    1.0 / x,
)

```

### `(~ x)`
Tests unary bitwise negation operator.

**Lisp S-Expression:**
```lisp
(~ x)
```

**Generated Python (after formatting):**
```python
~(x)

```

### `#C(1.0 2.0)`
Tests representation of complex numbers.

**Lisp S-Expression:**
```lisp
#C(1.0 2.0)
```

**Generated Python (after formatting):**
```python
1.0 + 1j * 2.0

```

### `(not a)`
Tests unary logical negation operator.

**Lisp S-Expression:**
```lisp
(not a)
```

**Generated Python (after formatting):**
```python
not a

```

### `("list" generator)`
Tests raw string in function call position.

**Lisp S-Expression:**
```lisp
("list" generator)
```

**Generated Python (after formatting):**
```python
list(generator)

```

### `(tuple (aref xf (slice)) (aref xf (slice "" max_len)))`
Tests empty slice and slice with raw strings.

**Lisp S-Expression:**
```lisp
(tuple (aref xf (slice)) (aref xf (slice "" max_len)))
```

**Generated Python (after formatting):**
```python
(
    xf[:],
    xf[:max_len],
)

```

### `(tuple (paren* * (+ a b)) (paren* + (+ a b)))`
Tests paren* construct for precedence-aware parentheses.

**Lisp S-Expression:**
```lisp
(tuple (paren* * (+ a b)) (paren* + (+ a b)))
```

**Generated Python (after formatting):**
```python
(
    (a + b),
    a + b,
)

```

### `(do0 "df = pd.read_csv('data.csv')" "@threaded" (def func nil (return 1)))`
Tests raw code insertion via bare strings at block level.

**Lisp S-Expression:**
```lisp
(do0 "df = pd.read_csv('data.csv')" "@threaded" (def func nil (return 1)))
```

**Generated Python (after formatting):**
```python
df = pd.read_csv("data.csv")


@threaded
def func():
    return 1

```

### `(do0
      (for (i (range 5))
       (if (== i 2)
           break
           continue)))`
Tests loop control statements break and continue.

**Lisp S-Expression:**
```lisp
(do0
 (for (i (range 5))
  (if (== i 2)
      break
      continue)))
```

**Generated Python (after formatting):**
```python
for i in range(5):
    if i == 2:
        break
    else:
        continue

```

### `(do0 yield (yield x))`
Tests yield statement and yield function call variants.

**Lisp S-Expression:**
```lisp
(do0 yield (yield x))
```

**Generated Python (after formatting):**
```python
yield
yield (x)

```

### `(tuple (lambda () 42) (lambda (x y) (+ x y)))`
Tests lambda functions with zero or multiple arguments.

**Lisp S-Expression:**
```lisp
(tuple (lambda () 42) (lambda (x y) (+ x y)))
```

**Generated Python (after formatting):**
```python
(
    lambda: 42,
    lambda x, y: x + y,
)

```

### `(tuple (super) (super ImageModel self))`
Tests calling superclass methods in Python.

**Lisp S-Expression:**
```lisp
(tuple (super) (super ImageModel self))
```

**Generated Python (after formatting):**
```python
(
    super(),
    super(ImageModel, self),
)

```

### `(dot model (aref weights i j) (item))`
Tests chained dot access including functions and array references.

**Lisp S-Expression:**
```lisp
(dot model (aref weights i j) (item))
```

**Generated Python (after formatting):**
```python
model.weights[i, j].item()

```

### `(setf (ntuple a b) (tuple 1 2))`
Tests assignment to unpacked values on the left hand side.

**Lisp S-Expression:**
```lisp
(setf (ntuple a b) (tuple 1 2))
```

**Generated Python (after formatting):**
```python
a, b = (
    1,
    2,
)

```

### `(with
      (ntuple (as (open (string "a.txt")) f) (as (open (string "b.txt")) g))
      (setf a 1))`
Tests with-statement with multiple context managers using ntuple.

**Lisp S-Expression:**
```lisp
(with (ntuple (as (open (string "a.txt")) f) (as (open (string "b.txt")) g))
 (setf a 1))
```

**Generated Python (after formatting):**
```python
with open("a.txt") as f, open("b.txt") as g:
    a = 1

```

### `(export (setf a 1))`
Tests 'export' comment cell construct.

**Lisp S-Expression:**
```lisp
(export (setf a 1))
```

**Generated Python (after formatting):**
```python
# |export
a = 1

```

### `(indent a)`
Tests standalone 'indent' formatting construct.

**Lisp S-Expression:**
```lisp
(indent a)
```

**Generated Python (after formatting):**
```python
a

```

### `(lambda (x &key (y 2)) (+ x y))`
Tests lambda expression with keyword arguments.

**Lisp S-Expression:**
```lisp
(lambda (x &key (y 2)) (+ x y))
```

**Generated Python (after formatting):**
```python
lambda x, y=2: x + y

```

### `(aref arr ":" 0)`
Tests raw string index insertion inside aref.

**Lisp S-Expression:**
```lisp
(aref arr ":" 0)
```

**Generated Python (after formatting):**
```python
arr[:, 0]

```

### `(with conn (setf a 1))`
Tests with-statement with a simple variable context manager.

**Lisp S-Expression:**
```lisp
(with conn (setf a 1))
```

**Generated Python (after formatting):**
```python
with conn:
    a = 1

```

### `(tuple (dict) (dictionary))`
Tests empty dict literal and empty dictionary constructor.

**Lisp S-Expression:**
```lisp
(tuple (dict) (dictionary))
```

**Generated Python (after formatting):**
```python
(
    {},
    dict(),
)

```

