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

