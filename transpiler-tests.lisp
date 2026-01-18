;;;; transpiler-tests.lisp

(ql:quickload :cl-py-generator)
(ql:quickload "uiop")
(ql:quickload "cl-ppcre")

(defpackage :cl-py-generator/tests
  (:use :cl :cl-py-generator))

(in-package :cl-py-generator/tests)

;; NOTE: The :lisp forms are now correctly quoted with ' instead of `
(defparameter *test-cases*
  '(;; Test for simple addition. Note the expected python is simple.
    ;; ruff will add the parentheses automatically for us.
    (:name "simple-addition"
     :description "Tests the '+' operator with two integer arguments."
     :lisp (+ 1 2)
     :python "((1) + (2))" ; <-- Ruff will format this and the output to be identical
     :tags '(:operator :arithmetic))

    (:name "function-definition"
     :description "Tests a simple function definition with one argument and a return statement."
     :lisp (def foo (x) (return x))
     :python "def foo(x):
    return x"
     :tags '(:core :control-flow))

    (:name "setf-multiple"
     :description "Tests 'setf' for assigning multiple variables in sequence."
     :lisp (setf a 1 b 2)
     :python "a = 1
b = 2"
     :tags '(:core :assignment))

    (:name "list-literal"
     :description "Tests list literal emission."
     :lisp (list 1 2)
     :python "[1, 2]"
     :tags '(:core :collection))

    (:name "tuple-literal"
     :description "Tests tuple literal emission."
     :lisp (tuple 1 2 3)
     :python "(1, 2, 3,)"
     :tags '(:core :collection))

    (:name "paren-literal"
     :description "Tests paren emission for comma-separated values."
     :lisp (paren a b)
     :python "(a, b)"
     :tags '(:core :collection))

    (:name "ntuple-literal"
     :description "Tests ntuple emission without surrounding parentheses."
     :lisp (ntuple a b c)
     :python "a, b, c"
     :tags '(:core :collection))

    (:name "curly-literal"
     :description "Tests curly emission for set literals."
     :lisp (curly 1 2 3)
     :python "{1, 2, 3}"
     :tags '(:core :collection))

    (:name "dict-literal"
     :description "Tests dict literal emission with explicit key/value pairs."
     :lisp (dict ((string "a") 1) ((string "b") 2))
     :python "{(\"a\"): (1), (\"b\"): (2)}"
     :tags '(:core :collection))

    (:name "dictionary-constructor"
     :description "Tests keyword-based dictionary constructor emission."
     :lisp (dictionary :a 1 :b 2)
     :python "dict(a=1, b=2)"
     :tags '(:core :collection))

    (:name "incf-basic"
     :description "Tests incf emission with explicit increment."
     :lisp (incf a 2)
     :python "a += 2"
     :tags '(:operator :assignment))

    (:name "decf-basic"
     :description "Tests decf emission with explicit decrement."
     :lisp (decf a 3)
     :python "a -= 3"
     :tags '(:operator :assignment))

    (:name "aref-index"
     :description "Tests array reference emission."
     :lisp (aref arr 1)
     :python "arr[1]"
     :tags '(:core :indexing))

    (:name "slice-index"
     :description "Tests slice emission inside indexing."
     :lisp (aref arr (slice 1 2))
     :python "arr[1:2]"
     :tags '(:core :indexing))

    (:name "slice-open-start"
     :description "Tests slice emission with an open start."
     :lisp (aref arr (slice nil 3))
     :python "arr[:3]"
     :tags '(:core :indexing))

    (:name "slice-open-end"
     :description "Tests slice emission with an open end."
     :lisp (aref arr (slice 1 nil))
     :python "arr[1:]"
     :tags '(:core :indexing))

    (:name "dot-access"
     :description "Tests dot form emission."
     :lisp (dot obj attr)
     :python "obj.attr"
     :tags '(:core :accessor))

    (:name "logical-and"
     :description "Tests logical and emission."
     :lisp (and a b)
     :python "((a) and (b))"
     :tags '(:operator :boolean))

    (:name "logical-or"
     :description "Tests logical or emission."
     :lisp (or a b)
     :python "((a) or (b))"
     :tags '(:operator :boolean))

    (:name "eq-compare"
     :description "Tests equality comparison emission."
     :lisp (== a b)
     :python "((a)==(b))"
     :tags '(:operator :comparison))

    (:name "neq-compare"
     :description "Tests inequality comparison emission."
     :lisp (!= a b)
     :python "((a)!=(b))"
     :tags '(:operator :comparison))

    (:name "lt-compare"
     :description "Tests less-than comparison emission."
     :lisp (< a b)
     :python "((a)<(b))"
     :tags '(:operator :comparison))

    (:name "lte-compare"
     :description "Tests less-than-or-equal comparison emission."
     :lisp (<= a b)
     :python "((a)<=(b))"
     :tags '(:operator :comparison))

    (:name "gt-compare"
     :description "Tests greater-than comparison emission."
     :lisp (> a b)
     :python "((a)>(b))"
     :tags '(:operator :comparison))

    (:name "gte-compare"
     :description "Tests greater-than-or-equal comparison emission."
     :lisp (>= a b)
     :python "((a)>=(b))"
     :tags '(:operator :comparison))

    (:name "in-compare"
     :description "Tests membership comparison emission."
     :lisp (in a b)
     :python "(a in b)"
     :tags '(:operator :comparison))

    (:name "not-in-compare"
     :description "Tests negative membership comparison emission."
     :lisp (not-in a b)
     :python "(a not in b)"
     :tags '(:operator :comparison))

    (:name "is-compare"
     :description "Tests identity comparison emission."
     :lisp (is a b)
     :python "(a is b)"
     :tags '(:operator :comparison))

    (:name "is-not-compare"
     :description "Tests negative identity comparison emission."
     :lisp (is-not a b)
     :python "(a is not b)"
     :tags '(:operator :comparison))

    (:name "mod-operator"
     :description "Tests modulo operator emission."
     :lisp (% a b)
     :python "((a)%(b))"
     :tags '(:operator :arithmetic))

    (:name "sub-operator"
     :description "Tests subtraction operator emission."
     :lisp (- a b)
     :python "((a)-(b))"
     :tags '(:operator :arithmetic))

    (:name "mul-operator"
     :description "Tests multiplication operator emission."
     :lisp (* a b c)
     :python "((a)*(b)*(c))"
     :tags '(:operator :arithmetic))

    (:name "div-operator"
     :description "Tests division operator emission."
     :lisp (/ a b)
     :python "((a)/(b))"
     :tags '(:operator :arithmetic))

    (:name "matmul-operator"
     :description "Tests matrix-multiplication operator emission."
     :lisp (@ a b)
     :python "((a)@(b))"
     :tags '(:operator :arithmetic))

    (:name "floor-div-operator"
     :description "Tests floor division operator emission."
     :lisp (// a b)
     :python "((a)//(b))"
     :tags '(:operator :arithmetic))

    (:name "pow-operator"
     :description "Tests exponentiation operator emission."
     :lisp (** a b)
     :python "((a)**(b))"
     :tags '(:operator :arithmetic))

    (:name "shift-left-operator"
     :description "Tests left shift operator emission."
     :lisp (<< a b)
     :python "((a)<<(b))"
     :tags '(:operator :bitwise))

    (:name "shift-right-operator"
     :description "Tests right shift operator emission."
     :lisp (>> a b)
     :python "((a)>>(b))"
     :tags '(:operator :bitwise))

    (:name "bitwise-and-operator"
     :description "Tests bitwise and operator emission."
     :lisp (& a b)
     :python "((a) & (b))"
     :tags '(:operator :bitwise))

    (:name "logand-operator"
     :description "Tests logand emission."
     :lisp (logand a b c)
     :python "((a) & (b) & (c))"
     :tags '(:operator :bitwise))

    (:name "bitwise-xor-operator"
     :description "Tests bitwise xor operator emission."
     :lisp (^ a b)
     :python "((a) ^ (b))"
     :tags '(:operator :bitwise))

    (:name "logxor-operator"
     :description "Tests logxor emission."
     :lisp (logxor a b)
     :python "((a) ^ (b))"
     :tags '(:operator :bitwise))

    (:name "bitwise-or-operator"
     :description "Tests bitwise or operator emission."
     :lisp (cl-py-generator::|\|| a b)
     :python "((a) | (b))"
     :tags '(:operator :bitwise))

    (:name "bitwise-ior-operator"
     :description "Tests bitwise or operator emission."
     :lisp (logior a b c)
     :python "((a) | (b) | (c))"
     :tags '(:operator :bitwise))

    (:name "string-literal"
     :description "Tests string literal emission."
     :lisp (string "hi")
     :python "\"hi\""
     :tags '(:core :string))

    (:name "string-bytes-literal"
     :description "Tests byte string literal emission."
     :lisp (string-b "data")
     :python "b\"data\""
     :tags '(:core :string))

    (:name "string3-literal"
     :description "Tests triple-quoted string literal emission."
     :lisp (string3 "block")
     :python "\"\"\"block\"\"\""
     :tags '(:core :string))

    (:name "rstring3-literal"
     :description "Tests raw triple-quoted string literal emission."
     :lisp (rstring3 "raw")
     :python "r\"\"\"raw\"\"\""
     :tags '(:core :string))

    (:name "fstring-literal"
     :description "Tests f-string emission."
     :lisp (fstring "{x}")
     :python "f\"{x}\""
     :tags '(:core :string))

    (:name "fstring3-literal"
     :description "Tests triple-quoted f-string emission."
     :lisp (fstring3 "{x}")
     :python "f\"\"\"{x}\"\"\""
     :tags '(:core :string))

    (:name "comment-literal"
     :description "Tests single line comment emission."
     :lisp (comment "note")
     :python "# note"
     :tags '(:core :comment))

    (:name "comments-literal"
     :description "Tests multi-line comment emission."
     :lisp (comments "line1" "line2")
     :python "# line1
# line2"
     :tags '(:core :comment))

    (:name "symbol-literal"
     :description "Tests symbol emission with hyphen to colon conversion."
     :lisp (symbol foo-bar)
     :python "foo:bar"
     :tags '(:core :symbol))

    (:name "lambda-basic"
     :description "Tests lambda emission with a single expression body."
     :lisp (lambda (x) (+ x 1))
     :python "lambda x: ((x)+(1))"
     :tags '(:core :lambda))

    (:name "if-else-basic"
     :description "Tests if/else emission."
     :lisp (if (== a b) (return 1) (return 2))
     :python "if ( ((a)==(b)) ):
    return 1
else:
    return 2"
     :tags '(:control-flow))

    (:name "when-basic"
     :description "Tests when emission."
     :lisp (when (> a b) (return a))
     :python "if ( ((a)>(b)) ):
    return a"
     :tags '(:control-flow))

    (:name "unless-basic"
     :description "Tests unless emission."
     :lisp (unless (> a b) (return b))
     :python "if ( not(((a)>(b))) ):
    return b"
     :tags '(:control-flow))

    (:name "while-basic"
     :description "Tests while loop emission."
     :lisp (while (< a b) (setf a (+ a 1)))
     :python "while (((a)<(b))):
    a=((a)+(1))"
     :tags '(:control-flow))

    (:name "for-basic"
     :description "Tests for loop emission."
     :lisp (for (i (range 3)) (print i))
     :python "for i in range(3):
    print(i)"
     :tags '(:control-flow))

    (:name "for-generator-basic"
     :description "Tests for-generator emission."
     :lisp (for-generator (i (range 3)) (* i 2))
     :python "((i)*(2)) for i in range(3)"
     :tags '(:control-flow))

    (:name "class-basic"
     :description "Tests class emission with a simple initializer."
     :lisp (class Foo () (def __init__ (self x) (setf (dot self x) x)))
     :python "class Foo:
    def __init__(self, x):
        self.x=x"
     :tags '(:core :class))

    (:name "import-basic"
     :description "Tests single import emission."
     :lisp (import sys)
     :python "import sys"
     :tags '(:import))

    (:name "import-alias"
     :description "Tests import emission with alias."
     :lisp (import (np numpy))
     :python "import numpy as np"
     :tags '(:import))

    (:name "imports-basic"
     :description "Tests multiple import emissions with aliases."
     :lisp (imports (sys (np numpy) (plt matplotlib.pyplot)))
     :python "import sys
import numpy as np
import matplotlib.pyplot as plt"
     :tags '(:import))

    (:name "import-from-basic"
     :description "Tests from-import emission."
     :lisp (import-from math sin cos)
     :python "from math import sin, cos"
     :tags '(:import))

    (:name "imports-from-basic"
     :description "Tests multiple from-import emissions."
     :lisp (imports-from (math sin cos) (pathlib Path))
     :python "from math import sin, cos
from pathlib import Path"
     :tags '(:import))

    (:name "with-basic"
     :description "Tests with-statement emission."
     :lisp (with (open \"f.txt\") (setf data (dot f read)))
     :python "with open(\"f.txt\"):
    data=f.read"
     :tags '(:control-flow))

    (:name "with-as-basic"
     :description "Tests with-statement emission using 'as'."
     :lisp (with (as (open \"f.txt\") f) (setf data (dot f read)))
     :python "with open(\"f.txt\") as f:
    data=f.read"
     :tags '(:control-flow))

    (:name "try-basic"
     :description "Tests try/except/else/finally emission."
     :lisp (try (setf a 1) (Exception (setf a 2)) (else (setf a 3)) (finally (setf a 4)))
     :python "try:
    a=1
except Exception:
    a=2
else:
    a=3
finally:
    a=4"
     :tags '(:control-flow))

    (:name "cond-basic"
     :description "Tests cond emission."
     :lisp (cond ((> a b) (return a)) ((< a b) (return b)) (t (return 0)))
     :python "if ( ((a)>(b)) ):
    return a
elif ( ((a)<(b)) ):
    return b
else:
    return 0"
     :tags '(:control-flow))

    (:name "ternary-basic"
     :description "Tests ternary emission."
     :lisp (? (> a b) a b)
     :python "(a) if (((a)>(b))) else (b)"
     :tags '(:control-flow))

    (:name "return_-basic"
     :description "Tests return_ emission inside a function."
     :lisp (def foo () (return_ (x)))
     :python "def foo():
    return x"
     :tags '(:control-flow))

    (:name "space-basic"
     :description "Tests space emission."
     :lisp (space alpha beta)
     :python "alpha beta"
     :tags '(:core :utility))

    (:name "functional-addition"
     :description "Verifies that the generated code for '+' executes correctly."
     :lisp (print (+ 5 8))
     :python "print(((5) + (8)))"
     :exec-test t
     :expected-output "13"
     :tags '(:operator :arithmetic :functional))))

;; ===================================================================
;; NEW HELPER FUNCTION TO RUN RUFF
;; ===================================================================
(defun run-ruff-format (python-code-string)
  "Takes a string of Python code, writes it to a temp file with a .py extension,
   formats it with 'ruff format', and returns the formatted code.
   Returns the original string if ruff fails (e.g., syntax error)."
  ;; The :type "py" argument ensures the file is created with a .py extension.
  ;; All logic is now correctly placed *inside* this block, ensuring the file
  ;; exists when we operate on it.
  (uiop:with-temporary-file (:pathname p :stream s :type "py" :keep nil)
    ;; 1. Write the Python code to the temporary file.
    (write-string python-code-string s)
    
    ;; 2. IMPORTANT: Close the stream to ensure all content is flushed to disk
    ;;    and the file handle is released before ruff tries to access it.
    (finish-output s)
    (close s)

    ;; 3. Run ruff format on the file, which now definitely exists.
    (multiple-value-bind (output error-output exit-code)
        (uiop:run-program (list "ruff" "format" (uiop:native-namestring p))
                          :output :string
                          :error-output :string
                          :ignore-error-status t)
      (declare (ignore output))
      (if (= 0 exit-code)
          ;; 4. If successful, read the newly formatted content back from the file.
          (uiop:read-file-string p)
          
          ;; 5. If ruff fails (e.g., syntax error in generated code), warn the user
          ;;    and return the original, unformatted code for a clearer debugging diff.
          (progn
            (warn "Ruff format failed with exit code ~A. Stderr: ~A" exit-code error-output)
            python-code-string)))))


(defun normalize-string (s)
  "Removes leading/trailing whitespace and normalizes line endings to LF."
  (string-trim '(#\Space #\Newline #\Tab #\Return)
               (cl-ppcre:regex-replace-all "\\r\\n?" s "\n")))

;; ===================================================================
;; UPDATED TEST RUNNER
;; ===================================================================
(defun run-transpiler-tests (&key (tests *test-cases*) (tags nil))
  "Runs the defined test suite for the s-expression to Python transpiler.
   Normalizes both actual and expected Python code using 'ruff format'."
  (let ((passed 0)
        (failed 0)
        (test-count 0)
        (selected-tests (if tags
                            (remove-if-not (lambda (tc) (intersection tags (getf tc :tags))) tests)
                            tests)))

    (format t "~&Running ~D tests..." (length selected-tests))

    (dolist (test-case selected-tests)
      (incf test-count)
      (let* ((name (getf test-case :name))
             (lisp-code (getf test-case :lisp))
             (expected-python-raw (getf test-case :python))
             (actual-python-raw (emit-py :clear-env t :code lisp-code))
             (expected-python (run-ruff-format expected-python-raw))
             (actual-python (run-ruff-format actual-python-raw)))

        (format t "~&[~D] Testing '~A'... " test-count name)

        ;; TIER 1: Transpilation Correctness
        (if (string= (normalize-string actual-python) (normalize-string expected-python))
            (format t "TRANSPILATION [PASS]")
            (progn
              (incf failed)
              (format t "TRANSPILATION [FAIL]~%  Expected (after ruff):~%---~%~A~%---~%  Got (after ruff):~%---~%~A~%---"
                      expected-python actual-python)
              ;; Skip to the next test case in the dolist loop
              (return)))

        ;; TIER 2: Functional Execution Correctness
        ;; The entire functional test logic is now correctly wrapped.
        (if (getf test-case :exec-test)
            ;; THEN branch: Run the execution test
            (uiop:with-temporary-file (:pathname p :stream s :type "py")
              (write-string actual-python s)
              (finish-output s)
              (close s)
              (multiple-value-bind (output error-output exit-code)
                  (uiop:run-program (list "python3" (uiop:native-namestring p))
                                    :output :string :error-output :string :ignore-error-status t)
                (if (and (= exit-code 0)
                         (string= (normalize-string output) (normalize-string (getf test-case :expected-output))))
                    (progn (incf passed) (format t ", EXECUTION [PASS]"))
                    (progn
                      (incf failed)
                      (format t ", EXECUTION [FAIL]~%  Exit Code: ~D~%  Expected Output: ~S~%  Actual Output:   ~S~%  Stderr: ~S"
                              exit-code (getf test-case :expected-output) output error-output)))))
            ;; ELSE branch: No execution test, just count the transpilation pass
            (incf passed))))
    
    (format t "~2&--- Test Summary ---~%")
    (format t "Total Tests Run: ~D~%" test-count)
    (format t "Assertions Passed: ~D~%" passed)
    (format t "Assertions Failed: ~D~%" failed)
    (format t "--------------------~%")
    (unless (= 0 failed) (uiop:quit 1))))


(defun generate-documentation (&key (tests *test-cases*) (output-file "SUPPORTED_FORMS.md"))
  "Generates a markdown documentation file from the test cases."
  (with-open-file (s output-file :direction :output :if-exists :supersede)
    (format s "# Supported S-Expression Forms~%")
    (format s "This documentation is auto-generated from the test suite. Do not edit manually.~2%")

    (let ((tagged-tests (make-hash-table :test 'equal)))
      ;; Group tests by their primary tag for better organization
      (dolist (test-case tests)
        (let ((primary-tag (first (getf test-case :tags))))
          (push test-case (gethash primary-tag tagged-tests))))
      
      (loop for tag in (sort (alexandria:hash-table-keys tagged-tests) #'string<)
            do
               (format s "## ~(~a~) Forms~%" tag)
               (dolist (test-case (reverse (gethash tag tagged-tests)))
                 ;; Run ruff format here as well to ensure docs are clean
                 (let ((formatted-python (run-ruff-format (getf test-case :python))))
                   (format s "### `~S`~%" (getf test-case :lisp))
                   (format s "~A~2%" (getf test-case :description))
                   (format s "**Lisp S-Expression:**~%```lisp~%~S~%```~2%" (getf test-case :lisp))
                   (format s "**Generated Python (after formatting):**~%```python~%~A~%```~2%" formatted-python)))))))



;; This confirmation message now correctly uses the output-file variable
;; which is still in scope from the function's parameter list.
#+nil
(format t "~&Documentation successfully written to '~A'~%" (uiop:native-namestring (merge-pathnames output-file)))
  ;; Return the pathname for programmatic use
#+nil
(merge-pathnames output-file)
