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
