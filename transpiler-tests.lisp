(in-package :cl-py-generator)
(ql:quickload "uiop") ;; For running external programs

(defparameter *test-cases*
  '((:name "simple-addition"
     :description "Tests the '+' operator with two integer arguments."
     :lisp `(+ 1 2)
     :python "((1) + (2))"
     :tags '(:operator :arithmetic))

    (:name "function-definition"
     :description "Tests a simple function definition with one argument and a return statement."
     :lisp `(def foo (x) (return x))
     :python #.(format nil "def foo(x):~%    return x")
     :tags '(:core :control-flow))

    (:name "setf-multiple"
     :description "Tests 'setf' for assigning multiple variables in sequence."
     :lisp `(setf a 1 b 2)
     :python #.(format nil "a=1~%b=2")
     :tags '(:core :assignment))

    ;; A test case for functional testing (Tier 2)
    (:name "functional-addition"
     :description "Verifies that the generated code for '+' executes correctly."
     :lisp `(print (+ 5 8))
     :python "print(((5) + (8)))"
     :exec-test t
     :expected-output "13"
     :tags '(:operator :arithmetic :functional))
     
    ;; Add test cases for EVERY form: if, cond, dot, aref, slice, class, etc.
    ))

(defun normalize-string (s)
  "Removes leading/trailing whitespace and normalizes line endings."
  (string-trim '(#\Space #\Newline #\Tab #\Return)
               (cl-ppcre:regex-replace-all "\\r\\n?" s "\n")))

(defun run-transpiler-tests (&key (tests *test-cases*) (tags nil))
  "Runs the defined test suite for the s-expression to Python transpiler."
  (let ((passed 0)
        (failed 0)
        (test-count 0)
        (selected-tests (if tags
                            (remove-if-not (lambda (tc)
                                             (intersection tags (getf tc :tags)))
                                           tests)
                            tests)))

    (format t "~&Running ~D tests..." (length selected-tests))

    (dolist (test-case selected-tests)
      (incf test-count)
      (let* ((name (getf test-case :name))
             (lisp-code (getf test-case :lisp))
             (expected-python (getf test-case :python))
             (actual-python (emit-py :code lisp-code)))

        (format t "~&[~D] Testing '~A'... " test-count name)

        ;; TIER 1: Transpilation Correctness
        (if (string= (normalize-string actual-python)
                     (normalize-string expected-python))
          (progn
            (incf passed)
            (format t "TRANSPILATION [PASS]"))
          (progn
            (incf failed)
            (format t "TRANSPILATION [FAIL]~%  Expected: ~S~%  Got:      ~S"
                    expected-python actual-python)
            ;; Skip functional test if transpilation failed
            (return-from run-transpiler-tests)))

        ;; TIER 2: Functional Execution Correctness
        (when (getf test-case :exec-test)
          (let ((temp-py-file (format nil "/tmp/~A.py" (gensym "test-"))))
            (unwind-protect
                 (progn
                   (with-open-file (s temp-py-file :direction :output :if-exists :supersede)
                     (write-string actual-python s))
                   
                   (multiple-value-bind (output error-output exit-code)
                       (uiop:run-program (list "python3" temp-py-file)
                                         :output :string
                                         :error-output :string
                                         :ignore-error-status t)
                     (if (and (= exit-code 0)
                              (string= (normalize-string output)
                                       (normalize-string (getf test-case :expected-output))))
                       (progn
                         (incf passed)
                         (format t ", EXECUTION [PASS]"))
                       (progn
                         (incf failed)
                         (format t ", EXECUTION [FAIL]~%  Exit Code: ~D~%  Expected Output: ~S~%  Actual Output:   ~S~%  Stderr: ~S"
                                 exit-code (getf test-case :expected-output) output error-output)))))
              ;; Cleanup the temporary file
              (uiop:delete-file-if-exists temp-py-file))))
        ))
    
    (format t "~2&--- Test Summary ---~%")
    (format t "Total Tests: ~D~%" test-count)
    (format t "Passed: ~D~%" passed)
    (format t "Failed: ~D~%" failed)
    (format t "--------------------~%")
    (= 0 failed)))

;; To run the tests:
#+nil
(run-transpiler-tests)
