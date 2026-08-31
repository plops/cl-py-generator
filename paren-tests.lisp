;;;; paren-tests.lisp --- differential tests for the parenthesis elision
;;;;
;;;; `emit-py' has two modes: it either parenthesizes every operand
;;;; (`:omit-redundant-parentheses nil') or it drops the parentheses that the
;;;; operator precedence makes unnecessary (the default).  The first mode never
;;;; depends on precedence and is therefore used here as the oracle: every
;;;; expression is emitted twice, both variants are evaluated by python3 and the
;;;; values have to be identical.  A dropped or misplaced pair of parentheses
;;;; regroups the expression and changes the value.
;;;;
;;;; Three layers:
;;;;   1. unit tests for `effective-operator' (which operator does a form
;;;;      actually print?)
;;;;   2. a fixed list of regression expressions, one per bug that was found
;;;;   3. randomized expressions with a fixed seed
;;;;
;;;; Usage:  ./run-paren-tests.sh
;;;;         sbcl --noinform --disable-debugger --load paren-tests.lisp \
;;;;              --eval '(cl-py-generator/paren-tests::run-paren-tests :count 2000 :depth 4 :seed 7)' \
;;;;              --quit

(ql:quickload :cl-py-generator)
(ql:quickload "uiop")

(defpackage :cl-py-generator/paren-tests
  (:use :cl :cl-py-generator))

(in-package :cl-py-generator/paren-tests)

;;; ------------------------------------------------------------------
;;; 1. unit tests for effective-operator
;;; ------------------------------------------------------------------

(defparameter *effective-operator-cases*
  '(;; form                 expected operator
    ((- a)                  cl-py-generator::unary-)	; prints -a
    ((- a b)                -)				; prints a-b
    ((/ a)                  /)				; prints 1.0/a
    ((/ a b)                /)
    ((+ a)                  nil)			; prints a
    ((+ (* a b))            *)				; prints a*b
    ((or 255)               nil)			; prints 255
    ((in a b)               nil)			; prints (a in b)
    ((f a b)                nil)			; prints f(a, b)
    ((dot a b)              dot)
    ((paren a b)            paren)
    ((? a b c)              ?)
    ((lambda (x) x)         lambda)
    ((not a)                not)
    ((~ a)                  ~)
    (a                      nil)
    (1                      nil))
  "Test cases for `cl-py-generator::effective-operator'.  The printed operator
   is not always the head of the form; getting this wrong silently drops
   parentheses, which is what made (** (- a) 2) print -a**2.")

(defun run-effective-operator-tests ()
  "Return the number of failed checks."
  (let ((failed 0))
    (loop for (form expected) in *effective-operator-cases*
          do (let ((got (cl-py-generator::effective-operator form)))
               (unless (eq expected got)
                 (incf failed)
                 (format t "~&effective-operator ~s: expected ~s, got ~s~%"
                         form expected got))))
    (format t "~&effective-operator: ~d case~:p, ~d failed~%"
            (length *effective-operator-cases*) failed)
    failed))

;;; ------------------------------------------------------------------
;;; 2./3. the expressions that are evaluated by python
;;; ------------------------------------------------------------------

(defparameter *regression-expressions*
  '(;; ** binds tighter than unary minus: -a**2 is -(a**2)
    (** (- 2) 2)
    (** -2 2)
    (** (- (- 5 3)) 2)
    ;; python chains comparisons: a==b<c means (a==b) and (b<c)
    (== (< 1 2) (< 3 4))
    (== 1 (== 1 1))
    (< 3 (< 1 2))
    (!= (== 1 2) 0)
    (in (== 1 2) (list 0 1))
    ;; the shift operators are left associative and share one precedence level
    (<< 1 (>> 8 1))
    (>> 64 (<< 1 2))
    ;; * and @ share a precedence level but are not associative with each other
    (* 2 (// 9 4))
    (// 9 (* 2 2))
    ;; the conditional expression is right associative
    (? 1 (? 0 2 3) 4)
    (? 0 1 (? 1 2 3))
    ;; a bit operation below a shift binds looser
    (<< 1 (& 7 3))
    (& 1 (<< 1 3))
    ;; a form with two elements is not automatically a primary expression
    (* (- 3) (- 4))
    (% (- 7) 5)
    ;; indexing and attribute access of an expression
    (aref (+ (list 1 2) (list 3)) 2)
    (aref (? 1 (list 1 2) (list 3 4)) 1)
    (dot (- 7 3) (__str__)))
  "One expression per bug that the parenthesis elision used to have.  They are
   evaluated in both emitter modes, so they must not contain free variables.")

;;; A tiny deterministic linear congruential generator, so that a failing run
;;; can be reproduced by passing the same seed.
(defparameter *random-state-value* 42)

(defun next-random (limit)
  (setf *random-state-value*
        (mod (+ (* 1664525 *random-state-value*) 1013904223)
             (expt 2 32)))
  (mod (ash *random-state-value* -16) limit))

(defparameter *random-leaves* '(1 2 3 5 7 -1 -3))

(defparameter *random-forms*
  (list
   ;; Only forms that are total on integers are generated: no division by a
   ;; possibly zero expression, no ** with a large exponent (the values would
   ;; explode) and shifts only with a masked right operand, i.e. a small
   ;; non-negative one.  Everything else would make the test fail on a python
   ;; exception instead of on a missing parenthesis.
   (lambda (sub) `(+ ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(+ ,(funcall sub) ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(- ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(- ,(funcall sub)))
   (lambda (sub) `(* ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(& ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(logand ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(|\|| ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(logior ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(^ ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(logxor ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(~ ,(funcall sub)))
   (lambda (sub) `(and ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(or ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(not ,(funcall sub)))
   (lambda (sub) `(< ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(<= ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(> ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(>= ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(== ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(!= ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(< ,(funcall sub) ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(? ,(funcall sub) ,(funcall sub) ,(funcall sub)))
   (lambda (sub) `(<< ,(funcall sub) (& ,(funcall sub) 3)))
   (lambda (sub) `(>> ,(funcall sub) (& ,(funcall sub) 3)))
   (lambda (sub) `(// ,(funcall sub) 5))
   (lambda (sub) `(% ,(funcall sub) 5))
   (lambda (sub) `(** (& ,(funcall sub) 3) 2))))

(defun random-expression (depth)
  (if (or (zerop depth)
          (zerop (next-random 4)))
      (elt *random-leaves* (next-random (length *random-leaves*)))
      (funcall (elt *random-forms* (next-random (length *random-forms*)))
               (lambda () (random-expression (1- depth))))))

;;; ------------------------------------------------------------------
;;; the differential test itself
;;; ------------------------------------------------------------------

(defun emit-both-modes (form)
  "Return the sparingly and the fully parenthesized python code of FORM."
  (values (emit-py :clear-env t :code form :omit-redundant-parentheses t)
          (emit-py :clear-env t :code form :omit-redundant-parentheses nil)))

(defun write-comparison-program (stream forms)
  "Write a python program that evaluates every expression of FORMS in both
   emitter modes and prints a line for every mismatch.  Each expression is
   evaluated on its own, so that a syntax error does not hide the other
   results."
  (format stream "# generated by paren-tests.lisp~%")
  (format stream "import sys, warnings~%")
  ;; ~ on a bool is deprecated in recent python versions; the value is the same
  ;; in both modes, so the warning is only noise here
  (format stream "warnings.simplefilter('ignore')~%")
  (format stream "PAIRS = [~%")
  (loop for form in forms
        do (multiple-value-bind (omitted parenthesized) (emit-both-modes form)
             (loop for code in (list omitted parenthesized)
                   do (when (or (find #\' code) (find #\" code)
                                (find #\\ code) (find #\Newline code))
                        (error "the expression ~s does not fit into one python string: ~a"
                               form code)))
             (format stream "    ('~a', '~a'),~%" omitted parenthesized)))
  (format stream "]~%")
  (format stream "fails = 0~%")
  (format stream "def value(code):~%")
  (format stream "    try:~%")
  (format stream "        return repr(eval(code))~%")
  (format stream "    except Exception as e:~%")
  (format stream "        return 'EXCEPTION ' + type(e).__name__ + ': ' + str(e)~%")
  (format stream "for i, (omitted, parenthesized) in enumerate(PAIRS):~%")
  (format stream "    a, b = value(omitted), value(parenthesized)~%")
  (format stream "    if a != b:~%")
  (format stream "        fails += 1~%")
  (format stream "        print('MISMATCH', i)~%")
  (format stream "        print('  omitted      :', omitted, '=', a)~%")
  (format stream "        print('  parenthesized:', parenthesized, '=', b)~%")
  (format stream "print('checked', len(PAIRS), 'expressions,', fails, 'mismatches')~%")
  (format stream "sys.exit(1 if fails else 0)~%"))

(defun report-mismatching-forms (output forms)
  "Print the s-expression of every expression that PYTHON reported as a
   mismatch and return the number of those reports."
  (let ((mismatches 0))
    (dolist (line (uiop:split-string output :separator (list #\Newline)))
      (when (eql 0 (search "MISMATCH " line))
        (incf mismatches)
        (let ((index (parse-integer line :start (length "MISMATCH ")
                                         :junk-allowed t)))
          (when index
            (format t "~&  form ~d: ~s~%" index (elt forms index))))))
    mismatches))

(defun run-differential-test (forms &key (label "expressions"))
  "Emit every form of FORMS in both modes, let python3 compare the values and
   return the number of mismatches."
  (uiop:with-temporary-file (:pathname p :stream s :type "py" :keep nil)
    (write-comparison-program s forms)
    (finish-output s)
    (close s)
    (multiple-value-bind (output error-output code)
        (uiop:run-program (list "python3" (uiop:native-namestring p))
                          :output :string :error-output :string
                          :ignore-error-status t)
      (format t "~&~a: ~a" label output)
      (when (string/= "" error-output)
        (format t "~&stderr: ~a~%" error-output))
      (if (eql 0 code)
          0
          (max 1 (report-mismatching-forms output forms))))))

(defun run-paren-tests (&key (count 400) (depth 3) (seed 42))
  "Run all parenthesis tests.  COUNT random expressions of nesting DEPTH are
   generated from SEED.  Exits with status 1 when a check fails."
  (let ((failed 0))
    (incf failed (run-effective-operator-tests))
    (incf failed (run-differential-test *regression-expressions*
                                        :label "regression expressions"))
    (setf *random-state-value* seed)
    (let ((forms (loop repeat count collect (random-expression depth))))
      (incf failed (run-differential-test
                    forms
                    :label (format nil "~d random expressions (depth ~d, seed ~d)"
                                   count depth seed))))
    (format t "~&--- Parenthesis Test Summary ---~%")
    (format t "Failed checks: ~d~%" failed)
    (format t "--------------------------------~%")
    (unless (zerop failed)
      (uiop:quit 1))
    failed))
