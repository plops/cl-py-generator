(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(progn
  (defparameter *source* "example/171_casadi/")
  (defun sym (name vals)
    `(SX.sym (string ,name)
	     ,@vals))
  (defun sym1 (name)
    (sym name (list (format nil "n~a" name))))
  (defun fun (name-args-vals)
    (destructuring-bind (&key name args vals) name-args-vals
     `(Function (string ,name)
		(list ,@args)
		(list ,@vals))))
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p05_rootfinder"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  #+nil (imports ((np numpy)
			  (plt matplotlib.pyplot)))
	  )
     (setf z ,(sym1 "z")
	   x (sym1 "x")
	   g0 (sin (+ x z))
	   g1 (cos (- x z))
	   g ,(fun `(:name g :args (z x) :vals (g0 g1)))
	   G (rootfinder (string "G")
			 (string "newton")
			 g))
     
     )))
