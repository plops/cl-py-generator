(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(progn
  (defparameter *source* "example/171_casadi/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p01_learn"
						   *source*))
   `(do0
     
     
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  #+nil (imports (	     ;os
					;(pd pandas)
			  ))
	  )
     ;; symbolic variables
     (setf x (SX.sym (string "x"))     ;; scalar
	   y (SX.sym (string "y") 5)   ;; vector
	   z (SX.sym (string "y") 4 2) ;;matrix
	   )
     (setf f (+ (** x 2)
		10)
	   f (sqrt f))
     ;; constants
     (setf B1 (SX.zeros 4 5) ;; dense
	   B2 (SX 4 5) ;; sparse
	   B3 (SX.eye 4) ;; sparse diagonal
	   )
     (setf v (SX (list 1  2 3)))	;; column vector
     (setf M (SX (list (list 1 2)       ;; dense matrix
		       (list 2 3)
		       (list 4 5))))
     )
   ))


