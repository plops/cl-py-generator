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
				  (merge-pathnames #P"p07_nlp_ipopt"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *)))
     ,@(loop for e in `(x y z)
		       collect
		       `(setf ,e (SX.sym (string ,e))))
     (setf nlp (dictionary :x (vertcat x y z)
			   :f (+ (** x 2)
				 (* 100 (** z 2)))
			   :g (+ z
				 (** (- 1 x) 2)
				 -y))
	   S (nlpsol (string "S")
		     (string "ipopt")
		     nlp)
	   r (S :x0 (list 2.5 3 .75)
		:lbg 0 :ubg 0)
	   x_opt (aref r (string "x")))
     (print (fstring "x_opt={x_opt}"))
     )))
