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
			  (np numpy)
					;(pd pandas)
			  ))
	  )
     ;; symbolic expression (meaning of SX?)
     (setf x (SX.sym (string "x"))     ;; scalar
	   y (SX.sym (string "y") 5)   ;; vector
	   z (SX.sym (string "z") 4 2) ;;matrix
	   )
     (setf f (+ (** x 2)
		10)
	   f (sqrt f))

     (setf g (+ (* 3 z) x))
     
     
     ;; constants
     (setf B1 (SX.zeros 4 5) ;; dense
	   B2 (SX 4 5)	     ;; sparse
	   B3 (SX.eye 4)     ;; sparse diagonal
	   )
     (setf v (SX (list 1  2 3)))	;; column vector
     (setf M (SX (list (list 1 2)       ;; dense matrix
		       (list 2 3)
		       (list 4 5))))
     ;; DM (data matrix ?) is to store numerical values as inputs and outputs of
     ;; functions (but not intended for calculations). use numpy in
     ;; python or eigen, ublas, mtl in C++ instead

     (setf C (DM 2 3)
	   C_dense (C.full)
	   C_dense2 (np.array C)
	   C_sparse (C.sparse))

     ;; matrix expression (MX)
     ;; can be more economical when working with operations that are naturally vector or matrix valued
     ;; they are also more general than SX
     (setf
      u (MX.sym (string "u") 2 2)
      u2 (MX 2 2)
      v (MX.sym (string "v"))
	   fu (+ (* 3 u) v))


     (setf (aref u2 0 0)
	   v)

     ;; SX ist besser fuer sequenz skalarer operationen (rechte seite
     ;; fuer DAE), MX ist besser fuer glue also z.b constraints of NLP
     ;; wo aufrufe fuer ODE integratoren enthalten sein koennten, die
     ;; eine zu grosse expression werden wuerden
     )
   ))


