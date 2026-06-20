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
     ;; SX symbolics: Scalar-based expression graphs, fast and lightweight for low-level math.
     (setf x (SX.sym (string "x"))     ;; 1-by-1 scalar symbolic primitive
	   y (SX.sym (string "y") 5)   ;; 5-by-1 vector symbolic primitive
	   z (SX.sym (string "z") 4 2) ;; 4-by-2 matrix symbolic primitive
	   )
     (setf f (+ (** x 2)
		10)
	   f (sqrt f))

     (setf g (+ (* 3 z) x))
     
     
     ;; Creating constant SX matrices (structural vs. actual zeros)
     (setf B1 (SX.zeros 4 5) ;; Dense 4-by-5 matrix of actual zeros
	   B2 (SX 4 5)	     ;; Sparse 4-by-5 matrix of structural zeros
	   B3 (SX.eye 4)     ;; Identity matrix (sparse diagonal)
	   )
     (setf v (SX (list 1  2 3)))	;; Column vector from literal list
     (setf M (SX (list (list 1 2)       ;; Dense matrix from nested literal lists
		       (list 2 3)
		       (list 4 5))))
     ;; DM (Data Matrix): Used to store concrete numerical values for function inputs/outputs.
     ;; Not intended for general-purpose numerical linear algebra; use NumPy instead for calculations.

     (setf C (DM 2 3)
	   C_dense (C.full)
	   C_dense2 (np.array C)
	   C_sparse (C.sparse))

     ;; MX symbolics: Matrix-based expression graphs, where nodes are matrix operations rather than scalar operations.
     ;; More general than SX and more memory/computationally efficient for large-scale block-structured matrix operations.
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


