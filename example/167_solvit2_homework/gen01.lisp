(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(progn
  (defparameter *source* "example/167_solvit2_homework/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p01_client"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(claudette *))
	  ;(imports ((pd pandas)))
	  )
     (print models)
     )))


