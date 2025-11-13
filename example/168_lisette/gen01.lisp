(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(progn
  (defparameter *source* "example/168_lisette/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p01_client"
						   *source*))
   `(do0
     (comments "run with:  export ANTHROPIC_API_KEY=`cat ~/anthropic.key`; uv run python -i p01_client.py")
     (do0 (imports-from (__future__ annotations)
			(lisette *)) ;; 163 MB
	  #+nil (imports (;os
		    ;(pd pandas)
		    ))
	  ))
   ))


