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
				  (merge-pathnames #P"p02_fs_tools"
						   *source*))
   `(do0
     (comments "run with:  export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_fs_tools.py

docs: lisette.answer.ai")
     
     (do0 (imports-from (__future__ annotations)
			(fastcore.tools *) ;; 18MB
			(lisette *)) ;; 163 MB
	  #+nil (imports (	     ;os
					;(pd pandas)
			  ))
	  )
     (fc_tool_info)
     (setf model (string "gemini/gemini-2.5-flash")
	   chat (Chat model)
	   
	   r (chat (rstring3 "tell me about the tools")))
     (print r))
   ))


