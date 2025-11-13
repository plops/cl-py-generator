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
     (comments "run with:  export ANTHROPIC_API_KEY=`cat ~/anthropic.key`; uv run python -i p01_client.py")
     (do0 (imports-from (__future__ annotations)
			(claudette *)) ;; 15MB
					;(imports ((pd pandas)))
	  )
     (print models)

     (comments "['claude-opus-4-1-20250805', 'claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-20250514', 'claude-3-opus-20240229', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022']")

     (comments "haiku is cheapest")
     (setf m (string "claude-haiku-4-5"))
     (setf c (Client m))
     (setf r
      (c (string "Hi there, I am jeremy.")))
     (print r)
     (comments "Message(id='msg_01QeZAHVGFTUUiW6DjCZ3umV', content=[TextBlock(citations=None, text='Hi Jeremy! Nice to meet you. How can I help you today?', type='text')], model='claude-haiku-4-5-20251001', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=In: 14; Out: 18; Cache create: 0; Cache read: 0; Total Tokens: 32; Search: 0)")
     )))


