					;(ql:quickload "optima")
					;(ql:quickload "alexandria")
(defpackage :cl-py-generator
  (:use :cl
					;:optima
	:alexandria)
  (:export
   #:emit-py
   #:tuple
   #:indent
   #:do
   #:do0
   #:def
   #:slice
   #:dot
   #:**
   #:imports
   #:try
   #:write-source
   #:run
   #:start-python))
