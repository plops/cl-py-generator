(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_top"
						 "example/163_fasthtml_sse/"))
 `(do0
   
   (imports-from (fasthtml.common *))
       
   (imports-from
					
    (loguru logger)
    )
   
   
   (do0
    (logger.remove)
    (logger.add
     sys.stdout
     :format (string "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>")
     :colorize True
     :level (string "DEBUG")
					;:utc True
     ))

    (logger.info (string "Logger configured"))
))


