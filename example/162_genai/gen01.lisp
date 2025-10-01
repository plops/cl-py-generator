(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_gen"
						 "example/162_genai/"))
 `(do0
   #+nil
   (do0
    (imports (matplotlib))
    (matplotlib.use (string "qtagg"	; "webagg"
			    ))
    (imports ((plt matplotlib.pyplot)))
    (plt.ion)
    )

   (imports ((np numpy)
	     (pd pandas)
	     sys
	     os
	     ))
    
    
    
    (imports-from (sqlite_minutils *)
		  (loguru logger)
		  (google genai)
		  (google.genai types))

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

   (do0
    (setf client (genai.Client :api_key (os.getenv (string "GEMINI_API_KEY")))))
   ))


