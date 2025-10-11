(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:mbti
  (:use #:cl #:cl-py-generator)) 

(in-package #:mbti)

(progn
  (defparameter *source* "example/166_mbti")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p01_mbti"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations))
	  (imports ((pd pandas))))

     (comments "Data Source form here: https://personalitymax.com/personality-types/population-gender/"
	       "They write: To supplement our data, we have also turned to another well-known and authoritative study on gender differences and stereotypes. This normative study was conducted in 1996 by Allen Hammer and Wayne Mitchell, and is titled “The Distribution of Personality Types In General Population.” It surveyed 1267 adults on a number of different demographic factors.")
     (setf df (pd.reads_csv (string "personality_type.csv")))
     ))
)


