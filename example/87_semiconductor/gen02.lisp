(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-r-generator"))

(in-package :cl-r-generator)

(progn
  ;; the following code needs inverted readtable, otherwise symbols
  ;; and filenames may have the wrong case and everything breaks in
  ;; horrible ways
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/87_semiconductor")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*))
  (let ((show-counter 1))
    (defun show (name code &key width height)
      (prog1
	  `(do0
	    (png (string ,(format nil "~2,'0d_~a.png" show-counter name))
		 ,@(when width
		     `(:width ,width))
		 ,@(when height
		     `(:height ,height)))
	    ,code
	    (dev.off))
	(incf show-counter))))
  (write-source (format nil "~a/source/run02_fit" *path*)
		`(do0

		  (require gamlss)
		  (setf location
			(read.csv
			 (string "/home/martin/stage/cl-py-generator/example/87_semiconductor/source/dir87_gen01_location.csv")))
		  (setf dx ($ (aref location
				      (== location$max_phot 10000)
				      "")
			      dx))
		  (setf fit (fitDist dx :k 2
				     :type (string "realAll")))
		  )))




