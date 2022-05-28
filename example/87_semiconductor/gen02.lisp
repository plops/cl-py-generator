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
  (let ((nb-counter 2))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((r (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ()
	(gen `(run_fit)
	     `((r
		(cell
		 (comments "load dependencies and file")
		 (require gamlss)
		 (setf location
		       (read.csv
			(string "/home/martin/stage/cl-py-generator/example/87_semiconductor/source/dir87_gen01_location.csv")))))
	       (r
		(cell
		 (do0
		  (comments "look at statistics of localization with 10e3 photons")
		  (setf dx ($ (aref location
				    (== location$max_phot 10000)
				    "")
			      dx))
		  (setf fit (fitDist dx :k 2
				     :type (string "realAll")))
		  (comments "i tried 10 and 10e3 photons, the best distribution fit seems to be normal")


		  )
		 ))

	       (r
		(cell
		 (comments "explicitly fit normal distribution, and generate some diagnostic plots")
		 (setf mNO (histDist dx (string "NO")
					;:bins 30
					;:n.cyc 100
				     ))
		 (do0
		  (commeents "compare empirical cumulative distribution function with the cdf for the gaussian fit")
		  (plot (ecdf dx))
		  (setf xs (seq -4 4 .01))
		  (lines
		   xs
		   ((lambda (y)
		      (pNO y
			   :mu mNO$mu
			   :sigma  mNO$sigma)) xs)

		   :col (string "red")
		   :lwd 3))
		 ))
	       (r
		(cell
		 (comments "show worm plot")
		 (wp mNO)))
	       )))))
  )




