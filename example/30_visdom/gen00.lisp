(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/30_visdom")
  (defparameter *code-file* "run_00_start")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 "# %% imports"
		 
		 (imports (		;os
			   ;sys
			   ;time
					;docopt
			   pathlib
			   (np numpy)
			   ;serial
			   (pd pandas)
			   ;(xr xarray)
			   ;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			   ;scipy.ndimage
			   ;scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
			   ;datetime
			   ;dask.distributed
					;(da dask.array)
					;PIL
					;libtiff
			   visdom
			   ))
		 


		 (setf
	       _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

		  _code_generation_time
		  (string ,(multiple-value-bind
				 (second minute hour date month year day-of-week dst-p tz)
			       (get-decoded-time)
			     (declare (ignorable dst-p))
		      (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			      hour
			      minute
			      second
			      (nth day-of-week *day-names*)
			      year
			      month
			      date
			      (- tz)))))

		 (do0
		  (setf vis (visdom.Visdom
			     ))
		  (setf trace (dict ,@(loop for (e f) in
					    `((x (list 1 2 3))
					      (y (list 4 5 6))
					      (mode (string "markers+lines")
						    )
					      (type (string "custom"))
					      (marker (dict ((string "color") (string "red"))
							    ((string "symbol") 104)
							    ((string "size") 10)))
					      (text (list (string "one")
							  (string "two")
							  (string "three")))
					      (name (string "first trace")))
					    collect
					    `((string ,e) ,f)))
			layout (dict ,@(loop for (e f) in
					     `((title (string "first plot"))

					       (xaxis (dict ((string "title") (string "x1"))))
					       (yaxis (dict ((string "title") (string "x2")))
						))
					    collect
					     `((string ,e) ,f)))
			)
		  (vis._send (dict ,@(loop for (e f) in
					     `((data (list trace))
					       (layout layout)
					       (win (string "mywin")))
					    collect
					     `((string ,e) ,f)))))
		 ))
 	   ))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

