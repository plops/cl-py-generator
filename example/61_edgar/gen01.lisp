(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)
;; https://pypi.org/project/python-edgar/
;; python3 -m pip install --user  git+https://github.com/edgarminers/python-edgar

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/61_edgar")
  (defparameter *code-file* "run_01_read_edgar")
  (defparameter *source* (format nil "~a/source/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
		   (format (- (time.time) start_time)
			   ,@rest)))))
  (let* (
	 (code
	   `(do0
	     (imports (			;os
					;sys
					;time
					;docopt
					;pathlib
					;(np numpy)
					;serial
		       (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
		       scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
		       (np numpy)
					; scipy.sparse
					;scipy.sparse.linalg
					; jax
					;jax.random
					;jax.config
					;copy
		       subprocess
		       threading
					;datetime
					;time
		       ; mss
		       ;cv2
		       time
		       edgar
		       
		       ))
	     #+nil (imports-from (selenium webdriver)
			   (selenium.webdriver.common.keys Keys)
			   )
	     
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"))
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
	     (setf start_time (time.time)
		   debug True)

	     (setf df (pd.read_csv (string "data/2021-QTR3.tsv")
				   :sep (string "\\t")
				   :lineterminator (string "\\n")
				   :names None))
	     df
	     )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

