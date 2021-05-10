(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/52_temp_monitor")
  (defparameter *code-file* "run_00_temp_monitor")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 
		 #+nil(do0
		  
		  (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		  (imports ((plt matplotlib.pyplot)
			  ;  (animation matplotlib.animation) 
                            ;(xrp xarray.plot)
			    ))
                  
		  (plt.ion)
					;(plt.ioff)
		  ;;(setf font (dict ((string size) (string 6))))
		  ;; (matplotlib.rc (string "font") **font)
		  )
		 (imports (		;os
			   ;sys
			   ;time
					;docopt
			   ;pathlib
			   ;(np numpy)
			   ;serial
			   ;(pd pandas)
			   ;(xr xarray)
			   ;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			   ;scipy.ndimage
			;   scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
					;datetime
			   ;(np numpy)
			   
			  ; jax
			   ;jax.random
					;jax.config
			   ;copy
			   subprocess
			   datetime
			   time
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
				     (- tz))
			     )))
		 (setf count 0)
		 (while True
			(setf count (+ count 1))
			(setf now (datetime.datetime.now)
			      nowstr (now.strftime (string "%Y%m%d_%H%M_%S")))
		  ,(let ((l `((sensors (/usr/bin/sensors))
			      (smart (sudo /usr/sbin/smartctl -xa /dev/nvme0))
			      (nvme (sudo /usr/sbin/nvme smart-log /dev/nvme0))
			      (nvda (/opt/bin/nvidia-smi))
			      (nvda2 (/opt/bin/nvidia-smi -q)))))
		     `(do0
		       ,@(loop for (name e) in l
			       collect
			      
			       `(with (as (open (dot (string ,(format nil "{}_~a" name))
						     (format nowstr))
						(string "w"))
					  f)
				      (subprocess.call (list ,@(loop for ee in e
									collect
								     `(string ,ee)))
						       :stdout f)))
		       (time.sleep 30)
		       (print count))))


		 ))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



