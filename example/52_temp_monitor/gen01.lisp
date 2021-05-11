(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/52_temp_monitor")
  (defparameter *code-file* "run_01_plot_temp")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 
		 #-nil(do0
		  
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
			;   scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
					;datetime
					
			   
			  ; jax
			   ;jax.random
					;jax.config
			   ;copy
			   subprocess
			   datetime
			   time
			   re
			   ))
		
		 (setf
		  _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/~a" *path*)
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

		  (def grep_nvda (fn)
		  (with (as (open fn
				  )
			    f)
			(setf ls
			      (f.readlines))
			(for (l ls)
			     (when (in (string "W / ") l)
			       (return l)))))
		  
		 ,(let ((l `(;sensors
			     ;smart
			     ;nvme
			     nvda)))
		     `(do0
		       ,@(loop for name in l
			       collect
			       `(do0
				 (setf fns (dot (pathlib.Path (string "data/source/"))
						(glob (string ,(format nil "*_~a" name)))))
				 (setf df (pd.DataFrame (dictionary :fn fns)))
				 (setf (aref df (string "ts"))
				       (dot df fn
					    (apply
					     (lambda (x)
					       (dot datetime
						    datetime
						    (strptime
						     x.stem
						     (string ,(format nil "%Y%m%d_%H%M_%S_~a" name))))))))
				 (setf (aref df (string "line"))
				       (dot df fn
					    (apply grep_nvda)))
				 (setf (aref df (list (string "str_gpu_fan")
						      (string "str_gpu_temp")
						      ;(string "gpu_pow")
						      ;(string "gpu_mem")
						      ))
				       (dot df line str
					    (extract (rstring3 "\\| (\\d*)% *(\\d*).*"))))
				 (setf (aref df (string "gpu_fan"))
				       (dot (aref df (string "str_gpu_fan"))
					    (astype float))
				       )
				 (setf (aref df (string "gpu_temp"))
				       (dot (aref df (string "str_gpu_temp"))
					    (astype float))
				       )
				 (setf df_ (df.set_index (string "ts")))
				 (df_.gpu_temp.plot)
				 (df_.gpu_fan.plot)
				 ))
		       ))

		
		 ;(print (grep_nvda (aref df.fn.iloc 0)))
		 
		 

		 ))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



