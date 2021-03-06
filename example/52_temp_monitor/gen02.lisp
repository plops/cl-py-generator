(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/52_temp_monitor")
  (defparameter *code-file* "run_02_plot_temp")
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

		 (def parse_nvme (row)
		  (with (as (open row.fn
				  )
			    f)
			(setf ls
			      (f.readlines))
			(setf d (dictionary :ts row.ts))
			(for (l (aref ls "1:"))
			     (try
			      (do0
			       (setf (ntuple key val) (l.split (string ":"))
				     key (key.strip)
				     val (val.strip))
			       (setf (aref d key) val))
			      ("Exception as e"
			       pass)))
			(return d)))
		  
		 ,(let ((l `(;sensors
			     ;smart
			     nvme
			     ;nvda2
			     )))
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

				 (setf res (list))
				 (for ((ntuple idx row) (df.iterrows))
				      (res.append (parse_nvme row)))
				 (setf df0 (pd.DataFrame res))
				 (setf df1 (df0.set_index (string "ts")))
				 (setf (aref df1 (string "temp2"))
				       (dot (aref df1 (string "Temperature Sensor 2"))
					    str
					    (extract (rstring3 "(\\d*)\\ C"))
					    (astype int)))
				 (setf (aref df1 (string "temp1"))
				       (dot (aref df1 (string "Temperature Sensor 1"))
					    str
					    (extract (rstring3 "(\\d*)\\ C")) 
					    (astype int)))
				 (dot df1
				      temp1
				      (plot))
				 (dot df1
				      temp2
				      (plot))
				 (plt.legend)
				 (plt.grid)))
		       ))

				 
		 

		 ))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



