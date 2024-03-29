(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/67_self_reference")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (let ((nb-file "source/01_plot.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp plot01"))
      (python (do0
	       
	       "#export"
	       (do0
					;"%matplotlib notebook"
		(do0
		      
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
			  time
					;docopt
					;pathlib
					(np numpy)
					;serial
					(pd pandas)
					(xr xarray)
					(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
			  ;scipy.optimize
			  ;scipy.stats
			  ;scipy.special
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
					;re
					;json
					; csv
					;io.StringIO
					;bs4
					;requests
					;(nx networkx)
					;(np jax.numpy)
					;(mpf mplfinance)
			   
			  ))
	      
		#+nil 
		(imports-from (selenium webdriver)
			      (selenium.webdriver.common.keys Keys)
			      (selenium.webdriver.support.ui WebDriverWait)
			      (selenium.webdriver.common.by By)
			      (selenium.webdriver.support expected_conditions)
			     
			     
			      )
		#+nil
		(imports-from (flask Flask render_template
				     request redirect url_for))
		
		(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
		 
		)
	       ))
      (python
       (do0
	"#export"
	(setf
	 _code_git_version
	 (string ,(let ((str (with-output-to-string (s)
			       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		    (subseq str 0 (1- (length str)))))
	 _code_repository (string ,(format nil "~a/~a/~a" *repo-dir-on-github* *example-subdir* nb-file))
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
	      debug True)))

      (markdown
       "Twitter Fermat's Library @fermatslibrary Dec 13")
      (python
       (do0
	"#export"
	(setf x (* .1 (np.arange 1200))
	      y (* .1 (np.arange 150)))
	(setf xs (xr.DataArray
		  :data (np.zeros (list (len x) (len y)))
		  :dims (list (string "x")
			      (string "y"))
		  :coords
		  (dictionary

		   :x x
		   :y y
		   )))

	(setf x2 (dot np
		      (tile x (list (len y) 1))
		      (transpose))
	      y2 (dot np
		      (tile y (list (len x) 1))))
	#+nil (setf xs.values
	      (< .5
		 (np.floor
		  (np.mod
		   (* (// y2 17)
		      (np.power 2 (+
				  (* -17 (np.floor x2)
				     )
				  (np.mod (np.floor y2)
					  17))))
		   2))))
	(setf xs.values
	      (* (// y2 17)
		 (np.power 2 (+
			      (* -17 (np.floor x2)
				 )
			      (np.mod (np.floor y2)
				      17))))
	      )))
      (python
       (do0
	(xs.plot)))
      ))))



