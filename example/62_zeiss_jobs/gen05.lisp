(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria")
  ;(ql:quickload "cl-who")
  )
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/62_zeiss_jobs")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                           ,@rest)))))

  (write-notebook
   :nb-file (format nil "~a/source/04_overview.ipynb" *path*)
   :nb-code
   `(
     (python (do0
	      (do0
	       "%matplotlib notebook"
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
			 time
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
					;   scipy.optimize
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
					requests
			   
					;(np jax.numpy)
					;(mpf mplfinance)
			 ;selenium.webdriver ;.FirefoxOptions
			   
			 ))
	       
		 
	      #+nil  (imports-from (selenium webdriver)
			     (selenium.webdriver.common.keys Keys)
			     (selenium.webdriver.support.ui WebDriverWait)
			     (selenium.webdriver.common.by By)
			     (selenium.webdriver.support expected_conditions)
			     
			     
			     )
		 
	       
	       (imports-from (matplotlib.pyplot
			      plot imshow tight_layout xlabel ylabel
			      title subplot subplot2grid grid
			      legend figure gcf xlim ylim)
					;(helium *)
			     ;;https://pythonawesome.com/a-library-to-generate-html-with-python-3/
			     (domonic.html *))
		 
	       )
	      ))
     (python
      (do0
       (setf start_time (time.time)
	     debug True)
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
	     debug True)))
     (python
      (do0
       (setf df (pd.read_csv (string "contents3.csv")))
       ))
     

     (python
      (do0
       (setf tab (table
		     (tr
		      (th (string "idx"))
		      (th (string "name"))
		      (th (string "link")))
		     ))
       (for ((ntuple idx row) (df.iterrows))
	    (tab.appendChild
	     (tr
	      (td (dot (string "{}")
		       (format idx)))
	      (td (a row.job :_href (dot (string "/")
					 (join (dot  row
						     link
						     (aref
						      (split (string "/"))
						      (slice -4 "")))))))
	      ;(td row.link)
	      )))
       (setf page (html
		   (body
		    (h1 (string "Hello World"))
		    tab)))
       (render page (string "index.html"))))
     ))
  )



