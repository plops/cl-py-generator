(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
 
(in-package :cl-py-generator)
;; https://python-gtk-3-tutorial.readthedocs.io/en/latest/treeview.html


(progn
  (defparameter *repo-sub-path* "25_fastai2_again")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *repo-sub-path*))
  (defparameter *code-file* "run_00_pets")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  
  (let* ((code
	  `(do0
	    "#!/usr/bin/python3"
	    
	    (do0
	     #+nil (do0
		    (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
		    "from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
		    "from matplotlib.figure import Figure"
		    (imports ((plt matplotlib.pyplot)
			      matplotlib.colors
			      (animation matplotlib.animation)
			      (xrp xarray.plot)))
		    
		    (plt.ion)
					;(plt.ioff)
		    (setf font (dict ((string size) (string 8))))
		    (matplotlib.rc (string "font") **font)
		    )


	     
	     )
	    

	    
	    (imports (			;os
					;sys
					;traceback
					;pdb
					;time
					;docopt
					;pathlib
					;(yf yfinance)
					;(np numpy)
					;collections
					;serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration 
					;skimage.feature
					;skimage.morphology
					;skimage.measure
					; (u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;scipy.optimize
					;scipy.ndimage.morphology
					; nfft
					; ttv_driver
					;pathlib
					;re
					;requests
					;zipfile
					;io
					;sklearn
					;sklearn.linear_model
		      ;wx
		      ))

	    "from fastai.vision.all import *"
	    
	    (do0
	     (comment "%%")
	     (setf
	      _code_git_version
	      (string ,(let ((str 
			      #-sbcl "xxx"
			      #+sbcl (with-output-to-string (s)
				       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/~a.py" *repo-sub-path* *code-file*)
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
				 (- tz))))))

	    (do0

	     (setf path (/  (untar_data URLs.PETS)
			    (string "images")))
	     (def is_cat (x)
	       (return (dot (aref x 0)
			    (isupper))))
	     (setf dls (ImageDataLoaders.from_name_func path (get_image_files path)
							:valid_pct .2
							:seed 42
							:label_func is_cat
							:item_tfms (Resize 224)))
	     (setf learn (cnn_learner dls resnet34 :metrics error_rate))
	     (learn.fine_tune 1))

	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))





