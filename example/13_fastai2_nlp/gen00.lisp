(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/13_fastai2_nlp")
  (defparameter *code-file* "run_00_nlp")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb"
	    "# export LANG=en_US.utf8"
	    
	    
	    (do0
	     (imports (matplotlib))
			      ;(matplotlib.use (string "Agg"))
			      (imports ((plt matplotlib.pyplot)))
			 (plt.ion))
	    
	    #+nil (imports (			;os
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
					;scipy.ndimage
					;scipy.optimize
		      ))


	    "from fastai2.text.all import *"


	    (setf path (untar_data URLs.IMDB))
	    (comment "=> Path('/home/martin/.fastai/data/imdb')")

	    (setf files (get_text_files path
					:folders
					(list (string "train")
					      (string "test")
					      (string "unsup"))))
	    
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))


