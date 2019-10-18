(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/06_fastai")
  (defparameter *code-file* "run_00_pets")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# lesson 1: image classification"
	    "# export LANG=en_US.utf8"
	    "# https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb"
	    
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


	    "from fastai import *"
	    "from fastai.vision import *"
	    "from fastai.metrics import error_rate"
	    ;"from PIL import Image"
	    (setf path (untar_data URLs.PETS))
	    (print (dot (string "using image data from: {}")
			(format path)))
	    (np.random.seed 2)
	    (setf bs 64)
	    (do0
	     "# %%"
	     (setf (ntuple path_anno path_img) (path.ls)
		   fnames (get_image_files path_img)
		   pat (rstring3 "/([^/]+)_\\d+.jpg$")
		   data (ImageDataBunch.from_name_re path_img
						     fnames
						     pat
						     :ds_tfms (get_transforms)
						     :size 224
						     :bs bs))
	     (data.normalize imagenet_stats))
	    (def look ()
	     (do0
	      (data.show_batch :rows 3)
	      (print data.classes)))
	    (setf learn (cnn_learner data
				     models.resnet34
				     :metrics error_rate))
	    (learn.fit_one_cycle 4)
	    (learn.save (string "save-1"))
	    ))
	 )
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)
   ))


