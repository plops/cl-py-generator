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
	    (setf bs 140) ;;  5793MiB /  7981MiB
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
						     ;; if you run out of memory make bs smaller
						     :bs bs))
	     (data.normalize imagenet_stats))
	    (print (string "use look() to see example data"))
	    (def look ()
	     (do0
	      (data.show_batch :rows 3)
	      (print data.classes)))
	    (do0
	     (setf learn (cnn_learner data
					models.resnet34
					:metrics error_rate))
	     (print (string "learn extra layers at the end"))
	     (setf fn
		   (pathlib.Path (string "/home/martin/.fastai/data/oxford-iiit-pet/images/models/save-1.pth")))
	     (if (fn.is_file)
	      (do0
	       (learn.load fn.stem))
	      (do0
	       (learn.fit_one_cycle 4)
	       (learn.save fn.stem))))
	    (do0
	     "# %%"
	     (setf interp (ClassificationInterpretation.from_learner learn)
		   (ntuple losses idxs) (interp.top_losses))
	     (interp.plot_top_losses 9) ;; things we were the most confident about and got wrong
	     ;; prediction, actual, loss, probability of actual class
	     (interp.plot_confusion_matrix)
	     (print
	      (interp.most_confused :min_val 2)))
	    (do0
	     "# %%"
	     (print (string "learn parameters of the whole model"))
	     (do0
	      (setf fn2 (pathlib.Path (string "/home/martin/.fastai/data/oxford-iiit-pet/images/models/save-2.pth")))
	      (if (fn2.is_file)
		  (do0
		   (learn.load fn2.stem))
		  (do0
		   ;; learning rate finder finds what is the fastest way to train net
		   (learn.lr_find)
		   ;; for fine tuning we have to be carefu with learning rate
		   ;; from layer 4 on we see dog faces
		   (learn.recorder.plot)
		   ;; first layers with 1e-6, last layers with 1e-4
		   ;; other layer with rater in between
		   (learn.unfreeze)
		   (learn.fit_one_cycle 2 :max_lr ("slice" 1e-6 1e-4))
		   (learn.save fn2.stem))))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))


