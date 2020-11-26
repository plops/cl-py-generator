(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/29_ondrejs_challenge")
  (defparameter *code-file* "run_00_start")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 "# %% imports"
		 (do0
		  
		  (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		  (imports ((plt matplotlib.pyplot)
			    (animation matplotlib.animation) 
                            ;(xrp xarray.plot)
			    ))
                  
		  (plt.ion)
					;(plt.ioff)
		  ;;(setf font (dict ((string size) (string 6))))
		  ;; (matplotlib.rc (string "font") **font)
		  )
		 (imports (		;os
			   sys
			   time
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
			   ;scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
			   ;datetime
			   ;dask.distributed
					;(da dask.array)
					;PIL
			   libtiff
			   ))
		 

		 (setf
	       _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py")
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
			      (- tz)))))
		 (do0
		  (setf fns ("list" #-nil (dot (pathlib.Path (string "./supplementary_materials/video"))
					       (glob (string "*.tif")))
				    #+nil (dot (pathlib.Path (string "./supplementary_materials/photos"))
					 (glob (string "*.tiff")))))

		  (for (fn fns)
		       (do0
			(print fn)
		  #+nil
		    (setf fn (string "./supplementary_materials/photos/RIMG1832-1.tiff"))
					;(setf dat (plt.imread fn))
					;(setf im (PIL.Image.open fn))
		    (comments "pip3 install --user libtiff")
		    (setf tif (libtiff.TIFF.open fn))
		    (setf im (tif.read_image))
		    (comments "(256,512) float64"
			      "assume it is real and imag next to each other")
		    (setf k (+ (* 1s0 (aref im ":" ":256"))
			       (* 1j (aref im ":" "256:"))))
		    (comments "a dot is in the middle, so i will need fftshift")
		    (setf sk (np.fft.ifftshift k ; :axes (tuple 1 0)
					       ))
		    (setf ik (np.fft.ifft2 sk))
		    (do0
		     (setf pl (tuple 2 2))
		     (plt.close (string "all"))
		     (plt.figure 0 (tuple 16 9))
		     (do0
		      (setf ax (plt.subplot2grid pl (tuple 0 0)))
		      (plt.title fn)
		      (plt.imshow (np.log (np.abs k))))
		     (do0
		      (setf ax (plt.subplot2grid pl (tuple 1 0)))
		      (plt.title (string "fftshift"))
		      (plt.imshow (np.log (np.abs sk))))
		     (do0
		      (setf ax (plt.subplot2grid pl (tuple 0 1)))
		      (plt.title (string "inverse fft"))
		      (plt.imshow (np.real ik) :cmap (string "gray")))
		     
		     (plt.savefig (dot (string "/dev/shm/{}.png")
				       (format fn.stem))))))
		  )))
	   ))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

