(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/15_slab_waveguide_modes")
  (defparameter *code-file* "run_00_slab_mode")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# "
	    "# export LANG=en_US.utf8"
	    
	    
	    (do0
	     (imports (matplotlib))
			      ;(matplotlib.use (string "Agg"))
			      (imports ((plt matplotlib.pyplot)))
			 (plt.ion))
	    
	    (imports (			os
					;sys
					;time
					;docopt
					;pathlib
		      ;(np numpy)
					;serial
			    pathlib
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      ))

	    (setf lam0 1s0
		  n1 2s0
		  n2 1s0
		  a (* 3 lam0) ;; core thickness
		  b (* 5 lam0) ;; substrate thickness
		  dx (/ lam0 20) ;; grid resolution
		  M 5 ;; number of modes to calculate
		  )
	    ))) 
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
 

 
    
