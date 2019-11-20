(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *c-code* (cl-cpp-generator2::emit-c
			       :code
			       `(do)))
 )



(in-package :cl-py-generator)

(progn
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/12_cupy")
  (defparameter *code-file* "run_00_cupy")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* (
	 
	 (code
	  `(do0
	    "# https://youtu.be/CQDsT81GyS8?t=3238 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019"
	    "# pip3 install --user cupy-cuda101"
	    #+nil (do0
	     (imports (matplotlib))
					;(matplotlib.use (string "Agg"))
	     (imports ((plt matplotlib.pyplot)))
	     (plt.ion))
	    
	     (imports (			;os
					;sys
					;time
					;docopt
					;pathlib
		       (np numpy)
		       (cp cupy)
					;serial
					;(pd pandas)
					;(xr xarray)c
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      math
		      ))
	     (do0
	      (setf ary (dot (cp.arange 10)
			     (reshape (tuple 2 5))))
	      (print (repr ary))
	      ,@(loop for e in `(dtype shape strides device) collect
		     `(print (dot (string ,(format nil "~a = {}" e))
				  (format (dot ary ,e)))))
	      )
	    )))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
