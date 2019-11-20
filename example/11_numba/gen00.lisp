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
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/11_numba")
  (defparameter *code-file* "run_00_hypot")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* (
	 
	 (code
	  `(do0
	    "# https://youtu.be/CQDsT81GyS8?t=2394 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019"
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
					;serial
					;(pd pandas)
					;(xr xarray)c
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      ))
	    "from numba import jit"
	    "@jit"
	    (def hypot (x y)
	      (setf x (np.abs x)
		    y (np.abs y)
		    tt (np.min x y)
		    x (np.max x y)
		    tt (/ tt x))
	      (return (* x (np.sqrt (+ 1 (* tt tt)))))))))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
