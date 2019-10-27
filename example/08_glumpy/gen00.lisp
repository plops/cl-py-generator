(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))



(progn
  (in-package :cl-py-generator)
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/08_glumpy")
  (defparameter *code-file* "run_00_window")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((vertex-code (cl-cpp-generator2::emit-c :code `(do
		  "attribute vec2 position;"
		  (defun main ()
		    (setf gl_Position (vec4 position 0s0 1s0))))))
	 (code
	  `(do0
	    (do0
	     #+nil (imports (matplotlib))
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
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
			    ))
	    "from glumpy import app, gloo, gl"
	    (do0
	     (app.use (string "glfw"))
	     (setf window (app.Window))
	     (setf vertex (string3 ,vertex-code)))
	    
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))


