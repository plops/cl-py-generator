(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))



(progn
  
  (in-package :cl-py-generator)
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/08_glumpy")
  (defparameter *code-file* "run_01_window")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((vertex-code (cl-cpp-generator2::emit-c
		       :code
		       `(do
			 "attribute vec2 position;"
			 "attribute vec4 color;"
			  "varying vec4 v_color;"
			  (defun main ()
			   (setf gl_Position (vec4 position 0s0 1s0))
			   (setf v_color color)))))
	 (fragment-code (cl-cpp-generator2::emit-c
			 :code
			 `(do
			   "varying vec4 v_color;"
			   (defun main ()
			     (setf gl_FragColor v_color)))))
	 (code
	  `(do0
	    "# https://github.com/rougier/python-opengl/blob/master/code/chapter-03/glumpy-quad-solid.py"
	    "# https://www.labri.fr/perso/nrougier/python-opengl/#id7"
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
	     (setf vertex (string3 ,vertex-code)
		   fragment (string3 ,fragment-code)
		   quad (gloo.Program vertex fragment :count 4)
		   (aref quad (string "position"))  (ntuple (tuple -1 1)
							    (tuple 1 1)
							    (tuple -1 -1)
							    (tuple 1 -1))
		   (aref quad (string "color"))  (ntuple
						  (tuple 1 1 0 1)
						  (tuple 1 0 0 1)
						  (tuple 0 0 1 1)
						  (tuple 0 1 0 1)))
	     (do0
	      "@window.event"
	      (def on_draw (dt)
		(window.clear)
		(quad.draw gl.GL_TRIANGLE_STRIP)))
	     (app.run))
	    
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
