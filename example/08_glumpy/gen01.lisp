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
					;"attribute vec4 color;"
					;"varying vec4 v_color;"
			 "varying vec2 v_position;"
			  (defun main ()
			   (setf gl_Position (vec4 position 0s0 1s0))
			   ;(setf v_color color)
			   (setf v_position position)
			   ))))
	 (fragment-code (cl-cpp-generator2::emit-c
			 :code
			 `(do
					;"varying vec4 v_color;"
			   "varying vec4 v_position;"
			   (defun distance (p center radius)
			     (declare (type vec2 p center)
				      (type float radius)
				      (values float))
			     (return (- (length (- p center))
					radius)))
			    (defun color (d)
			      (declare (type float d)
				       (values vec4))
			      (let ((white (vec3 1 1 1))
				    (blue (vec3 .1 .4 .7))
				    (color (- white (* (sign d) blue))))
				(declare (type vec3 white blue color))
				(setf color (* color
					       (- 1s0
						  (* (exp (* -4s0 (abs d)))
						     (+ .8s0 (* .2s0 (cos (* 140s0 d))))))))
				(setf color (mix color white
						 (- 1s0 (smoothstep 0s0 .02s0 (abs d)))))
				(return (vec4 color 1s0))))
			    (defun main ()
			      (let ((epsilon .005s0)
				    (d (distance v_position.xy (vec2 0s0) .5s0))
				    )
				
				(declare (type "const float" epsilon)
					 (type float d))
				(setf gl_FragColor (color d))
				#+nil (if (< d (- epsilon))
				    (setf gl_FragColor (vec4 (- 1s0 (abs d))
							     0 0 1))
				    (if (< epsilon d)
					(setf gl_FragColor (vec4 0 0 (- 1s0 (abs d))
								 1))
					(setf gl_FragColor (vec4 1 1 1 1)))))))))
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
		   )
	     #+nil (setf (aref quad (string "color"))
			       (ntuple
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


 
