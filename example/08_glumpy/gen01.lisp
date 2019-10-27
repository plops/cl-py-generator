(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))


(in-package :cl-cpp-generator2)

(progn
  (defparameter *vertex-code* (cl-cpp-generator2::emit-c
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
  (defparameter *fragment-code*
    (cl-cpp-generator2::emit-c
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
				      (defun SDF_circle (p radius)
					(declare (type vec2 p)
						 (type float radius)
						 (values float))
					(return (- (length p)
						   radius)))
				      (defun SDF_plane (p p0 p1)
					(declare (type vec2 p p0 p1)
						 (values float))
					(let ((tt (- p1 p0))
					      (o (normalize (vec2 tt.y (- tt.x)))))
					  (declare (type vec2 tt o))
					  (return ("dot" o (- p0 p)))))
				      (defun SDF_box (p size)
					(declare (type vec2 p size)
						 (values float))
					(let ((d (- (abs p) size))
					      )
					  (declare (type vec2 d))
					  (return (+ (min (max d.x d.y)
							  0s0)
						     (length (max d 0s0))))))
				      (defun SDF_round_box (p size radius)
					(declare (type vec2 p size)
						 (type float radius)
						 (values float))
					(return (- (SDF_box p size)
						   radius)))
				      (defun SDF_fake_box (p size)
					(declare (type vec2 p size)
						 (values float))
					(return (max (- (abs p.x) size.x)
						     (- (abs p.y) size.y))))
				      (defun SDF_triangle (p p0 p1 p2)
					(declare (type vec2 p p0 p1 p2)
						 (values float))
					,@(loop for (e f) in `((1 0)
							       (2 1)
							       (0 2)) and i from 0
					     collect
					       (let ((name (format nil "e~a" i)))
						 `(let ((,name
							 (- ,(format nil "p~a" e)
							    ,(format nil "p~a" f))))
						    (declare (type vec2 ,name)))))
					,@(loop for (e f) in `(("" 0)
							       ("" 1)
							       ("" 2)) and i from 0
					     collect
					       (let ((name (format nil "v~a" i)))
						 `(let ((,name
							 (- ,(format nil "p~a" e)
							    ,(format nil "p~a" f))))
						    (declare (type vec2 ,name)))))
					,@(loop for i below 3 collect
					       (let ((name (format nil "pq~a" i))
						     (v (format nil "v~a" i))
						     (e (format nil "e~a" i)))
						 `(let ((,name
							 (- ,v
							    (* ,e
							       (clamp (/ ("dot" ,v ,e)
									 ("dot" ,e ,e))
								      0s0 1s0)))))
						    (declare (type vec2 ,name)))))
					(let ((s (sign (- (* e0.x e2.y)
							  (* e0.y e2.x))))
					      )
					  (declare 
					   (type float s)))
					,@(loop for i below 3 collect
					       (let ((name (format nil "vv~a" i))
						     (v (format nil "v~a" i))
						     (e (format nil "e~a" i))
						     (pq (format nil "pq~a" i)))
						 `(let ((,name
							 (vec2 ("dot" ,pq ,pq)
							       (* s (- (* (dot ,v x) (dot ,e y))
								       (* (dot ,v y) (dot ,e x)))))))
						    (declare (type vec2 ,name)))))
					(let (
					      (d (min (min vv0 vv1)
						      vv2))
					      )
					  (declare (type vec2 d)
						   )
					  (return (* (- (sqrt d.x))
						     (sign d.y)))))
		     
		     
		     
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
							(setf gl_FragColor (vec4 1 1 1 1))))))))))

(in-package :cl-py-generator)

(progn
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/08_glumpy")
  (defparameter *code-file* "run_01_window")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* (
	 
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
	     (setf vertex (string3 ,cl-cpp-generator2::*vertex-code*)
		   fragment (string3 ,cl-cpp-generator2::*fragment-code*)
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


 
