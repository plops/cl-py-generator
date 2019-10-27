(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))




;; define the shader within the cl-cpp-generator2 package. use C-c C-l in slime to compile/load this whole file and generate the python code (that also contains the shaders as a string

(in-package :cl-cpp-generator2)

(progn
  (defparameter *vertex-code* (cl-cpp-generator2::emit-c
			       :code
			       `(do
				 "uniform vec2 resolution;"
				 "attribute vec2 center;"
				  "attribute float radius;"
				  "varying vec2 v_center;"
				  "varying float v_radius;"
					;"attribute vec2 position;"
					;"attribute vec4 color;"
					;"varying vec4 v_color;"
					;"varying vec2 v_position;"
				  (defun main ()
				    (setf v_radius radius
					  v_center center
					  gl_PointSize (+ 2s0 (ceil (* 2s0 radius))))
				    (setf gl_Position (vec4 (+ -1s0 (* 2s0 (/ center resolution)))
							    0s0 1s0))
				    
				    ))))
  (defparameter *fragment-code*
    (cl-cpp-generator2::emit-c
     :code
     `(do
					;"varying vec4 v_color;"
					;"varying vec4 v_position;"
       "varying vec4 v_center;"
       "varying float v_radius;"
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
	    y	       (declare 
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
	  (let ((p (- gl_FragCoord.xy v_center))
		(a 1s0)
		(d (+ (length p)
		      (- v_radius)
		      1s0)))
	    (declare (type vec2 p)
		     (type float a d))
	    (setf d (abs d))
	    (when (< 0s0 d)
	      (setf a (exp (* -1 d d))))
	    (setf gl_FragColor (vec3 (vec3 0s0) a))))))))



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
	     (setf V (np.zeros 16 (list (tuple (string "center") np.float32 2)
					(tuple (string "radius") np.float32 1)))
		   (aref V (string "center")) (np.dstack
					       (list (np.linspace 32 (- 512 32) (len V))
						     (np.linspace 25 28 (len V))))
		   (aref V (string "radius")) 15)
	     (setf window (app.Window 512 50 :color (tuple 1 1 1 1)))
	     (setf vertex (string3 ,cl-cpp-generator2::*vertex-code*)
		   fragment (string3 ,cl-cpp-generator2::*fragment-code*)
		   points (gloo.Program vertex fragment)
		   )
	     (points.bind (V.view gloo.VertexBuffer))
	     (do0
	      "@window.event"
	      (def on_resize (width height)
		(setf (aref points (string "resolution"))
		      (tuple width height))))
	     (do0
	      "@window.event"
	      (def on_draw (dt)
		(window.clear)
		(points.draw gl.GL_POINTS)))
	     (app.run))
	    
	    )))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
