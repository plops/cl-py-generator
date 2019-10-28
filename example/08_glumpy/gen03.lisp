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
				 "uniform float antialias;"
				  "attribute float thickness;"
				  "attribute vec2 p0, p1, uv;"
				  "varying float v_alpha, v_thickness;"
				  "varying vec2 v_p0, v_p1, v_p;"
				  (defun main ()
				    ;; handle thin lines with v_alpha
				    (if (< (abs thickness) 1s0)
					(setf v_thickness 1s0
					      v_alpha (abs thickness))
					(setf v_thickness (abs thickness)
					      v_alpha 1s0))
				    (let (;; half of the width that the shader will touch
					  (tt (+ antialias (/ v_thickness 2s0)))
					  ;; length of the segment (without caps)
					  (l (distance p0 p1))
					  ;; u in [0..1] .. distance along segment
					  ;; u<0, u>l    .. cap area
					  (u (- (* 2s0 uv.x) 1s0))
					  ;; coordinate of line thickness
					  (v (- (* 2s0 uv.y) 1s0))
					  ;; unit vector tangential to segment
					  (TT (normalize (- p1 p0)))
					  ;; unit vector normal to segment 
					  (O (vec2 -TT.y TT.x))
					  (p (+ p0
						(vec2 .5s0 .5s0)
						(* uv.x TT l)
						(* u TT tt)
						(* v O tt))))
				      (declare (type float tt l u v)
					       (type vec2 TT O p))
				      (setf gl_Position (vec4 (- (/ (* 2s0 p)
								    resolution)
								 1s0)
							      0s0 1s0))
				      (do0 "// local space"
					   (setf TT (vec2 1s0 0s0)
						 O (vec2 0s0 1s0)
						 p (+ (* uv.x TT l)
						      (* u TT tt)
						      (* v O tt)))
					   (setf v_p0 (vec2 0s0 0s0)
						 v_p1 (vec2 1s0 0s0)
						 v_p p)))
				    
				    ))))
  (defparameter *fragment-code*
    (cl-cpp-generator2::emit-c
     :code
     `(do
       "uniform float antialias;"
       "varying float v_thickness, v_alpha;"
	"varying vec2 v_p0, v_p1, v_p;"
	
	
	(defun main ()
	  (let ((d 0s0)
		(offset (+ (/ v_thickness -2s0)
			   (/ antialias 2s0))))
	    (declare (type float d offset))
	    ;; compute signed distance to envelope
	    (if (< v_p.x 0)
		(setf d (+ (distance v_p v_p0)
			   offset))
		(if (< (distance v_p1 v_p0)
		       v_p.x)
		    (setf d (+ (distance v_p v_p1)
			       offset))
		    (setf d (+ (abs v_p.y)
			       offset))))
	    (if (< d 0)
		(setf gl_FragColor (vec4 0s0 0s0 0s0 v_alpha))
		(when (< d antialias)
		  (setf d (exp (* -d d))
			gl_FragColor (vec4 0s0 0s0 0s0 (* v_alpha d)))))))))))



(in-package :cl-py-generator)

(progn
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/08_glumpy")
  (defparameter *code-file* "run_03_line_segments")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* (
	 
	 (code
	  `(do0
	    "# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/agg-segments.py "
	    "# https://www.labri.fr/perso/nrougier/python-opengl/#id39"
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
	     (setf window (app.Window 1200 400 :color (tuple 1 1 1 1)))
	     
	     (setf n 100
		   V (np.zeros (tuple n 4)
			       :dtype (list (tuple (string "p0") np.float32 2)
					    (tuple (string "p1") np.float32 2)
					    (tuple (string "uv") np.float32 2)
					    (tuple (string "thickness") np.float32 1)
					    ))
		   (aref V (string "p0")) (dot (np.dstack
						(tuple (np.linspace 100 1100 n)
						       (* (np.ones n) 50)))
					       (reshape n 1 2))
		   (aref V (string "p1")) (dot (np.dstack
						(tuple (np.linspace 110 1110 n)
						       (* (np.ones n) 350)))
					       (reshape n 1 2))
		   (aref V (string "uv")) (tuple (tuple 0 0)
						 (tuple 0 1)
						 (tuple 1 0)
						 (tuple 1 1))
		   (aref V (string "thickness")) (dot (np.linspace .1s0 8s0 n)
						      (reshape n 1))
		   )
	     
	     (setf vertex (string3 ,cl-cpp-generator2::*vertex-code*)
		   fragment (string3 ,cl-cpp-generator2::*fragment-code*)
		   segments (gloo.Program vertex fragment :count (* 4 n))
		   )
	     (segments.bind (dot V
				 (ravel)
				 (view gloo.VertexBuffer)))
	     (setf (aref segments (string "antialias")) 2s0)

	     (setf I (np.zeros (tuple n 6) :dtype np.uint32)
		   (aref I ":") (list 0 1 2 1 2 3)
		   
		   I (+ I (* 4 (dot (np.arange n :dtype np.uint32)
				    (reshape n 1))))
		   I (dot I (ravel) (view gloo.IndexBuffer)))
	     (do0
	      "@window.event"
	      (def on_resize (width height)
		(setf (aref segments (string "resolution"))
		      (tuple width height))))
	     (do0
	      "@window.event"
	      (def on_draw (dt)
		(window.clear)
		(segments.draw gl.GL_TRIANGLES I)))
	     (app.run))
	    
	    )))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
