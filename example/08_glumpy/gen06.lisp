(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *vertex-code* (cl-cpp-generator2::emit-c
			       :code
			       `(do
				 "uniform vec2 viewport;"
				 "uniform mat4 model, view, projection;"
				  "uniform float antialias, thickness, linelength;"
				  "attribute vec3 prev, curr, next;"
				  "attribute vec2 uv;"
				  "varying vec2 v_uv;"
				  "varying vec3 v_normal;"
				  "varying float v_thickness;"
				  (defun main ()
				    ;; normalized device coordinates
				    ,@(loop for e in `(prev curr next) collect
					   (let ((name (format nil "NDC_~a" e)))
					    `(let ((,name
						    (* projection
						       view
						       model
						       (vec4 (dot ,e xyz) 1s0))))
					       (declare (type vec4 ,name)))))
				    ;; screen coordinates
				    ,@(loop for e in `(prev curr next) collect
					   (let ((name (format nil "screen_~a" e)))
					    `(let ((,name
						    (* viewport
						       (/ (+ (/ ,(format nil "NDC_~a.xy" e)
								,(format nil "NDC_~a.w" e))
							     1s0) 2s0))))
					       (declare (type vec2 ,name)))))
				    ;; compute thickness according to line orientation
				    (let ((normal (* model (vec4 curr.xyz 1s0))))
				      (declare (type vec4 normal))
				      (setf v_normal normal.xyz)
				      (if (< normal.z 0)
					  (setf v_thickness (/ thickness 2s0))
					  (setf v_thickness (/ (* thickness (+ (pow normal.z .5)
									       1))
							       2s0))))
				    (let ((w (+ (/ thickness 2s0)
						antialias))
					  (position)
					  (t0 (normalize (- screen_curr.xy screen_prev.xy)))
					  (n0 (vec2 -t0.y t0.x))
					  (t1 (normalize (- screen_next.xy screen_curr.xy)))
					  (n1 (vec2 -t1.y t1.x))
					  
					  
					  )
				      (declare (type float w)
					       (type vec2 position)
					       (type vec2 t0 n0 t1 n1

						     )
					       )
				      (setf v_uv (vec2 uv.x (* uv.y w)))
				      (if (== prev.xy curr.xy)
					  (setf v_uv.x -w
						position (+ screen_curr.xy
							    (* -w t1)
							    (* uv.y w n1)))
					  (if (== curr.xy next.xy)
					      (setf v_uv.x (+ w linelength)
						    position (+ screen_curr.xy
								(* w t0)
								(* uv.y w n0)))
					      (let ((miter (normalize (+ n0 n1)))
						    ;; max avoids glitch for too large miter
						    (dy (/ w (max ("dot" miter n1)
								  1s0))))
						(declare (type vec2 miter)
							 (type float dy))
						(setf position (+ screen_curr.xy
								  (* dy uv.y miter))))))
				      ;; ndc coordinates
				      (setf gl_Position (vec4 (- (/ (* 2s0 position)
								    viewport)
								 1s0)
							      (/ NDC_curr.z
								 NDC_curr.w)
							      1s0)))
				    ))))
  (defparameter *fragment-code*
    (cl-cpp-generator2::emit-c
     :code
     `(do
       "uniform float antialias, thickness, linelength;"
       "varying vec2 v_uv;"
	"varying float v_thickness;"
	"varying vec3 v_normal;"
	
	
	(defun main ()
	  (let ((d 0s0)
		(w (- (/ v_thickness 2s0)
		      antialias))
		(color (vec3 0s0 0s0 0s0)))

	    (declare (type float d w)
		     (type vec3 color))
	    (when (< v_normal.z 0)
	      (setf color (* .75 (vec3 (pow (abs v_normal.z)
					    .5s0)))))
	    (if (< v_uv.x 0)
		(setf d (- (length v_uv) w)) ;; cap at start
		(if (<= linelength
			v_uv.x)
		    (setf d (- (distance v_uv (vec2 linelength 0)) ;; cap at end
			       w))
		    (setf d (- (abs v_uv.y) w)) ;; body
		    ))
	    (if (< d 0)
		(setf gl_FragColor (vec4 color 1s0))
		(setf d (/ d antialias)
		      gl_FragColor (vec4 color (exp (* -d d)))))))))))



(in-package :cl-py-generator)

(progn
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/08_glumpy")
  (defparameter *code-file* "run_06_linestrip3d_thick")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* (
	 
	 (code
	  `(do0
	    "# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/linestrip-3d-better.py"
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
					;(xr xarray)c
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      ))
	    "from glumpy import app, gloo, gl, glm"
	    (do0
	     (setf vertex (string3 ,cl-cpp-generator2::*vertex-code*)
		   fragment (string3 ,cl-cpp-generator2::*fragment-code*)
		   
		   )
	     (app.use (string "glfw"))
	     (setf window (app.Window 1920 1080 :color (tuple 1 1 1 1)))

	     (def bake (P &key (closed False))
	       (setf epsilon 1e-10
		     n (len P))
	       (if (and closed
			(< epsilon (dot (** (- (aref P 0)
					       (aref P -1)) 2)
					(sum))))
		   (setf P (np.append P (aref P 0))
			 P (P.reshape (+ n 1) 3)
			 n (+ n 1)))
	       (setf V (np.zeros (tuple (+ 1 n 1)
					2
					3)
				 :dtype np.float32)
		     UV (np.zeros (tuple n 2 2)
				  :dtype np.float32)
		     (ntuple V_prev V_curr V_next) (tuple
						    (aref V ":-2")
						    (aref V "1:-1")
						    (aref V "2:"))
		     (aref V_curr "..." 0) (aref P ":" np.newaxis 0)
		     (aref V_curr "..." 1) (aref P ":" np.newaxis 1)
		     (aref V_curr "..." 2) (aref P ":" np.newaxis 2)
		     L (dot (np.cumsum
			     (np.sqrt (dot
				       (** (- (aref P "1:")
					      (aref P ":-1"))
					   2)
				       (sum :axis -1))))
			    (reshape (- n 1) 1))
		     ;(aref V_curr "1:" ":" 3) L
		     (aref UV "1:" ":" 0) L
		     (aref UV "..." 1) (tuple 1 -1))
	       (if closed
		   (setf (aref V 0) (aref V -3)
			 (aref V -1) (aref V 2))
		   (setf (aref V 0) (aref V 1)
			 (aref V -1) (aref V -2)))
	       (return (tuple V_prev V_curr V_next UV (aref L -1))))

	     (setf n 2048
		   TT (np.linspace 0
				   (* 20 2 np.pi)
				   n
				   :dtype np.float32)
		   R (np.linspace .1 (- np.pi .1) n :dtype np.float32)
		   X (* (np.cos TT)
			(np.sin R))
		   Y (* (np.sin TT)
			(np.sin R))
		   Z (np.cos R)
		   P (dot (np.dstack (tuple
				      X Y Z))
			  (squeeze))
		   (ntuple V_prev V_curr V_next UV length) (bake P)
		   segments (gloo.Program vertex fragment)
		   )
	     ,@(loop for (e f) in `((prev V_prev)
				    (curr V_curr)
				    (next V_next)
				    (uv UV)
				    (thickness 15s0)
				    (antialias 1.5s0)
				    (linelength length)
				    (model (np.eye 4 :dtype np.float32))
				    (view (glm.translation 0 0 -5))
				    ) collect
		    `(setf (aref segments (string ,e))
			   ,f))
	     (setf phi 0
		   theta 0)
	     
	     
	     (do0
	      "@window.event"
	      (def on_resize (width height)
		(setf (aref segments (string "projection"))
		      (glm.perspective 30s0
				       (/ width (float height))
				       2s0
				       100s0)
		      (aref segments (string "viewport"))
		      (tuple width height))))
	     (do0
	      "@window.event"
	      (def on_init ()
		(gl.glEnable gl.GL_DEPTH_TEST)
					;(gl.glDepthFunc gl.GL_GREATER)
		#+nil (do0
		 (gl.glEnable gl.GL_BLEND)
		 (gl.glBlendFunc gl.GL_SRC_ALPHA gl.GL_ONE_MINUS_SRC_ALPHA))
		))
	     (do0
	      "@window.event"
	      (def on_draw (dt)
		"global phi, theta, duration"
		(window.clear #+nil :clearflags
			      #+nil (logior gl.GL_COLOR_BUFFER_BIT
				      gl.GL_DEPTH_BUFFER_BIT))
					;(gl.glDepthMask gl.GL_FALSE)
		(segments.draw gl.GL_TRIANGLE_STRIP)
		(setf theta (+ theta .1)
		      phi (+ phi .2)
		      model (np.eye 4 :dtype np.float32))
		(glm.rotate model theta 0 1 0)
		(glm.rotate model phi 1 0 0)
		(setf (aref segments (string "model"))
		      model)))
	     (app.run :framerate 60))
	    
	    )))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
