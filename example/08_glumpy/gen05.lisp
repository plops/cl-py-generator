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
				  (defun main ()
				    (let (;; normalized device coordinates
					  ,@(loop for e in `(prev curr next) collect
						 `(,(format nil "NDC_~a" e)
						    (* projection
						       view
						       model
						       (vec4 (dot ,e xyz) 1s0))))
					  ;; screen coordinates
					  ,@(loop for e in `(prev curr next) collect
						 `(,(format nil "screen_~a" e)
						    (* viewport
						       (/ (+ (/ ,(format nil "NDC_~a.xy" e)
								,(format nil "NDC_~a.w" e))
							     1s0) 2s0)
						       (vec4 (dot ,e xyz) 1s0)))))
				      (declare (type vec4 NDC_prev NDC_curr NDC_next)
					       (type vec4 screen_prev screen_curr screen_next))
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
								     resolution)
								  1s0)
							       (/ NDC_curr.z
								  NDC_curr.w)
							       1s0))))
				    ))))
  (defparameter *fragment-code*
    (cl-cpp-generator2::emit-c
     :code
     `(do
       "uniform float antialias, thickness, linelength;"
       "varying vec2 v_uv;"
	
	
	(defun main ()
	  (let ((d 0s0)
		(w (- (/ thickness 2s0)
		      antialias)))
	    (declare (type float d w))
	    (if (< v_uv.x 0)
		(setf d (- (length v_uv) w))
		(if (<= linelength
			v_uv.x)
		    (setf d (- (distance v_uv (vec2 linelength 0))
			       w))
		    (setf d (- (abs v_uv.y) w))))
	    (if (< d 0)
		(setf gl_FragColor (vec4 0s0 0s0 0s0 1s0))
		(setf d (/ d antialias)
		      gl_FragColor (vec4 0s0 0s0 0s0 (exp (* -d d)))))))))))



(in-package :cl-py-generator)

(progn
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/08_glumpy")
  (defparameter *code-file* "run_05_linestrip3d")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* (
	 
	 (code
	  `(do0
	    "# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/linestrip-3d.py"
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
	    "from glumpy import app, gloo, gl"
	    (do0
	     (setf vertex (string3 ,cl-cpp-generator2::*vertex-code*)
		   fragment (string3 ,cl-cpp-generator2::*fragment-code*)
		   
		   )
	     (app.use (string "glfw"))
	     (setf window (app.Window 1200 400 :color (tuple 1 1 1 1)))

	     (def bake (P &key (closed False))
	       (setf epsilon 1e-10
		     n (len P))
	       (if (and closed
			(< epsilon (dot (** (- (aref P 0)
					       (aref P -1)) 2)
					(sum))))
		   (setf P (np.append P (aref P 0))
			 P (P.reshape (+ n 1) 2)
			 n (+ n 1)))
	       (setf V (np.zeros (tuple (+ 1 n 1)
					2
					4)
				 :dtype np.float32)
		     (ntuple V_prev V_curr V_next) (tuple
						    (aref V ":-2")
						    (aref V "1:-1")
						    (aref V "2:"))
		     (aref V_curr "..." 0) (aref P ":" np.newaxis 0)
		     (aref V_curr "..." 1) (aref P ":" np.newaxis 1)
		     (aref V_curr "..." 2) (tuple 1 -1)
		     L (dot (np.cumsum
			     (np.sqrt (dot
				       (** (- (aref P "1:")
					      (aref P ":-1"))
					   2)
				       (sum :axis 1))))
			    (reshape (- n 1) 1))
		     (aref V_curr "1:" ":" 3) L)
	       (if closed
		   (setf (ntuple (aref V 0)
				 (aref V -1))
			 (tuple (aref V -3)
				(aref V 2)))
		   (setf (ntuple (aref V 0)
				 (aref V -1))
			 (tuple (aref V 1)
				(aref V -2))))
	       (return (tuple V_prev V_curr V_next (aref L -1))))

	     (setf n 1024
		   TT (np.linspace 0
				   (* 12 2 np.pi)
				   n
				   :dtype np.float32)
		   R (np.linspace 10 246 n :dtype np.float32)
		   P (dot (np.dstack (tuple
				      (+ 256 (* (np.cos TT) R))
				      (+ 256 (* (np.sin TT) R))))
			  (squeeze))
		   (ntuple V_prev V_curr V_next length) (bake P)
		   segments (gloo.Program vertex fragment)
		   )
	     ,@(loop for (e f) in `((prev V_prev
					  )
				    (curr V_curr)
				    (next V_next)
				    (thickness
				     9s0)
				    (antialias 1.5s0)
				    (linelength length)) collect
		    `(setf (aref segments (string ,e))
			   ,f))
	     
	     
	     (do0
	      "@window.event"
	      (def on_resize (width height)
		(setf (aref segments (string "resolution"))
		      (tuple width height))))
	     (do0
	      "@window.event"
	      (def on_draw (dt)
		(window.clear)
		(segments.draw gl.GL_TRIANGLE_STRIP)))
	     (app.run))
	    
	    )))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
