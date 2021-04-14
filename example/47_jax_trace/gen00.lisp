(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/47_jax_trace")
  (defparameter *code-file* "run_00_start")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 
		 (do0
		  
		  (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		  (imports ((plt matplotlib.pyplot)
			  ;  (animation matplotlib.animation) 
                            ;(xrp xarray.plot)
			    ))
                  
		  (plt.ion)
		  "from mpl_toolkits.mplot3d import Axes3D"
					;(plt.ioff)
		  (setf font (dict ((string size) (string 6))))
		  (matplotlib.rc (string "font") **font)
		  )
		 (imports (		;os
			   ;sys
			   ;time
					;docopt
			   ;pathlib
			   ;(np numpy)
			   ;serial
			   ;(pd pandas)
			   (xr xarray)
			   (xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			   ;scipy.ndimage
			   scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
					;datetime
			   (np jax.numpy)
			   jax
			   jax.random
			   jax.config
			   ))
		 "from jax import grad, jit, jacfwd, jacrev, vmap, lax, random"
		 ,(format nil "from jax.numpy import ~{~a~^, ~}" `(sqrt newaxis sinc abs))
		 (jax.config.update (string "jax_enable_x64")
					   True)
		 (setf
		  _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

		  _code_generation_time
		  (string ,(multiple-value-bind 
				 (second minute hour date month year day-of-week dst-p tz)
			       (get-decoded-time)
			     (declare (ignorable dst-p))
			     (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				     hour
				     minute
				     second
				     (nth day-of-week *day-names*)
				     year
				     month
				     date
				     (- tz))
			     )))
		 (do0
		  (def length (p)
		    (return (np.linalg.norm p ;:axis 1 :keepdims True
					    )))
		  (def normalize (p)
		    (return (/ p (length p))))
		  (def raymarch (ro rd sdf_fn &key (max_steps 10))
		    (setf tt 0.0)
		    (for (i (range max_steps))
			 (setf p (+ ro (* tt rd))
			       tt (+ tt (sdf_fn p))))
		    (return tt))
		  (comments "numers in meter, light source is 1m x 1m, box is 4m x 4m")
		  ,(let ((l `((none 0.0)
			      (floor .1 :c py)
			      (ceiling .2 :c (- 4.0 py))
			      (leftwall .3 :c (- px -2.0))
			      (backwall .4 :c (- 4.0 pz))
			      (rightwall .5 :c (- 2.0 px))
			      (short_block .6 :c d :code
					  (setf bw .6
						p2 (rotateY (- p
							       (np.array (list .65
									       bw
									       1.7)))
							    (* -.1 np.pi))
						d (udBox p2 (np.array (list bw bw bw)))))
			      (tall_block .7 :c d :code
					   (setf bh 1.3
							       p2 (rotateY (- p
									      (np.array (list -.64
											      bh
											      2.6)))
								     (* .15 np.pi))
							 d (udBox p2 (np.array (list .6 bh .6)))))
			      (light 1.0 :c (udBox (- p (np.array (list 0 3.9 2.0))
						      )
						   (np.array (list .5 .01 .5))))
			      (sphere .9))))
		     `(do0
		         ,@(loop for e in l
				 collect
				 (destructuring-bind (object-name object-id &key c code) e
				   `(do0
				     (setf ,(string-upcase (format nil "obj_~a" object-name))
					   ,object-id))))
		       (def df (obj_id dist)
			 (string3 "hard coded enums for each object. associate intersection points with their nearest object. values are arbitrary")
			 (return (np.array (list obj_id dist))))
		       (def udBox (p b)
			 (comments "distance field of box, b .. half-widths")
			 (return (length (np.maximum (- (np.abs p)
							b)
						     0.0))))
		       ,(let ((lr `((X (px (- (* c py)
					      (* s pz))
					   (+ (* s py)
					      (* c pz))))
				    (Y ((+ (* c px)
					   (* s pz))
					py
					(+ (* -s px)
					   (* c pz)))

				       )
				    (Z ((- (* c px)
					   (* s py))
					(+ (* s px)
					   (* c py))
					pz)))))
			  `(do0
			    ,@(loop for e in lr
				    collect
				    (destructuring-bind (name code) e
				      `(def ,(format nil "rotate~a" name) (p a)
					(setf c (np.cos a)
					      s (np.sin a)
					      (ntuple px py pz) p)
					 (return (np.array (list ,@code))))))))
		       (def opU (a b)
			 (string3 "union of two solids")
			 (setf condition (np.tile (< (aref a 1 None)
						     (aref b 1 None))
						  (list 2)))
			 (return (np.where condition a b)))
		     
		       (def sdScene (p)
			 (string3 "Cornell box")
			 (setf (ntuple px py pz) p)
			 ,@(loop for e in l
				 collect
				 (destructuring-bind (object-name object-id &key c code) e
				   (let ((object (string-downcase (format nil "obj_~a" object-name))))
				     `(do0
				       ,(if code
					    code
					    `(comments " "))
				      (setf ,object
					    (df ,(string-upcase (format nil "OBJ_~a" object-name))
						,c)
					    res (opU res ,object))
				      ))))
			 (return res))
		       (def dist (p)
			 (return (aref (sdScene p) 1)))
		       (def calcNormalWithAutograd (p)
			 (return (normalize "grad(dist)(p)")))
		       (def sampleCosineWeightedHemisphere (rng_key n)
			 (setf (ntuple rng_key subkey ) (random.split rng_key)
			       u (random.uniform subkey :shape (tuple 2)
							:minval 0
							:maxval 1)
			       (ntuple u1 u2) u)
			 (setf uu (normalize (np.cross n (np.array (list 0.0 1.0 1.0))))
			       vv (np.cross uu n)
			       ra (np.sqrt u2)
			       rx (* ra (np.cos (* 2 np.pi u1)))
			       ry (* ra (np.sin (* 2 np.pi u1)))
			       rz (np.sqrt (- 1.0 u2))
			       rr (+ (* rx uu)
				     (* ry vv)
				     (* rz n)))
			 (return (normalize rr)))

		       #+nil
		       (do0
			(setf RNG_KEY (random.PRNGKey 0))
			(setf nor (normalize (np.array (list 1.0 1.0 0)))
			      nor (np.tile nor (list 1000 1))
			      rng_key (random.split RNG_KEY 1000))
			(setf rd "vmap(sampleCosineWeightedHemisphere)(rng_key,nor)")
			(setf fig (plt.figure :figsize (list 8 4))
			      
			      )
			(do0
			 (setf ax (fig.add_subplot 121 :projection (string "3d")))
			 (ax.scatter (aref rd ":" 0)
				     (aref rd ":" 2)
				     (aref rd ":" 1)
				     :alpha .3)
			 (plt.xlabel (string "x"))
			 (plt.ylabel (string "y"))
			 (ax.set_zlabel (string "z"))
			 ;(ax.set_aspect (string "equal"))
			 )
			(do0
			 (setf ax (fig.add_subplot 122))
			 (ax.scatter (aref rd ":" 0)
				     (aref rd ":" 1)
				     :alpha .3)
			 (plt.xlabel (string "x"))
			 (plt.ylabel (string "z"))
			 (plt.grid)
			 (ax.set_aspect (string "equal")))
			(plt.suptitle (string "cos sampling"))
			(plt.tight_layout :rect (list 0 0 1 .98))
			(fig.savefig (string "cos_sampling.png")
				     :pad_inches 0
				     :bbox_inches (string "tight")))
		       (do0
			(comments "perspective pinhole camera with 2.2m focal distance")
			(setf N 64
			      xs (np.linspace 0 1 N)
			      (ntuple us vs) (np.meshgrid xs xs)
			      uv (dot (np.vstack (list (us.flatten)
						       (vs.flatten)))
				      T)
			      )
			(comments "normalize pixel locations to [-1..1]")
			(setf p (np.concatenate (list (+ -1 (* 2 uv))
						      (np.zeros (tuple (* N N)
								       1)))
						:axis 1))
			(setf eye (np.tile (np.array (list 0 2.0 -3.5)
						     )
					   (tuple (aref p.shape 0)
						  1))
			      look (np.array (list 0 2.0 0))
			      vn (vmap normalize)
			      w (vn (- look eye))
			      up (np.array (list 0 1.0 0))
			      u (vn (np.cross w up))
			      v (vn (np.cross u w))
			      d 2.2
			      rd (vn (+ (* (aref p ":" 0 None)
					   u)
					(* (aref p ":" 1 None)
					   v)
					(* d w)))))
		       )))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

