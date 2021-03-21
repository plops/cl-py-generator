(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/37_jax")
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
					;(plt.ioff)
		  ;;(setf font (dict ((string size) (string 6))))
		  ;; (matplotlib.rc (string "font") **font)
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
			   (jnp jax.numpy)
			   jax
			   jax.random
			   ))
		 "from jax import grad, jit, jacfwd, jacrev"
		 ,(format nil "from jax.numpy import ~{~a~^, ~}" `(sqrt newaxis sinc abs))
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
		  (def tanh (x)
		    (setf y (jnp.exp (* -2s0 x))
			  )
		    (return (/ (- 1s0 y)
			       (+ 1s0 y))))
		  (setf grad_tanh (grad tanh))
		  (print (grad_tanh 1.0) ))

		 (do0
		  
		  (setf nx 32
			ny 27
			x (jnp.linspace -1 1 nx)
			y (jnp.linspace -1 1 ny))
		  (setf q (jnp.sqrt (+ (** (aref x "...,jnp.newaxis") 2)
			      (** (aref y "jnp.newaxis,...") 2))))
		   
		  (setf xs (xr.DataArray :data q
					 :coords (list x y)
					 :dims (list (string "x")

						     (string "y"))))

		  (def model (param &key xs (noise False))
		    (setf (ntuple x0 y0 radius amp) param)
		    (setf res (xs.copy))
		    (setf r (jnp.sqrt (+ (** (+ (aref xs.x.values "...,jnp.newaxis") x0) 2)
					 (** (+ (aref xs.y.values "jnp.newaxis,...") y0) 2))))
		    (setf s (abs (* amp (sinc (/ r radius)))))

		    (when noise
		     (do0 (setf key (jax.random.PRNGKey 0))
			  (setf s (+ s (jax.random.uniform key s.shape)))))
		    (setf res.values s)
		    
		    
		    (return res)
		    )
		  (def model_merit (param &key xs)
		    (setf res (model param :xs xs :noise False))
		    (return (dot (- (res.values.astype jnp.float64)
				    (xs.values.astype jnp.float64))
				 (ravel))))
		  (setf xs_mod (model (tuple .1 -.2 .5 10.0)
				      :xs xs
				      :noise True))
		  )

		 (do0
		  (def jax_model (param x y goal)
		    (setf (ntuple x0 y0 radius amp) param)
		    
		    (setf r (jnp.sqrt (+ (** (+ (aref x "...,jnp.newaxis") x0) 2)
					 (** (+ (aref y "jnp.newaxis,...") y0) 2))))
		    (setf s (abs (* amp (sinc (/ r radius)))))
		    (return (dot (- goal s)
				 (ravel)))))
		 

		 (do0
		   (setf param_opt
		    (scipy.optimize.least_squares model_merit
						  (tuple .12 -.27 .8 13.0)
						  :jac (jit (jacfwd model_merit))
						  ;:gtol None
						  :kwargs (dict ((string "xs") xs_mod)))))
		 (do0
		  (setf xs_fit (model param_opt.x
				      :xs xs))
		  (xrp.imshow (- xs_fit xs_mod)))
		 ))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

