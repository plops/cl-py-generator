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
			   (np jax.numpy)
			   jax
			   jax.random
			   jax.config
			   ))
		 "from jax import grad, jit, jacfwd, jacrev, vmap, lax"
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
		  (def raymarch (ro rd sdf_fn &key (max_steps 10))
		    (setf tt 0.0)
		    (for (i (range max_steps))
			 (setf p (+ ro (* tt rd))
			       tt (+ tt (sdf_fn p))))
		    (return tt))
		  ,(let ((l `((none 0.0)
			      (floor .1)
			      (ceil .2)
			      (wall_rd .3)
			      (wall_wh .4)
			      (wall_gr .5)
			      (short_block .6)
			      (tall_block .7)
			      (light 1.0)
			      (sphere .9))))
		     `(do0
		       (def df (obj_id dist)
			 (return (np.array (list obj_id dist))))
		       (def udBox (p b)
			 (comments "b .. half-widths")
			 (return (length (np.maximum (- (np.abs p)
							b)
						     0.0))))
		       ,(let ((lr `((X (px (- (* c py)
					      (* s pz))
					   (+ (* s py)
					      (* c pz))))
				    (Y ((+ (* c px)
					   (* s pz)))
				       )))))
		       )))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

