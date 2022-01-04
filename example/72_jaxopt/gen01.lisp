(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/72_jaxopt")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (pd pandas)
      jax
      
					;(xr xarray)
      matplotlib
					;(s skyfield)
      ;;(ds dataset)
					; cv2
      ;datoviz
      ))
  (let ((nb-file "source/01_play.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play01"))
      (python (do0
	       
	       "#export"
	        (do0
					;"%matplotlib notebook"
		 #-nil (do0
		      
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
			  time
					;docopt
					;pathlib
					;(np numpy)
					;serial
					;(pd pandas)
					;(xr xarray)
			  ,@*libs*
					;		(xrp xarray.plot)
			  ;skimage.restoration
			  ;skimage.morphology
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
			  ;scipy.optimize
			  ;scipy.stats
			  ;scipy.special
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
					;re
					;json
					; csv
					;io.StringIO
					;bs4
					;requests
					;(nx networkx)
					;(np jax.numpy)
					;(mpf mplfinance)
			  argparse
			  ;(sns seaborn)
			  ;skyfield.api
			  ;skyfield.data
					;skyfield.data.hipparcos
			  (jnp jax.numpy)
			  jax.config
			  jaxopt
			  numpy.random
			  ))
		
			(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))

		 )
	       ))
      (python
       (do0
	"#export"
	(jax.config.update (string "jax_enable_x64") True)
	(setf
	 _code_git_version
	 (string ,(let ((str (with-output-to-string (s)
			       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		    (subseq str 0 (1- (length str)))))
	 _code_repository (string ,(format nil "~a/~a/~a" *repo-dir-on-github* *example-subdir* nb-file))
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
			    (- tz)))))

	(setf start_time (time.time)
	      debug True)))
      (python
       (do0
	"#export"
	(setf df_status
	      (pd.DataFrame
	       (list
		,@(loop for e in *libs*
			collect
			(cond
			  ((listp e)
			   (destructuring-bind (nick name) e
			     `(dictionary
			       :name (string ,name)
			       :version
			       (dot ,nick __version__)
			       )))
			  ((symbolp e)
			   `(dictionary
			       :name (string ,e)
			       :version
			       (dot ,e __version__)
			       ))
			  (t (break "problem")))))))
	(print df_status)))

      (python
       (do0
	"#export"
	(setf x (np.linspace 0 4 120)
	      y0 (np.sin x)
	      y (+ y0
		   		   (np.random.normal :loc 0.0 :scale .1  :size (len x))))
	#+nil (do0 (plot x y)
	     (grid))))

      ,(let* ((params `((:name phase :start .1)
			(:name amplitude :start 1.0)
		       (:name offset :start 0.0)))
	      (fun-model `(+ offset
			     (* amplitude (jnp.sin (+ phase x)))))
	      (fun-residual `(- y ,fun-model)))
	`(python
	 (do0
	  "#export"
	  (setf res (list))
	  (def merit0 (params y x)
	    ,@(loop for e in params
		    and i from 0
		    collect
		    (destructuring-bind (&key name start) e
		      `(setf ,name (aref params ,i))))
	    (return (jnp.sum (** ,fun-residual
				 2))))
	  (do0 (setf gd (jaxopt.GradientDescent :fun merit0
						:tol 1e-5
						:maxiter 500
						:implicit_diff True)
		     x0 (jnp.array (list ,@(loop for e in params
							  and i from 0
							  collect
							  (destructuring-bind (&key name start) e
							    start))))
		     sol (gd.run x0
				 :x x
				 :y y)))
	  (do0
	   (setf d "{}")
	   ,@(loop for e in params and i from 0
		   collect
		   (destructuring-bind (&key name start) e
		    `(do0
		      (setf (aref d (string ,name))
			    (dot sol
				 (aref params ,i)
				 (item)))
		      (setf ,name
			    (dot sol
				 (aref params ,i)
				 (item)))
		      (setf (aref d (string ,(format nil "~a0" name)))
			    (dot
			     (aref x0 ,i)
			     (item))))))
	   ,@(loop for e in `(iter_num stepsize error "t")
		   collect
		   `(setf (aref d (string ,e))
			  (dot sol
			       state
			       ,e
			       (item))))
	   (res.append d)
	   (setf df (pd.DataFrame res))
	   (print df)
	   )
	  (do0
	   (figure)
	   (plot x y0 :label (string "noise-free data"))
	   (plot x y :label (string "data with noise"))
	   (plot x ,fun-model :label (string "fit"))
	   (legend)
	   (grid)
	   )
	  )))
      ))))



