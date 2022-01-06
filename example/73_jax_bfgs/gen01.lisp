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
      (python
       (do0
	"#export"
	
	(comments "spread jax work over 4 virtual cpu cores:")
	(imports (os
		  multiprocessing))
	(setf cpu_count (multiprocessing.cpu_count))
	(print (dot (string "jax will spread work to {} cpus")
		    (format cpu_count)))
	(setf (aref os.environ (string "XLA_FLAGS"))
	      (dot (string "--xla_force_host_platform_device_count={}")
		   (format cpu_count)))))
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
			  jax.scipy.optimize
			  jax.experimental.maps
			  jaxopt
			  numpy.random
			  ))
		
			(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
			(imports-from (jax.experimental.maps
				       xmap))

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

     #+nil ,(let* ((params `((:name phase :start .1)
			(:name amplitude :start 1.0)
		       (:name offset :start 0.0)))
	      (fun-model `(+ offset
			     (* amplitude (jnp.sin (+ phase x)))))
	      (fun-residual `(- y ,fun-model)))
	`(python
	 (do0
	  "#export"
	  (comments "try GradientDescent from jaxopt")
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

     #+nil 
      ,(let* ((params `((:name phase :start .1)
			(:name amplitude :start 1.0)
		       (:name offset :start 0.0)))
	      (fun-model `(+ offset
			     (* amplitude (jnp.sin (+ phase x)))))
	      (fun-residual `(- y ,fun-model)))
	`(python
	 (do0
	  "#export"
	  (comments "try minimize from jax")
	  (setf res (list))
	  (def merit1 (params y x)
	    ,@(loop for e in params
		    and i from 0
		    collect
		    (destructuring-bind (&key name start) e
		      `(setf ,name (aref params ,i))))
	    (return (jnp.sum (** ,fun-residual
				 2))))
	  (do0 (setf x0 (jnp.array (list ,@(loop for e in params
							  and i from 0
							  collect
							  (destructuring-bind (&key name start) e
							    start))))
		     sol (jax.scipy.optimize.minimize
			  merit1
			  x0
			  :args (tuple y x)
			  :method (string "BFGS"))))
	  (do0
	   (setf d "{}")
	   ,@(loop for e in params and i from 0
		   collect
		   (destructuring-bind (&key name start) e
		    `(do0
		      (setf (aref d (string ,name))
			    (dot sol
				 (aref x ,i)
				 (item)))
		      (setf ,name
			    (dot sol
				 (aref x ,i)
				 (item)))
		      (setf (aref d (string ,(format nil "~a0" name)))
			    (dot
			     (aref x0 ,i)
			     (item))))))
	   ,@(loop for e in `(success status fun nfev njev nit)
		   collect
		   `(setf (aref d (string ,e))
			  (dot sol
			       ,e
			       (item))))
	   (res.append d)
	   (setf df1 (pd.DataFrame res))
	   (print df1)
	   )
	  (do0
	   (figure :figsize (list 12 8))
	   (setf pl (list 2 1))
	   (do0 
	    (subplot2grid pl (list 0 0))
	    (plot x y0 :label (string "noise-free data"))
	    (plt.scatter x y :label (string "data with noise"))
	    (plot x ,fun-model :label (string "fit"))
	    (legend)
	    (grid))
	   (do0 
	    (subplot2grid pl (list 1 0))
	    (plot x (- y0 ,fun-model) :label (string "residual noise-free data"))
	    (plt.scatter x (- y ,fun-model) :label (string "residual data with noise"))
	    (legend)
	    (grid))
	   )
	  )))

     #+nil,(let* ((params `((:name phase :start .1)
			(:name amplitude :start 1.0)
		       (:name offset :start 0.0)))
	      (fun-model `(+ offset
			     (* amplitude (jnp.sin (+ phase x)))))
	      (fun-residual `(- y ,fun-model)))
	`(python
	 (do0
	  "#export"
	  (comments "try minimize many replicas with different noise with jax and vmap")
	  (setf n_batch 1000)
	  (def make2d (a)
	    (return (dot a
			 (repeat n_batch)
			 (reshape -1 n_batch))))
	  (setf nx 120
		x (np.linspace 0 4 nx)
		
		x_2d (make2d x)
		y0 (np.sin x)
		y0_2d (make2d y0) ;; 100x120
		noise_2d (np.random.normal :loc 0.0 :scale .1  :size y0_2d.shape) 
		y_2d (+ y0_2d
			noise_2d)
		)
	  (setf x0 (jnp.array (list ,@(loop for e in params
						       and i from 0
						       collect
						       (destructuring-bind (&key name start) e
							 start))))
		x0_2d (make2d x0))
	  (setf res (list))
	  (def merit2 (params y x)
	    ,@(loop for e in params
		    and i from 0
		    collect
		    (destructuring-bind (&key name start) e
		      `(setf ,name (aref params ,i))))
	    (return (jnp.sum (** ,fun-residual
				 2))))
	  (do0 (setf 
		jv #+nil(jax.jit
			 (jax.pmap
			  (jax.vmap (lambda (x0 y x)
				      (jax.scipy.optimize.minimize
				       merit2
				       x0
				       :args (tuple y x)
				       :method (string "BFGS")))
				    :in_axes (tuple 1 1 1)
				    :out_axes 0
				    )
			  :in_axes (tuple 1 1 1)
			  :out_axes 0)
			 )
		   #-nil(jax.jit
			 (jax.vmap (lambda (x0 y x)
				     (jax.scipy.optimize.minimize
				      merit2
				      x0
				      :args (tuple y x)
				      :method (string "BFGS")))
				   :in_axes (tuple 1 1 1)
				   :out_axes 0)
		    )
		#+nil(jax.vmap (lambda (x0 y x)
				     (jax.scipy.optimize.minimize
				      merit2
				      x0
				      :args (tuple y x)
				      :method (string "BFGS")))
				   :in_axes (tuple 1 1 1)
				   :out_axes 0)
		     ))

	  
	  (do0
	   (do0 (setf jv0_start (time.time))
	    (setf 
	     sol (jv x0_2d y_2d x_2d))
	    (setf jv0_end (time.time)))
	   (setf jv_start (time.time))
	   (setf 
	    sol (jv x0_2d y_2d x_2d))
	   (setf jv_end (time.time)))
	  
	  (do0
	   (setf d (dictionary :duration_jit_and_call (- jv0_end jv0_start)
			       :duration_call (- jv_end jv_start)))
	   ,@(loop for e in params and i from 0
		   collect
		   (destructuring-bind (&key name start) e
		    `(do0
		      (setf (aref d (string ,name))
			    (dot sol
				 (aref x ":" ,i)
				 (mean)
				 (item)))
		      (setf (aref d (string ,(format nil "~a_err" name)))
			    (dot sol
				 (aref x ":" ,i)
				 (std)
				 (item)))
		      (setf ,name
			    (aref d (string ,name)))
		      (setf (aref d (string ,(format nil "~a0" name)))
			    (dot
			     (aref x0 ,i)
			     (item))))))
	   ,@(loop for e in `(success status fun nfev njev nit)
		   collect
		   `(do0
		     (setf (aref d (string ,e))
			   (dot sol
				(aref ,e ":")
				(mean)
				(item)))
		     (setf (aref d (string ,(format nil "~a_err" e)))
			   (dot sol
				(aref ,e ":")
				(std)
				(item)))))
	   (res.append d)
	   (setf df2 (pd.DataFrame res))
	   (print (dot df2 (aref iloc 0)))
	   )
	  (do0
	   (figure :figsize (list 12 8))
	   (setf pl (list 2 1))
	   (do0 
	    (subplot2grid pl (list 0 0))
	    (plot x y0 :label (string "noise-free data"))
	    (for (i (range n_batch))
		 (plt.scatter x (aref y_2d ":" i) :label (? (== i 0)
							    (string "data with noise")
							    (string "_no_legend_"))
						  :alpha .09
			      :color (string "r")))
	    (plot x ,fun-model :label (string "fit avg"))
	    (for (i (range n_batch))
		 ,@(loop for e in params and i from 0
		   collect
		   (destructuring-bind (&key name start) e
		    `(do0
		      (setf ,name
			    (dot sol
				 (aref x i ,i)
				 (mean)
				 (item)))
		      )))
		 (plot x ,fun-model :color (string "k") :label (? (== i 0)
							(string "fit")
								  (string "_no_legend_"))
		       :alpha .1))
	    (legend)
	    (grid))
	   (do0 
	    (subplot2grid pl (list 1 0))
	    (plot x (- y0 ,fun-model) :label (string "residual noise-free data"))
	    (plt.scatter x (- y ,fun-model) :label (string "residual data with noise"))
	    (legend)
	    (grid))
	   )
	  )))

     ,(let* ((params `((:name phase :start .1)
			(:name amplitude :start 1.0)
		       (:name offset :start 0.0)))
	      (fun-model `(+ offset
			     (* amplitude (jnp.sin (+ phase x)))))
	      (fun-residual `(- y ,fun-model)))
	`(python
	 (do0
	  "#export"
	  (comments "try minimize many replicas with different noise with jax and xmap for named axes")
	  (setf n_batch (* 4 10))
	  (def make2d (a)
	    (return (dot a
			 (repeat n_batch)
			 (reshape -1 n_batch))))
	  (setf nx 120
		x (np.linspace 0 4 nx)
		
		x_2d (make2d x)
		y0 (np.sin x)
		y0_2d (make2d y0) ;; 100x120
		noise_2d (np.random.normal :loc 0.0 :scale .1  :size y0_2d.shape) 
		y_2d (+ y0_2d
			noise_2d)
		)
	  (setf x0 (jnp.array (list ,@(loop for e in params
						       and i from 0
						       collect
						       (destructuring-bind (&key name start) e
							 start))))
		x0_2d (make2d x0))
	  (setf res (list))
	  (def merit3 (params y x)
	    ,@(loop for e in params
		    and i from 0
		    collect
		    (destructuring-bind (&key name start) e
		      `(setf ,name (aref params ,i))))
	    (return (jnp.sum (** ,fun-residual
				 2))))

	  (setf mesh_devices (np.array (jax.devices))
		mesh_def (tuple mesh_devices
				(tuple (string "x"))))
	    
	  (do0 (setf 
		distr_jv
		(xmap (lambda (x0 y x)
			(jax.scipy.optimize.minimize
			 merit3
			 x0
			 :args (tuple y x)
			 :method (string "BFGS")))
		      
		      :in_axes (tuple (dict (1 (string "left")))
				      (dict (1 (string "left")))
				      "{}"
				      ;(dict (1 (string "left")))
				      )
		      :out_axes (list (string "left") "...")
		      :axis_resources (dictionary :left (string "x"))
		      )))

	  ,@(loop for i below 2 collect
		  `(do0
		    ,(if (eq i 0)
			 `(print (string "initialize jit"))
			 `(print (string "call jit second time")))
		    (setf ,(format nil "jv~a_start" i) (time.time))
		    (with (jax.experimental.maps.mesh *mesh_def)
			  (setf ,(if (eq i 0)
				     `sol0
				     `sol)
				(distr_jv x0_2d y_2d x))
			  )
		    (setf ,(format nil "jv~a_end" i) (time.time))))
	  
	  
	  (do0
	   (setf d (dictionary :duration_jit_and_call (- jv0_end jv0_start)
			       :duration_call (- jv1_end jv1_start)))
	   ,@(loop for e in params and i from 0
		   collect
		   (destructuring-bind (&key name start) e
		    `(do0
		      (setf (aref d (string ,name))
			    (dot sol
				 (aref x ":" ,i)
				 (mean)
				 (item)))
		      (setf (aref d (string ,(format nil "~a_err" name)))
			    (dot sol
				 (aref x ":" ,i)
				 (std)
				 (item)))
		      (setf ,name
			    (aref d (string ,name)))
		      (setf (aref d (string ,(format nil "~a0" name)))
			    (dot
			     (aref x0 ,i)
			     (item))))))
	   ,@(loop for e in `(success status fun nfev njev nit)
		   collect
		   `(do0
		     (setf (aref d (string ,e))
			   (dot sol
				(aref ,e ":")
				(mean)
				(item)))
		     (setf (aref d (string ,(format nil "~a_err" e)))
			   (dot sol
				(aref ,e ":")
				(std)
				(item)))))
	   (res.append d)
	   (setf df2 (pd.DataFrame res))
	   (print (dot df2 (aref iloc 0)))
	   )
	  (do0
	   (figure :figsize (list 12 8))
	   (setf pl (list 2 1))
	   (do0 
	    (subplot2grid pl (list 0 0))
	    (plot x y0 :label (string "noise-free data"))
	    (for (i (range n_batch))
		 (plt.scatter x (aref y_2d ":" i) :label (? (== i 0)
							    (string "data with noise")
							    (string "_no_legend_"))
						  :alpha .09
			      :color (string "r")))
	    (plot x ,fun-model :label (string "fit avg"))
	    (for (i (range n_batch))
		 ,@(loop for e in params and i from 0
		   collect
		   (destructuring-bind (&key name start) e
		    `(do0
		      (setf ,name
			    (dot sol
				 (aref x i ,i)
				 (mean)
				 (item)))
		      )))
		 (plot x ,fun-model :color (string "k") :label (? (== i 0)
							(string "fit")
								  (string "_no_legend_"))
		       :alpha .1))
	    (legend)
	    (grid))
	   (do0 
	    (subplot2grid pl (list 1 0))
	    (plot x (- y0 ,fun-model) :label (string "residual noise-free data"))
	    (plt.scatter x (- y ,fun-model) :label (string "residual data with noise"))
	    (legend)
	    (grid))
	   )
	  )))
      ))))



