(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)

;; https://github.com/sciapp/python-gr/tree/master/examples
(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/73_jax_bfgs")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (pd pandas)
      jax
      
      (xr xarray)
      matplotlib
					;(s skyfield)
      ;;(ds dataset)
					; cv2
      ;datoviz
      ))
  (defun scale (a &key (percentile 2) (range 10))
	  "Set mi1 and ma1 for robust display. percentile [default=2%] gives the range of samples that will be ignored at the extrema. range [default=10%] increases the range a bit."
	  `(do0
	    (setf mi0 (np.nanpercentile ,a ,percentile)
		  ma0 (np.nanpercentile ,a ,(- 100 percentile))
		  d (- ma0 mi0)
		  mi1 (- mi0 (/ (* d ,range) 100))
		  ma1 (+ ma0 (/ (* d ,range) 100))))
	  )
  (let ((nb-file "source/02_play.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play02"))
      (python
       (do0
	"#export"
	#+nil (do0
	 (import os)
	 (setf 
	  (dot os (aref environ (string "MPLBACKEND"))
		      )
	  (string "module://gr.matplotlib.backend_gr"))
	 (imports-from (pylab      plot draw pause)
		       )
	 (imports (gr)))
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
			  (xrp xarray.plot)
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
			  jax.numpy.linalg
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
	(display df_status)))

      (python
       
       (do0
	(setf dtype np.float32)
	(setf nx 64
	      ny 43
	      nz 32
	      nch 3
	      x (np.linspace 0 4 nx :dtype dtype) 
	      y (np.linspace -1 1 ny          :dtype dtype)
	      z (np.linspace -1 1 nz :dtype dtype)
	      ch (np.arange nch :dtype dtype)
	      xx (aref x ":" None None None)
	      yy (aref y None ":" None None)
	      zz (aref z None None ":" None)
	      cc (aref ch None None None ":")
	      data (+
		    (np.exp (/ (- (** (- zz
					 .3)
				      2))
			       -.1))
		    (np.cos yy)
		    (np.sin (+ xx
			       (* .33 cc))))
	      data_std .052 
	      noise (np.random.normal :loc 0.0
				      :scale data_std
				      :size data.shape)
	      )
	,(let ((dim-def `((:name x :contents x)
			  (:name y :contents y)
			  (:name z :contents z)
			  (:name ch :contents ch))))
	   `(setf xs (xr.DataArray
		     :data (+ data noise)
		     :dims (list ,@(mapcar (lambda (x)
					     `(string ,(getf x :name)))
					   dim-def))
		     :coords (dict
			      ,@(mapcar (lambda (x)
					  `((string ,(getf x :name))
					    ,(getf x :contents)))
					dim-def)))))
	xs
	))
      (python
       (do0
	(comments "1d plot")
	(setf ch 0
	      z 16
	      y 20)
	(setf da (aref xs ":" y z ch))
	(da.plot)
	(grid)))
      (python
       (do0
	(comments "2d plot")
	(setf ch 0
	      z 16)
	(setf da (aref xs ":" ":" z ch))
	
	,(scale `da)
	(do0
	 (da.plot :vmin mi1 :vmax ma1
		  :cmap (string "cubehelix"))
	 (plt.clabel
	  (xrp.contour da
		       :levels (np.linspace mi1 ma1 6)
		       :colors (string "k"))))
	))
      
      ,@(let* ((x-poly `(0 1 2 3 4 5))
	       (params `(
			 ,@(loop for i in x-poly
				 collect
				 `(:name ,(format nil "xp_a~a" i)
				   :start .1))
			 (:name xp_c0 :start 1.5)
			 ))
	       (fun-model `(+ ,@(loop for i in x-poly
				      collect
				      `(* ,(format nil "xp_a~a" i)
					  (** (- xx xp_c0) ,i)))))
	       (fun-residual `(/ (- data ,fun-model)
				 data_std))
	       (merit-args `(data data_std xx)))
	`((python
	   (do0
	    "#export"
	    
	    (setf x0 (jnp.array (list ,@(loop for e in params
					      and i from 0
					      collect
					      (destructuring-bind (&key name start) e
						start))))
		  x0_d (dot (aref x0
			      None	;x
			      None	;y
			      None	;z
			      None	;ch
			      ":"
				  )
			    (tile
			     (list
			      1 ;nx
			      ny
			      nz
			      nch
			      1))))
	    ,@(loop for e in `(x0_d xx yy zz cc)
		    collect
		    `(print (dot (string ,(format nil "~a.shape={}" e))
				 (format (dot ,e shape)))))
	    (setf res (list))
	    (def merit (params ,@merit-args)
	      ,@(loop for e in params
		      and i from 0
		      collect
		      (destructuring-bind (&key name start) e
			`(setf ,name (aref params ,i))))
	      (return (jnp.sum (** ,fun-residual
				   2))))

	    (setf mesh_devices (np.array (jax.devices))
		  mesh_def (tuple mesh_devices
				  (tuple (string "cpu"))))
	    (def optimize_and_error_estimate (x0 ,@merit-args)
	      (setf sol (jax.scipy.optimize.minimize
			 merit
			 x0
			 :args (tuple ,@merit-args)
			 :method (string "BFGS"))
		    )
	      (setf hessian_fun (jax.jacfwd (jax.jacrev
					     (lambda (p) (merit p ,@merit-args)))))
	      (setf hessian_mat (hessian_fun sol.x))
	      (setf sol_err (jnp.sqrt (jnp.diag (jnp.linalg.inv hessian_mat))))
	      (return (jnp.hstack (tuple sol.x
					 sol_err))))
	    (do0
	     (comments "call 1d optimization")
	     (setf data1 (aref data
							 ":"
							 (// ny 2)
							 (// nz 2)
							 (// ch 2)))
	     (setf sol (optimize_and_error_estimate x0
						   data1
						   data_std
						   (dot xx (squeeze))))
	     (do0 (plot (dot xx (squeeze)) data1)
		  ,@(loop for e in params
		      and i from 0
		      collect
		      (destructuring-bind (&key name start) e
			`(setf ,name (aref sol ,i))))
		  (setf model1 ,fun-model)
		  (plot (dot xx (squeeze)) (dot model1 (squeeze)))
		  (grid))
	     )
           #+nil ,(let* ((all-axes `(y z ch))
		    (in-axes `(dict
			       ,@(loop for e in all-axes
				       and i from 1
				       collect
				       `(,i (string ,e))))))
	       `(do0 (setf 
	 	      distr_jv
		       (xmap 
			optimize_and_error_estimate
			:in_axes (tuple ,in-axes ; data
					"{}"	 ; data_std
					,in-axes ; x
					)
			:out_axes (list ,@(loop for e in all-axes
						collect
						`(string ,e))
					"...")
			; :axis_resources (dictionary :x (string "cpu"))
			))))

	    #+nil ,@(loop for i below 2 collect
		    `(do0
		      ,(if (eq i 0)
			   `(print (string "initialize jit"))
			   `(print (string "call jit second time")))
		      (setf ,(format nil "jv~a_start" i) (time.time))
		      (with (jax.experimental.maps.mesh *mesh_def)
			    (setf ,(if (eq i 0)
				       `sol0
				       `sol)
				  (distr_jv x0_d ,@merit-args))
			    )
		      (setf ,(format nil "jv~a_end" i) (time.time))))
	  
	  #+nil
	    (do0
	     (setf d (dictionary :duration_jit_and_call (- jv0_end jv0_start)
				 :duration_call (- jv1_end jv1_start)))
	     ,@(loop for e in params and i from 0
		     collect
		     (destructuring-bind (&key name start) e
		       `(do0
			 (setf (aref d (string ,name))
			       (dot 
				    (aref sol ":" ,i)
				    (mean)
				    (item)))
			 (setf (aref d (string ,(format nil "~a_err" name)))
			       (dot 
				    (aref sol ":" ,i)
				    (std)
				    (item)))
			 (setf (aref d (string ,(format nil "~a_fiterr" name)))
			       (dot 
				    (aref sol ":" ,(+ (length params) i))
				    (mean)
				    (item)))
			 (setf ,name
			       (aref d (string ,name)))
			 (setf (aref d (string ,(format nil "~a0" name)))
			       (dot
				(aref x0 ,i)
				(item))))))
	     #+nil
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
	     (setf df (pd.DataFrame res))
	     (print (dot df (aref iloc 0)))
	     )
	    )
	   #+nil
	   (do0
	    (figure :figsize (list 12 10))
	    (setf pl (list 3 1))
	    (do0 
	     (subplot2grid pl (list 0 0))
	     (plot x y0 :linewidth 4
		   	:label (string "noise-free data")
			:color (string "green"))
	     (for (i (range n_batch))
		  (plt.scatter x (aref y_2d ":" i) :label (? (== i 0)
							     (string "data with noise")
							     (string "_no_legend_"))
						   :alpha .09
						   :color (string "r")))
	     (plot x ,fun-model :linewidth 2 :label (string "fit avg")
		   :color (string "red"))
	     (for (i (range n_batch))
		  ,@(loop for e in params and i from 0
			  collect
			  (destructuring-bind (&key name start) e
			    `(do0
			      (setf ,name
				    (dot 
					 (aref sol i ,i)
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
	     (setf model ,fun-model)
	     (setf noisefree_residual (- y0 model))
	     (setf noisy_residual (- y model))
	     (plot x noisefree_residual
		   :label (string "residual noise-free data"))
	     (plt.scatter x noisy_residual
			  :label (string "residual data with noise")
			  :alpha .3)
	     (legend)
	     (grid))
	    (do0 
	     (subplot2grid pl (list 2 0))
	     (setf noisefree_residual_weighted (/ (- y0 model)
						  delta_y))
	     (setf noisy_residual_weighted (/ (- y model)
					      delta_y))
	     (plot x noisefree_residual_weighted
		   :label (string "weighted residual noise-free data"))
	     (plt.scatter x noisy_residual_weighted
			  :label (string "weighted residual data with noise")
			  :alpha .3)
	     (legend)
	     (grid))
	    )
	    )))
      ))))



