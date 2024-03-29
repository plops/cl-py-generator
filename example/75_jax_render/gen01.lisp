(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/75_jax_render")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (pd pandas)
      jax
      gr
					;(xr xarray)
      ;;matplotlib
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
		 #+nil (do0
		      
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
			  jax.numpy.linalg
			  jax.nn
			  numpy.random
			  ))
		#+nil
			(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
		(imports-from (gr.pygr
			       mlab))
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
	(def volumetric_rendering (color sigma z_vals dirs)
	  (comments "formulate rendering process for semi-transparent volumes that mimic front-to-back additive blending"
		    "as camera ray traverses volume of inhomogeneous material, it accumulates color in proportion to the local color and density of the material at each point along its path")
	  (setf eps 1s-10
		inv_eps (/ 1s0 eps)
		z_right (aref z_vals "..." (slice 1 ""))
		z_left (aref z_vals "..." (slice "" -1))
		dists (- z_right
			 z_left)
		)
	  #+nil (do0
	   (comments "ray relative distance to absolute distance"
		     "not necessary if rays_d is normalized")
	   (setf 
	    dists (* dists (jnp.linalg.norm
			    (aref dirs "..."
				  None (string ":"))
			    :axis -1))))
	  (do0
	   (comments "fraction of light stuck in each voxel")
	   (setf alpha (- 1s0
			  (jnp.exp (* -1
				      (jax.nn.relu sigma)
				      dists)))))
	  (setf accum_prod
		(jnp.concatenate
		 (list (jnp.ones_like (aref alpha
					    "..."
					    (slice "" 1))
				      alpha.dtype)
		       (jnp.cumprod (+ -1s0
				       (aref alpha
					     "..."
					     (slice "" -1))
				       eps)
				    :axis -1))))
	  (do0
	   (comments "absolute amount of light stuck in each voxel")
	   (setf weights (* alpha accum_prod))
	   (setf comp_color (dot
			     (* (aref weights
				      "..."
				      None)
				(jax.nn.sigmoid color))
			     (sum :axis -2))))
	  (do0
	   (comments "weighted average of the depths by contribution to final color")
	   (setf depth (dot (* weights
			       z_left)
			    (sum :axis -1))))
	  (do0
	   (comments "total amount of light absorbed along the ray")
	   (setf acc (dot weights
			  (sum :axis -1)))
	   )
	  (do0
	   (comments "equivalent to disp = 1/max(eps, where(acc>eps,depth/acc,0))"
		     "but more efficient and stable"
		     "to model occlusions the ray accumulates not only color but also opacity"
		     "if accumulated opacity reaches 1 for example when the ray traverses an opaque region then no further color can be accumulated on the ray (not sure if this code behaves like this)")
	   (setf disparity (/ acc depth)
		 disparity (jnp.where (&
				       (< 0 disparity)
				       (< disparity inv_eps)
				       (< eps acc))
				      disparity
				      inv_eps)))
	  #+nil (do0
	   (comments "include white background in final color")
	   (setf comp_color (+ comp_color
			       (- 1s0
				  (aref acc "..." None)))))
	  (return comp_color
		  disparity
		  acc
		  weights)
	  )
	))
      
      ))))



