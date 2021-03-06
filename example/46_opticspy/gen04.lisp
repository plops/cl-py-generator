(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)


;; nbdev_build_lib

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/46_opticspy")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  
  (write-notebook
   :nb-file (format nil "~a/source/04_trace.ipynb" *path*)
   :nb-code
   `(
     (python
      (do0
       "# default_exp ray"))
     (python (do0 "#export"
		  (do0
		   "%matplotlib notebook"
		   #-nil(do0
		     
		  
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
			     pathlib
					;(np numpy)
					;serial
			     (pd pandas)
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
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
			     (opr opticspy.ray_tracing)
					;cProfile
			     (np jax.numpy)
			     jax
			     jax.random
			     jax.config
			     IPython
			   
			     ))
		   "from opticspy.ray_tracing.glass_function.refractiveIndex import *"

		   "from jax import grad, jit, jacfwd, jacrev, vmap, lax, random, value_and_grad"
		   ,(format nil "from jax.numpy import ~{~a~^, ~}" `(sqrt newaxis sinc abs))
		   ,(format nil "from jax import ~{~a~^, ~}" `(grad jit jacfwd jacrev vmap lax random))
		 

		 
		 
		   ,(format nil "from matplotlib.pyplot import ~{~a~^, ~}"
			    `(plot imshow tight_layout xlabel ylabel
				   suptitle contour contourf clabel title subplot subplot2grid grid
				   legend figure gcf xlim ylim))

		   (IPython.core.display.display
		    (IPython.core.display.HTML
		     (string "<style>.container { width:100% !important; }</style>")
		     )
		    )
	       
		   (jax.config.update (string "jax_enable_x64")
				      True)
		   (do0
		    (comments "https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions")
		    (class Timer (object)
			   (def __init__ (self &key (name None))
			     (setf self.name name))
			   (def __enter__ (self)
			     (setf self.tstart (time.time)))
			   (def __exit__ (self type value traceback)
			     (print (dot (string "[{}] elapsed: {}s")
					 (format self.name (- (time.time)
							      self.tstart)))))))
		 
		   )
		  ))

     (python (do0
	      "#export"
	      (do0
	       ,(let ((system-def `((1e9 1e9 air 30)
				    (41.15909 6.09755 S-BSM18_ohara 20  :comment (string "first"))
				    (-957.83146 9.349 air 20)
				    (-51.32 2.032 N-SF2_schott 12)
				    (42.378 5.996 air 12)
				    (1e9 4.065 air 8 :STO True :comment (string "stop"))
				    (247.45 6.097 S-BSM18_ohara 15)
				    (-40.04 85.59 air 15 :comment (string "last"))
				    (1e9 10 air 40)))
		      (l-wl `(656.3 587.6 486.1))
		      (l-wl-name `(red green blue))
		      (system-fn "system.csv"))
		  `(do0
		       
		    (do0
		     (setf system_data (pathlib.Path (string ,system-fn)))
		     (if (dot system_data
			      (exists))
			 (do0
			  (setf df (pd.read_csv (string ,system-fn))))
			 (do0
			  (with (Timer (string "system definition"))
				(setf df
				      (pd.DataFrame
				       (list
					,@(loop for e in system-def
						and i from 1
						collect
						(destructuring-bind (radius thickness material aperture
								     &key (STO 'False) (comment 'None) (output 'True)) e
						  `(dict ,@(loop for e in `(radius thickness material aperture STO output comment)
								 and f in (list radius thickness `(string ,material) aperture STO output comment)
								 collect
								 `((string ,e) ,f))
							 ,@(loop for color in l-wl-name
								 and wl in l-wl
								 and i from 0
								 collect
								 `((string ,(format nil "n_~a" color))
								   (aref (opr.glass_funcs.glass2indexlist
									  (list ,wl)
									  (string ,material))
									 0)
								   )))
						  ))))))
			     
			  (df.to_csv (str system_data))
			  ))
		     df

		     )))
	       )))

     (python (do0
	      "#export"
	      (do0 (comments "accumulate thicknesses (replace first infinity with a finite number for plotting)")
		   (setf xstart -30)
		   (setf (aref df (string "thickness_cum0"))
			 (numpy.cumsum (aref df.thickness.iloc "0:"))
			 #+nil
			 (+ (list xstart)
			    ("list" (+ xstart
				       (numpy.cumsum (aref df.thickness.iloc "1:"))))))
		   (comments "shift thickness_cum0 to have first surface at 0")
		   (setf (aref df (string "thickness_cum"))
			 (- df.thickness_cum0 (dot (aref df.thickness_cum0 (== df.comment (string "first")))
						   (item))))
		   (setf (aref df (string "center_x"))
			 (+ df.radius (- df.thickness_cum df.thickness))
			 )
		   #+nil(setf (aref df (string "aperture")) 20 ;40
			      )

		   (def compute_arc_angles (row)
		     #+nil (setf theta1 0
				 theta2 360)
		     #-nil (do0 (setf h row.aperture
				      r (numpy.abs row.radius)
				      arg (/ h r)
				      theta_ (numpy.rad2deg (numpy.arcsin arg)))
				(comments "starting and ending angle of the arc. arc is drawn in counter-clockwise direction")
			     
				#-nil (if (< 0 row.radius)
					  (setf
					   theta1 (- 180 theta_)
					   theta2 (+ 180 theta_))
					  (setf
					   theta1 (- 360 theta_)
					   theta2 theta_)))
		     (return (pd.Series (dict ((string "theta1") theta1)
					      ((string "theta2") theta2)))))
		      
		   (do0
		       
		    (setf df_new
			  (df.apply compute_arc_angles
				    :axis (string "columns")
				    :result_type (string "expand"))
			  df (pd.concat (list df df_new)
					:axis (string "columns"))
			    
			  )
		    df))))

     (python
      (do0
       "#export"
       (comments "https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/")
       (comments "https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection")
       "@jit"
       (def eval_ray (tau &key ro rd)
	 (return (+ ro (* tau rd))))
       "@jit"
       (def hit_sphere (&key ro rd sc sr)
	 (do0
	  (string3 "intersection of ray and a sphere, returns two ray parameters tau{1,2}"
		   "rd .. ray direction"
		   "ro .. ray origin, with p(tau) = ro + tau * rd"
		   "sc .. sphere center"
		   "sr .. sphere radius"
		   " a t^2 + b t + c = 0")
	  (setf oc (- sc ro))
	  (comments "rd.oc (in b) is the distance along the ray to the point that is closest to the center of the sphere. if it is negative there will be no intersection (i.e. it would be behind the rays origin)")
	  (setf a 1			;(np.dot rd rd)
		b (* 2 (np.dot rd oc))
		c (- (np.dot oc oc)
		     (** sr 2))
		discriminant (- (** b 2) (* 4 a c)))
		    
	  (comments "two solutions when discriminant > 0 (intersection)")
	  (comments "sign of b controls which tau is the earlier hitpoint.")
	  (setf s .5			;(/ .5s0 a)
		)
	  ;; avoid catastrophic cancellation
	  (setf q (* -.5 (+ b (* (np.sign b) (np.sqrt discriminant))))
		tau1 (/ q a)
		tau0 (/ c q))
	  #-nil (setf tau0 (* s (+ b (np.sqrt discriminant)))
		      tau1 (* s (- b (np.sqrt discriminant)))
		      )
	  (comments "if both tau positive: ray faces sphere and is intersecting. the smaller tau1 (with b-..) is the first intersection")
	  (comments "if one tau is positive and one is negative, then ray is shooting from inside the sphere (and we want the positive)")
	  #+nil
	  (do0 (setf tau tau0)
	       (when (and (< 0 tau0)
			  (< 0 tau1))
		 (setf tau tau1))
	       (when (and (< (* tau1 tau0) 0))
		 (if (< 0 tau0)
		     (setf tau tau0)
		     (setf tau tau1)))
	       (return tau))

	  (do0
		     
	   (setf tau_out (np.where (& (< 0 tau0)
				      (<= tau1 0))
				   tau0
				   tau1
				   ))
	   (return tau_out))

	  #+nil 
	  (do0 (setf p0 (eval_ray tau0 :ro ro :rd rd)
		     p1 (eval_ray tau1 :ro ro :rd rd)
		     p (eval_ray tau :ro ro :rd rd))

	       #+nil ,@(loop for e in `(a b c ro rd oc sc tau0 tau1 tau p0 p1 p)
			     collect
			     `(print (dot (string ,(format nil "~a={}" e))
					  (format ,e)))))
					;(return tau0)
	  ;; outside-directed normal at intersection point is P_hit - Sphere_Center
	  ))
       #+nil
       (print
	(hit_sphere :ro (np.array (list -10 0 0))
		    :rd (np.array (list 1 0 0))
		    :sc (np.array (list 0 0 0))
		    :sr 1))
       "@jit"
       (def sphere_normal_out (&key p_hit sc sr)
	 (return (/ (- p_hit sc) sr)))
       "@jit"
       (def snell (&key rd n ni no)
	 ;;https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
	 (string3 "rd .. ray direction"
		  "n  .. surface normal"
		  "ni .. refractive index on input side"
		  "no .. refractive index on output side")
	 (setf u (/ ni no)
	       p (np.dot rd n))
	 (return (+ (* u
		       (- rd (* p n)))
		    (* (np.sqrt (- 1 (* (** u 2)
					(- 1 (** p 2)))))
		       n))))

		  
       ))
 
     (python
      (do0
       
       (def trace (&key df ro rd (start 1) (end None))
	 (string3 "trace ray (ro,rd) through the system defined in df. start and end define the surfaces to consider. ")
         (do0
	  (if (is end None)
	      (setf end (len df))
	      (setf end (+ end 1)))
	  #-nil (setf
		 rd (/ rd (np.linalg.norm rd)))
	  (setf res (list))
	  (for (surface_idx (range start end))
	       (do0
			
		(setf sc (np.array (list (aref df.center_x surface_idx) 0 0))
		      sr (aref df.radius surface_idx))
		(setf tau (hit_sphere :ro ro :rd rd
				      :sc sc
				      :sr sr))
	      
		,@(loop for e in `(tau	; tau2
				   ) and color in `(k)
			collect
			`(do0 (setf p1 (eval_ray ,e :ro ro :rd rd)) ;; hit point
			      (setf n (sphere_normal_out :p_hit p1 :sc sc :sr sr))
			      (setf normal (* -1 n)
				    ni (aref df.n_green (- surface_idx 1))
				    no (aref df.n_green (- surface_idx 0)))
			      (setf rd_trans (snell :rd rd
						    :n normal
						    :ni ni
						    :no no))
			      (setf dist (np.linalg.norm (- p1 ro))
				    opt_distance (* dist ni))
			      (res.append (dictionary :surface_idx surface_idx
						      :ro ro
						      :rd rd
						      :tau tau
						      :phit p1
						      :normal normal
						      :rd_trans rd_trans
						      :sc sc
						      :sr sr
						      :ni ni
						      :no no
						      :distance dist
						      :optical_distance opt_distance))
			      (setf rd rd_trans
				    ro p1)
			      ))))
	  (setf df (pd.DataFrame res))
	  (setf (aref df (string "optical_distance_cum"))
			 (numpy.cumsum (aref df.optical_distance.iloc "0:"))
			 #+nil
			 (+ (list xstart)
			    ("list" (+ xstart
				       (numpy.cumsum (aref df.thickness.iloc "1:"))))))
	  (return df)))
       ))

     ,@(let ((l `(radius n_green center_x)))
	   `(
	     (python
	      (do0
	       "#export"
	       (comments "the only required parameters for trace")
	       (setf adf0 (aref df (list ,@ (loop for e in l collect `(string ,e)))))
	       (setf adf (jax.numpy.asarray
			  (dot adf0
			       values)))
	       (comments "first argument selects surface_idx, second argument is column: adf[3,1] is surface_idx=3, column=n_green")
	       adf0))
	     (python
	      (do0
	       "#export"
	       (comments "create a trace function that only uses arrays (so that all computations can be represented with jax)")
	       ;"@jit"
	       (def trace2 (&key adf ro rd (start 1) (end None))
		 (string3 "trace ray (ro,rd) through the system defined in adf. start and end define the surfaces to consider. return the hit point in the last surface")
		 (do0
		  (if (is end None)
		      (setf end (len adf))
		      (setf end (+ end 1)))
		  (setf rd (/ rd (np.linalg.norm rd)))
		  #+nil(setf res (list))
		  (for (surface_idx (range start end))
		       (do0
			(setf sr (aref adf surface_idx ,(position 'radius l))
			      center_x (aref adf surface_idx ,(position 'center_x l))
			      ni (aref adf (- surface_idx 1) ,(position 'n_green l))
			      no (aref adf (- surface_idx 0) ,(position 'n_green l)))	
			(setf sc (np.array (list center_x 0 0)))
			(setf tau (hit_sphere :ro ro :rd rd
					      :sc sc
					      :sr sr))
		       
			(do0 (setf p1 (eval_ray tau :ro ro :rd rd)) ;; hit point
			     (setf n (sphere_normal_out :p_hit p1 :sc sc :sr sr))
			     (setf normal (* -1 n)
				   )
			     (setf rd_trans (snell :rd rd
						   :n normal
						   :ni ni
						   :no no))
			     #+nil
			     (res.append (dictionary :surface_idx surface_idx
						     :ro ro
						     :rd rd
						     :tau tau
						     :phit p1
						     :normal normal
						     :rd_trans rd_trans
						     :sc sc
						     :sr sr
						     :ni ni
						     :no no))
			     (setf rd rd_trans
				   ro p1)
			     )))
		  #+nil (return (pd.DataFrame res))
		  #-nil
		  (return p1)))

	      
	       ;"@jit"
	       (def trace2_op (&key adf ro rd (start 1) (end None))
		 (string3 "trace ray (ro,rd) through the system defined in adf. start and end define the surfaces to consider. return the optical path length between ray origin and the last hit point")
		 (do0
		  (if (is end None)
		      (setf end (len adf))
		      (setf end (+ end 1)))
		  (setf rd (/ rd (np.linalg.norm rd)))
		  (setf op 0d0)
		  (for (surface_idx (range start end))
		       (do0
			(setf sr (aref adf surface_idx ,(position 'radius l))
			      center_x (aref adf surface_idx ,(position 'center_x l))
			      ni (aref adf (- surface_idx 1) ,(position 'n_green l))
			      no (aref adf (- surface_idx 0) ,(position 'n_green l)))	
			(setf sc (np.array (list center_x 0 0)))
			(setf tau (hit_sphere :ro ro :rd rd
					      :sc sc
					      :sr sr))
		       
			(do0 (setf p1 (eval_ray tau :ro ro :rd rd)) ;; hit point
			     (setf op (+ op (* ni (np.linalg.norm (- p1 ro)))))
			     (setf n (sphere_normal_out :p_hit p1 :sc sc :sr sr))
			     (setf normal (* -1 n))
			     (setf rd_trans (snell :rd rd
						   :n normal
						   :ni ni
						   :no no))
			    
			     (setf rd rd_trans
				   ro p1)
			     )))
		 
		  (return op)))
	      
	       ))
	 ))
     
     ,@(progn
	  `((python
	(do0
	 "#export"
	 (def draw (&key df ros rds (start 0) (end None))
	   (string3 "trace rays (list of origins in ros and directions in rds) through a system in dataframe df.")
	   (do0
	    (plt.close (string "all"))
	    (setf fig (figure :figsize (list 9 3))
		  ax (fig.gca)
		  )
	    (grid)
	    (ax.set_aspect (string "equal"))
	    (when (is end None)
	      (setf end (len df)))
	    ,(let ((palette `(black rosybrown lightcoral darkred red tomato sienna darkorange
				    darkkhaki olive yellowgreen darkseagreen springgreen  teal deepskyblue dedgerblue
				    slategray nvay rebeccapurple mediumorchid fuchsia deeppink)))
	       `(do0
		
		 (setf colors (list ,@(loop for e in palette collect `(string ,e))))
		 (for ((ntuple idx row) (dot (aref df.iloc (slice start end)) (iterrows)))
		      (do0		;when (< row.radius 1e8)
		       (setf r (numpy.abs row.radius))
		       (setf x (- (dot row thickness_cum)
				  (dot row thickness)))
		       (ax.add_patch (matplotlib.patches.Arc (tuple row.center_x 0)
							     (* 2 r) (* 2 r)
							     :angle 0
							     :theta1 row.theta1
							     :theta2 row.theta2
							     :alpha 1
							     :linewidth 3
							     :facecolor None
							     :edgecolor (aref colors idx)))
		       ))))
	   
	    (do0

	     (for ((ntuple ro rd1) (zip ros rds) )
		  (setf 
		   rd1 (/ rd1 (np.linalg.norm rd1)))
		  (for (surface_idx (range 1 9))
		       (do0
					;(setf surface_idx 2)
			(setf sc (np.array (list (aref df.center_x surface_idx) 0 0))
			      sr (aref df.radius surface_idx))
			(setf tau (hit_sphere :ro ro :rd rd1
					      :sc sc
					      :sr sr))
		
			,@(loop for e in `(tau ) and color in `(k)
				collect
				`(do0 (setf p1 (eval_ray ,e :ro ro :rd rd1)) ;; hit point
				      (setf n (sphere_normal_out :p_hit p1 :sc sc :sr sr))
				      (setf rd_trans (snell :rd rd1
							    :n (* -1 n)
							    :ni (aref df.n_green (- surface_idx 1))
							    :no (aref df.n_green (- surface_idx 0))))
				     
				
				      (plot (list (aref ro 0)
						  (aref p1 0)
				
						  )
					    (list (aref ro 1)
						  (aref p1 1)
				
						  )
					    :color (string ,color)
					    :alpha .3
					    )
				      (setf rd1 rd_trans
					    ro p1)
				 ))))))
	    (xlim (tuple -35 125))
	    (ylim (tuple -50 50))
	    (xlabel (string "x (mm)"))
	    (ylabel (string "y (mm)"))))))

	    
	    ))

     ,@(progn
	 `( (markdown "add more parameters to the chief2 function to vary field position and the target point in the pupil. the new function can find more than just chief rays, so the name changes to `into_stop`")
	   (python
       (do0
	"#export"
	;"@jit"
	(def into_stop (&key (ro_x -20) (ro_y 10) (ro_z 0) (theta 1) (phi 0))
	   (string "trace from field point into stop. this can be used to find theta that corresponts to the chief ray or pairs of theta and phi that belong to marginal rays. Angles are in degree.")
	   (setf theta_ (np.deg2rad theta)
		phi_ (np.deg2rad phi))
	   (setf phit (trace2 :adf adf
			     :ro (np.array (list ro_x ro_y ro_z))
			     :rd (np.array (list (np.cos theta_)
						 (* (np.cos phi_)
						    (np.sin theta_))
						 (* (np.sin phi_)
						    (np.sin theta_))))
			     :end 5))
	   (comments "the stop plane fixes the x coordinate, only return y and z")
	   (return (aref phit "1:")))
	
	))
	    
	    ))

     ,@(progn
	   `((python
	      (do0
	       "#export"
	       (setf theta 5
		     phi 15
		     ro_z 0)
	       
	       (setf into_stop_chief
		     (jit (value_and_grad
			   (lambda (x)
			     (aref  (into_stop :ro_y x :ro_z ro_z :theta theta :phi phi)
				    ;; i only care about the y coordinate because phi=0
				    0)
			     )
			   :argnums 0)))))
	     (python
			 (do0
			  (setf sol_chief
				(scipy.optimize.root_scalar
				 into_stop_chief
				 :method (string "newton")
				 :x0 0
				 :fprime True
				 ))
			  sol_chief)
			 )
	     
	     (markdown "coordinate system. (theta,phi,ro_x) defines illumination angle and field position")
	     (python
	      ,(let ((code `(do0 (setf theta_ (np.deg2rad theta)
			       phi_ (np.deg2rad phi))
			 (comments "create a direction perpendicular to chief ray in the tangential plane of the chief ray")
			 (setf chief_rd (np.array (list (np.cos theta_)
							(* (np.cos phi_)
							   (np.sin theta_))
							(* (np.sin phi_)
							   (np.sin theta_)))))
			 (if tan
			     (setf sideways_rd (np.array (list 0.0 0 1)))
			     (setf sideways_rd (np.array (list 0.0 1 0))))
			 (setf rd (np.cross sideways_rd
					    chief_rd
					    ))
			 (setf new_ro (eval_ray tau
						:ro chief_ro
						:rd rd)))))
		`(do0
	       
	       
		 (def into_stop_parallel_to_chief (&key tau chief_ro theta phi (tan True))
		   (do0
		    ,code
		    
		    (return (into_stop :ro_x (aref new_ro 0)
				       :ro_y (aref new_ro 1)
				       :ro_z (aref new_ro 2)
				       :theta theta
				       :phi phi))
		    ))

		 (def rays_parallel_to_chief (&key tau chief_ro theta phi (tan True))
		   (do0
		    ,code		 
		    (return (dictionary :ro new_ro
					:tau_rd rd
					:chief_rd chief_rd
					:tau tau
					:tan tan))
		    ))
		 (def rays_parallel_to_chief2 (&key tau_tan tau_sag chief_ro theta phi (tan True))
		   (do0
		    (do0 (setf theta_ (np.deg2rad theta)
			       phi_ (np.deg2rad phi))
			 (comments "create a direction perpendicular to chief ray in the tangential plane of the chief ray")
			 (setf chief_rd (np.array (list (np.cos theta_)
							(* (np.cos phi_)
							   (np.sin theta_))
							(* (np.sin phi_)
							   (np.sin theta_)))))
			 (setf sideways_rd_tan (np.array (list 0.0 0 1)))
			 (setf sideways_rd_sag (np.array (list 0.0 1 0)))
			 (setf rd_sag (np.cross sideways_rd_sag
						chief_rd
						)
			       rd_tan (np.cross sideways_rd_tan
						chief_rd
						))
			 (setf new_ro_sag (eval_ray tau_sag
						:ro (np.array (list 0.0 0 0))
						:rd rd_sag))
			 (setf new_ro_tan (eval_ray tau_tan
						:ro chief_ro
						:rd rd_tan))
			 (setf new_ro (+ new_ro_sag
					 new_ro_tan)))		 
		    (return new_ro #+nil (dictionary :ro new_ro
					:tau_rd_tan rd_tan
					:tau_rd_sag rd_sag
					:chief_rd chief_rd
					:tau_sag tau_sag
					:tau_tan tau_tan))
		    ))
	       
	       
		 (setf into_stop_coma_tan
		       (jit (value_and_grad
			     (lambda (tau target)
			       (- (aref (into_stop_parallel_to_chief
					 :tau tau
					 :chief_ro (np.array (list -20
								   (sol_chief.root.item)
								   0))
					 :theta theta
					 :phi phi
					 :tan True)
					0)
				  target)
			       )
			     :argnums 0)))
		 (setf into_stop_coma_sag
		       (jit (value_and_grad
			     (lambda (tau target)
			       (- (aref (into_stop_parallel_to_chief
					 :tau tau
					 :chief_ro (np.array (list -20
								   (sol_chief.root.item)
								   0))
					 :theta theta
					 :phi phi
					 :tan False)
					0)
				  target)
			       )
			     :argnums 0))))))
	    
	     ,@(let ((l `((coma_up :tan True :target  (aref df.aperture 5))
			  (coma_lo :tan True :target  (* -1 (aref df.aperture 5)))
			  (coma_up :tan False :target  (aref df.aperture 5))
			  (coma_lo :tan False :target  (* -1 (aref df.aperture 5)))
			 )))
		 `(,@(loop for e in l
			   collect
			   (destructuring-bind (name &key (tan `True) (target 0.0)) e
			    `(python
			      (do0
			       (setf ,(format nil "tau_~a_~a" (if (eq tan 'True) "tan" "sag") name)
				     (scipy.optimize.root_scalar 
							  ,(format nil "into_stop_coma_~a" (if tan "tan" "sag"))
							  :args (tuple ,target)
							  :method (string "newton")
							  :x0 ,target
							  :fprime True))
			       ,(format nil "tau_~a_~a" (if (eq tan 'True) "tan" "sag") name)))))))
	     
	     
	     (python
	      (do0
	       (setf chief_ro (np.array (list -20
					      (sol_chief.root.item)
					      0)))
	       (setf n 32)
	       (setf taus_tan (np.concatenate
			   (tuple (np.linspace (dot tau_tan_coma_lo root (item))
					       0
					       (// n 2))
				  (aref (np.linspace 0
						     (dot tau_tan_coma_up root (item))
						     (// n 2))
					"1:"))))
	       (setf taus_sag (np.concatenate
			   (tuple (np.linspace (dot tau_sag_coma_lo root (item))
					       0
					       (// n 2))
				  (aref (np.linspace 0
						     (dot tau_sag_coma_up root (item))
						     (// n 2))
					"1:"))))
	       (setf ros (list)
		     rds (list)
		     res (list))
	       (for (tau taus_tan)
		    (setf d (rays_parallel_to_chief :tau tau
						    :chief_ro chief_ro
						    :theta theta
						    :phi phi
						    :tan True))
		    (setf pupil (trace2 :adf adf
				  :ro (aref d (string "ro"))
				  :rd (aref d (string "chief_rd"))
				  :start 1
				  :end 5))
		    (setf (aref d (string "pupil"))
			  pupil)
		    (setf (aref d (string "pupil_y"))
			  (aref pupil 1))
		    (setf (aref d (string "pupil_y_normalized"))
			  (/ (aref pupil 1)
			     (aref df.aperture 5)))
		    (setf (aref d (string "op"))
			  (dot (trace2_op :adf adf
				      :ro (aref d (string "ro"))
				      :rd (aref d (string "chief_rd"))
					  )
			       (item)))
		    (res.append d)
		    (ros.append (aref d (string "ro")))
		    (rds.append (aref d (string "chief_rd"))))
	       (for (tau taus_sag)
		    (setf d (rays_parallel_to_chief :tau tau
						    :chief_ro chief_ro
						    :theta theta
						    :phi phi
						    :tan False))
		    (setf pupil (trace2 :adf adf
				  :ro (aref d (string "ro"))
				  :rd (aref d (string "chief_rd"))
				  :start 1
				  :end 5))
		    (setf (aref d (string "pupil"))
			  pupil)
		    (setf (aref d (string "pupil_y"))
			  (aref pupil 1))
		    (setf (aref d (string "pupil_z"))
			  (aref pupil 2))
		    (setf (aref d (string "pupil_y_normalized"))
			  (/ (aref pupil 1)
			     (aref df.aperture 5)))
		    (setf (aref d (string "pupil_z_normalized"))
			  (/ (aref pupil 2)
			     (aref df.aperture 5)))
		    (setf (aref d (string "op"))
			  (dot (trace2_op :adf adf
				      :ro (aref d (string "ro"))
				      :rd (aref d (string "chief_rd"))
					  )
			       (item)))
		    (res.append d)
		    (ros.append (aref d (string "ro")))
		    (rds.append (aref d (string "chief_rd"))))
	       (setf df_taus (pd.DataFrame res))
	       df_taus
	       ))
	     (python
	      (do0
	       (draw :df df
		     :ros ros
		     :rds rds
		     :end None)))

	     (python
	      (do0
	       "#export"
	       (figure :figsize (tuple 9 5))
	       (do0
		(subplot 1 2 1)
		(setf df_ (aref df_taus df_taus.tan))
		(plot df_.pupil_y_normalized
		      df_.op
		      )
		(do0 (grid)
		     (title (string "tangential rays"))
		    (xlabel (string "normalized pupil y"))
		    (ylabel (string "optical path (mm)"))))
	       (do0
		(subplot 1 2 2)
		(setf df_ (aref df_taus ~df_taus.tan))
		(plot df_.pupil_z_normalized
		      df_.op
		      )
		(do0 (grid)
		     (title (string "sagittal rays"))
		    (xlabel (string "normalized pupil z"))
		    (ylabel (string "optical path (mm)"))))
	        (plt.suptitle (dot (string "phi={} theta={}")
				  (format phi theta)))
	       ))
	   
	  ;   (markdown "as long as waveaberration<lambda/14 the contributions to the field increase focus intensity.")

	     (python
	      (do0
	       "#export"
	       (figure :figsize (tuple 9 5))
	       (do0
		 (comments "wavelength in mm")
		 (setf wl_green 587.6e-6))
	       
	       (do0
		(subplot 1 2 1)
		(setf df_ (aref df_taus df_taus.tan))
		(setf da (/ df_.op wl_green))
		(plot df_.pupil_y_normalized
		      (- da (da.min))
		      )
		(do0 (grid)
		     (title (string  "tangential rays"))
		    (xlabel (string "normalized pupil y"))
		    (ylabel (string "optical path (wavelengths)"))))
	       (do0
		(subplot 1 2 2)
		(setf df_ (aref df_taus ~df_taus.tan))
		(setf da (/ df_.op wl_green))
		(plot df_.pupil_z_normalized
		      (- da (da.min))
		      )
		(do0 (grid)
		     (title (string "sagittal rays"))
		    (xlabel (string "normalized pupil z"))
		    (ylabel (string "optical path (wavelengths)"))))
	       (plt.suptitle (dot (string "phi={} theta={}")
				  (format phi theta)))
	       ))
	     (markdown "process all points with vmap")
	     (python
	      (do0
	       (setf all_ro (np.stack df_taus.ro.values)
		     )
	       (setf chief_rd  (aref df_taus.chief_rd.iloc 0))
	       (setf vtrace2_op (jit (vmap (lambda (x) (trace2_op :adf adf :ro x
								  :rd chief_rd)))))
	       (setf all_op
		(vtrace2_op all_ro))
	       ))
	     (markdown "sample pupil with 2d")
	     (python
	      (do0
	       (comments "create 2d map of tau values that will cover the pupil")
	       (setf (ntuple sag2 tan2)
		     (np.meshgrid taus_sag
				  taus_tan))
	       (comments "convert 2d tau values to ray origins perpendicular to the chief ray")
	       (setf vrays_parallel_to_chief2 (jit (vmap (lambda (x y)
							   (rays_parallel_to_chief2 :tau_sag x
										    :tau_tan y
										    :chief_ro chief_ro
										    :theta theta
										    :phi phi
										    )))))
	       (setf ros (vrays_parallel_to_chief2 (sag2.ravel)
						 (tan2.ravel)))
	       ))
	     (markdown "get exact pupil positions for the 2d maps")
	     (python
	      (do0
	       (setf vpupil (jit (vmap
				  (lambda (x)
				   (trace2 :adf adf
					   :ro x
					   :rd chief_rd
					   :start 1
					   :end 5)))))
	       (setf pupil_coords (dot  (vpupil ros)
					(reshape taus_sag.size
						 taus_tan.size
						 3)))))
	     (markdown "get optical path difference for each ray in each point of the 2d pupil")
	     (python
	      (do0
	       (setf vtrace2_op (jit (vmap (lambda (x)
					     (trace2_op :adf adf
				      :ro x
				      :rd chief_rd
					  )))))
	       (setf pathlengths (dot (vtrace2_op ros)
				      (reshape taus_sag.size
					       taus_tan.size
					       )))))
	     (markdown "construct xarray with pathlengths and pupil coordinates")
	     (python
	      (do0
	       (figure :figsize (list 8 6 ))
	       (setf wl_green 587.6e-6)
	       (setf wave_aberration0 (/ pathlengths wl_green))
	       (setf s wave_aberration0.shape)
	       (setf wave_aberration1 (- wave_aberration0 (aref wave_aberration0
								(// (aref s 0) 2)
								(// (aref s 1) 2))))
	       (setf wave_aberration (np.where (< (np.hypot (aref pupil_coords ":" ":" 1)
							    (aref pupil_coords ":" ":" 2))
						  8)
					       wave_aberration1
					       np.nan))
	       (contourf (aref pupil_coords ":" ":" 1)
			     (aref pupil_coords ":" ":" 2)
			     wave_aberration
			     :levels 100)
	       (plt.colorbar)
	       (setf cs (contour (aref pupil_coords ":" ":" 1)
			     (aref pupil_coords ":" ":" 2)
			     wave_aberration
			     :colors (string "k")
			     ))
	       (clabel cs)
	       (grid)
	       (dot plt (gca) (set_aspect (string "equal")))
	       (xlabel (string "y (mm)"))
	       (ylabel (string "z (mm)"))
	       (title (dot (string "wave aberration phi={} theta={}")
			   (format phi theta)))
	       (plt.savefig (string "/home/martin/plot.png"))
	       )
	      )
	    
	     )))))



