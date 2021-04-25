(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/46_opticspy")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  (write-notebook
   :nb-file (format nil "~a/source/02_trace.ipynb" *path*)
   :nb-code
   `(
     (python (do0
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
					;(xr xarray)
					;(xrp xarray.plot)
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
			       title subplot subplot2grid grid
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

			)))
		  )))

     (python (do0
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
		    (setf a 1 ;(np.dot rd rd)
			  b (* 2 (np.dot rd oc))
			  c (- (np.dot oc oc)
			       (** sr 2))
			  discriminant (- (** b 2) (* 4 a c)))
		    
		    (comments "two solutions when discriminant > 0 (intersection)")
		    (comments "sign of b controls which tau is the earlier hitpoint.")
		    (setf s .5 ;(/ .5s0 a)
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
       (setf jh (jit hit_sphere))
       (setf o (jh :ro (np.array (list -20 0 0))
		   :rd (np.array (list 1 0 0))
		   :sc (np.array (list 0 0 0))
		   :sr 1))))
     (python
      (do0
       (def trace (&key df ro rd (start 1) (end None))
	 (string3 "trace ray (ro,rd) through the system defined in df. start and end define the surfaces to consider. ")
        (do0
	 (if (is end None)
	     (setf end (- (len df) 1))
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
	      
	       ,@(loop for e in `(tau ; tau2
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
			     ))))
	 (return (pd.DataFrame res))))
       ))

     (python
      (do0
       (comments "trace a ray through the full system")
       (trace :df df
	      :ro (np.array (list -20 0 0))
	      :rd (np.array (list 1 .2 0)))))
     (python
      (do0
       (comments "reduce number of arguments and return values to minimum.")
       (def chief (x)
	 (setf dfo
	  (trace :df df
		 :ro (np.array (list -20 10 0))
		 :rd (np.array (list 1 x 0))
		 :end 5))
	 (return (dot dfo
		      (aref iloc -1)
		      (aref phit 1)
		      (item))))
       
       ))
     (python
      (do0
       (comments "search for chief ray (manually)")
       (chief .2)))
     (python
      (do0
       (comments "search for chief ray with a graph")
       
       (setf xs (np.linspace -3 3 17)
	     ys (list))
       (for (x xs)
	    (ys.append (chief x)))
       (figure)
       (plt.plot xs ys)
       (grid)
       ))
      (python
      (do0
       (comments "search for chief ray with root finder (without gradient)")
       (setf sol
	     (scipy.optimize.root_scalar chief :method (string "brentq")
					  :bracket (list -0.5 0.5)
				    ))
       sol
       ))
     ,@(let ((l `(radius n_green center_x)))
	 `(
	   (python
	    (do0
	     (comments "the only required parameters for trace")
	     (setf adf0 (aref df (list ,@ (loop for e in l collect `(string ,e)))))
	     (setf adf (jax.numpy.asarray
			(dot adf0
			     values)))
	     (comments "first argument selects surface_idx, second argument is column: adf[3,1] is surface_idx=3, column=n_green")
	     adf0))
	   (python
	    (do0
	     
	     (comments "create a trace function that only uses arrays (so that all computations can be represented with jax)")
	     "@jit"
	     (def trace2 (&key adf ro rd (start 1) (end None))
	 (string3 "trace ray (ro,rd) through the system defined in adf. start and end define the surfaces to consider. return the hit point in the last surface")
        (do0
	 (if (is end None)
	     (setf end (- (len adf) 1))
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

       
       "@jit"
       (def trace2_opd (&key adf ro rd (start 1) (end None))
	 (string3 "trace ray (ro,rd) through the system defined in adf. start and end define the surfaces to consider. return the optical path length between ray origin and the last hit point")
        (do0
	 (if (is end None)
	     (setf end (- (len adf) 1))
	     (setf end (+ end 1)))
	 (setf rd (/ rd (np.linalg.norm rd)))
	 (setf opd 0d0)
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
		    (setf opd (+ opd (* ni (np.linalg.norm (- p1 ro)))))
		       (setf n (sphere_normal_out :p_hit p1 :sc sc :sr sr))
		       (setf normal (* -1 n))
		       (setf rd_trans (snell :rd rd
					     :n normal
					     :ni ni
					     :no no))
		    
		       (setf rd rd_trans
			     ro p1)
		    )))
	 
	 (return opd)))
       
       ))
	   (python
	    (do0
	     "@jit"
	     (def chief2 (x)
	       (setf phit (trace2 :adf adf
				  :ro (np.array (list -20 10 0))
				  :rd (np.array (list 1 x 0))
				  :end 5))
	       #+nil (return (dot phit
				  (aref iloc -1)
				  (aref ro 1)
				  (item)))
	       #-nil
	       (return (aref phit 1)
		       )
	       )))
	   (python
      (do0
       (comments "compare pandas based chief ray search with jax/numpy-based")
       
       (setf xs (np.linspace -3 3 17)
	     ys (list)
	     ys2 (list))
       (for (x xs)
	    (ys.append (chief x))
	    (ys2.append (chief2 x)))
       (figure)
       (plt.plot xs ys :label (string "pandas"))
       (plt.plot xs ys2 :label (string "jax/numpy"))
       (legend)
       (grid)
       ))

	   (python
	    (do0
	     (setf chief2j (jit chief2))
	     (setf d_chief2 (jacfwd chief2))
	     (setf d_chief2j (jit d_chief2))))
	   (python
	    (do0
	     (d_chief2 .2)))
	   (python
	    (do0
	     (with (Timer (string "newton search (conventional) 2.1s")
			  )
		   (setf sol
			   (scipy.optimize.root_scalar chief2 :method (string "newton")
							      :x0 .4
							      :fprime d_chief2)))
	     sol))
	   (python
	    (do0
	     (with
	      (Timer (string "newton search (jit) 0.016")
		     )
	      (setf sol
			   (scipy.optimize.root_scalar chief2j :method (string "newton")
							       :x0 .4
							       :fprime d_chief2j)))
	     sol))
	   ))
     
     (python
      (do0
       
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
		   (do0			;when (< row.radius 1e8)
		    (setf r (numpy.abs row.radius))
		    (setf x (- (dot row thickness_cum)
			       (dot row thickness)))
		    #+nil(try
			  (setf x (dot (aref df.iloc (- idx 1)) thickness_cum))
			  ("Exception as e"
			   (setf x -10))) 
			 
		    (ax.add_patch (matplotlib.patches.Arc (tuple row.center_x 0)
							  (* 2 r) (* 2 r)
							  :angle 0
							  :theta1 row.theta1
							  :theta2 row.theta2
							  :alpha 1
							  :linewidth 3
							  :facecolor None
							  :edgecolor (aref colors idx)))
		    #+nil (do0 (plt.text x
			       40
			       (dot (string "{}")
				    (format idx))
			       :color  (aref colors idx))
			 (plt.text row.center_x
				   (+ 35 (* -5 (== (% idx 2) 0)))
				   (dot (string "{:3.2f}")
					(format row.thickness))
				   :color  (aref colors idx)))
		    #+nil  ,@(loop for e in `(row.theta1 ;row.theta2
					      )
				   collect
				   `(do0
				     (setf avec (* (/ np.pi 180) ,e ;row.theta1
						   ))
				     (setf dx (* r (numpy.cos avec))
					   dy (* r (numpy.sin avec)))
				     (ax.add_patch (matplotlib.patches.Arrow
						    row.center_x
						    0
						    (+ dx) 
						    (+ dy)
						    :edgecolor  (aref colors idx))
						   )))
		    #+nil(do0
		     (plt.text row.center_x
			       -40
			       (dot (string "c{}")
				    (format idx))
			       :color (aref colors idx) )
		     (plt.text row.center_x
			       (- 45 (* -5 (== (% idx 2) 0)))
			       (dot (string "x{:3.2f}")
				    (format row.center_x))
			       :color (aref colors idx) ))))))
	 #+nil(xlim (tuple (aref df.thickness_cum.iloc 0)
			   (aref df.thickness_cum.iloc -1))
		    )
	 (do0

	  (for ((ntuple ro rd1) (zip ros rds) #+nil (list ,@(loop for e in `(0 1e-6 1) ; from -25 upto 15 by 2
								  collect
								  `(np.array (list -20 ,e 0)))))
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
		     #+nil,@(loop for e in `(tau)
				  collect
				  `(print (dot (string ,(format nil "~a={}" e))
					       (format ,e))))
		     ,@(loop for e in `(tau ; tau2
					) and color in `(k)
			     collect
			     `(do0 (setf p1 (eval_ray ,e :ro ro :rd rd1)) ;; hit point
				   (setf n (sphere_normal_out :p_hit p1 :sc sc :sr sr))
				   (setf rd_trans (snell :rd rd1
							 :n (* -1 n)
							 :ni (aref df.n_green (- surface_idx 1))
							 :no (aref df.n_green (- surface_idx 0))))
				   
					;(setf p2 (eval_ray 10 :ro p1 :rd rd_trans))
				   #+nil (plt.scatter
					  (aref ro 0)
					  (aref ro 1))
				   (plot (list (aref ro 0)
					       (aref p1 0)
					;(aref p2 0)
					       )
					 (list (aref ro 1)
					       (aref p1 1)
					;(aref p2 1)
					       )
					 :color (string ,color)
					 :alpha .3
					 )
				   (setf rd1 rd_trans
					 ro p1)
				   #+nil
				   (plot (list (aref p1 0)
					       (aref (+ p1 (* 3 n)) 0))
					 (list (aref p1 1)
					       (aref (+ p1 (* 3 n)) 1))
					 :color (string "r")
					 :alpha .3
					 )))))))
	 (xlim (tuple -35 125))
	 (ylim (tuple -50 50))))))

     (markdown "plot a single ray")
     
     (python
      (do0
       (draw :df df
	     :ros (list (np.array (list -20 10 0)))
	     :rds (list (np.array (list 1 0 0)))
	     )))
     (markdown "plot a ray bundle")

     (python
	   (do0
	    (setf theta_ (np.deg2rad theta)
		  phi_ (np.deg2rad phi))
	    (setf ros (list)
		  rds (list))

	    (for (ro_y (np.linspace -20 20 30))
		 (do0
			
			(ros.append (np.array (list -20 ro_y
						    0)))
			(rds.append  (np.array (list (np.cos theta_)
						     (* (np.cos phi_)
							(np.sin theta_))
						     (* (np.sin phi_)
							(np.sin theta_)))))
			
			
			))
	   
	    (draw :df df
		 :ros ros
		  :rds rds
		  :end 5
		  ))
	   
	   )
     
     (markdown "add more parameters to the chief2 function to vary field position and the target point in the pupil. the new function can find more than just chief rays, so the name changes to `into_stop`")
     (python
      (do0
       "@jit"
       (def into_stop (&key (ro_y 10) (ro_z 0) (theta .1) (phi 0))
	 (string "trace from field point into stop. this can be used to find theta that corresponts to the chief ray or pairs of theta and phi that belong to marginal rays. Angles are in degree.")
	 (setf theta_ (np.deg2rad theta)
	       phi_ (np.deg2rad phi))
	 (setf phit (trace2 :adf adf
			    :ro (np.array (list -20 ro_y ro_z))
			    :rd (np.array (list (np.cos theta_)
						(* (np.cos phi_)
						   (np.sin theta_))
						(* (np.sin phi_)
						   (np.sin theta_))))
			    :end 5))
	 (comments "the stop plane fixes the x coordinate, only return y and z")
	 (return (aref phit "1:")))
      ; (setf into_stop_j (jacfwd into_stop))
       ))

     ,@(let ((l `((chief (:ro_y x :ro_z ro_z :theta theta :phi phi)
			 :x0 10
			 :target 0)
		  (coma_up (:ro_y x :ro_z ro_z :theta theta :phi phi)
			     :x0 10
			     :target (aref df.aperture 5))
		  (coma_low (:ro_y x :ro_z ro_z :theta theta :phi phi)
			       :x0 10
			       :target (* -1 (aref df.aperture 5)))
		 )))
	`((python
	   (do0
	    (setf theta 5
		  phi 0
		  ro_z 0)
		 #+nil ,@(loop for e in l
			 collect
			 (destructuring-bind (name args &key x0 target) e
			   `(setf ,(format nil "merit_~a" name)
				  (jit (value_and_grad
					(lambda (x)
					  (- (aref  (into_stop ,@args)
						  ;; i only care about the y coordinate because phi=0
						    0)
					     ,target))
					:argnums 0)))))
		 (setf into_stop_meridional
				  (jit (value_and_grad
					(lambda (x target)
					  (- (aref  (into_stop :ro_y x :ro_z ro_z :theta theta :phi phi)
						  ;; i only care about the y coordinate because phi=0
						    0)
					     target))
					:argnums 0)))))
	  ,@(loop for e in l
		  collect
		  (destructuring-bind (name args &key x0 target) e
		    `(python
		     (do0
		      (setf ,(format nil "sol_~a" name)
			    (scipy.optimize.root_scalar ;,(format nil "merit_~a" name)
			     into_stop_meridional
			     :args (tuple ,target)
			     :method (string "newton")
							:x0 ,x0
							:fprime True
							))
		      ,(format nil "sol_~a" name))
		     )))
	  (python
	   (do0
	    (setf theta_ (np.deg2rad theta)
		  phi_ (np.deg2rad phi))
	    (setf ros (list)
		  rds (list)) 
	    ,@(loop for e in l
		    collect
		    (destructuring-bind (name args &key x0 target) e
		      `(do0
			
			(ros.append (np.array (list -20 (dot ,(format nil "sol_~a" name)
							     root
							     (item))
						    0)))
			(rds.append  (np.array (list (np.cos theta_)
						     (* (np.cos phi_)
							(np.sin theta_))
						     (* (np.sin phi_)
							(np.sin theta_)))))
			
			
			)
		      ))
	    (draw :df df
		 :ros ros
		  :rds rds
		  :end None)))

	  (python
	   (do0
	    (setf n 32)
	    (setf normalized_pupil
		  (np.concatenate
		      (tuple (np.linspace -1
					  0 (// n 2))
			     (aref (np.linspace 0
						1
						(// n 2))
				   "1:"))))
	    (setf ys (np.concatenate
		      (tuple (np.linspace sol_coma_low.root
					  sol_chief.root (// n 2))
			     (aref (np.linspace sol_chief.root
						sol_coma_up.root
						(// n 2))
				   "1:")))))) 
	  

	  (python
	   (do0
	    (setf theta_ (np.deg2rad theta)
		  phi_ (np.deg2rad phi))
	    (setf ros (list)
		  rds (list))

	    (for (ro_y ys)
		 (do0
		  
		  (ros.append (np.array (list -20 ro_y
					      0)))
		  (rds.append  (np.array (list (np.cos theta_)
					       (* (np.cos phi_)
						  (np.sin theta_))
					       (* (np.sin phi_)
						  (np.sin theta_)))))
		  
		  
		  ))
	    
	    (draw :df df
		  :ros ros
		  :rds rds
		  
		  ))
	   
	   )

	  (python
	   (do0
	    (setf opd (list))
	       (for (ro_y ys)
		 (do0
		  
		  (setf ro (np.array (list -20 ro_y
					      0)))
		  (setf rd  (np.array (list (np.cos theta_)
					       (* (np.cos phi_)
						  (np.sin theta_))
					       (* (np.sin phi_)
						  (np.sin theta_)))))
		  
		  (opd.append
		   (dot
		    (trace2_opd :adf adf :ro ro :rd rd)
		    (item)))))
	       ))
	  (python
	   (do0
	    (figure)
	    (plot normalized_pupil
		  opd)
	    (grid)
	    (xlabel (string "normalized pupil"))
	    (ylabel (string "opd"))))

	  

	  
	  
	  


	  ))
     

     
     
     ))
  )



