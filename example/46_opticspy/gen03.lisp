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
   :nb-file (format nil "~a/source/02_trace.ipynb" *path*)
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
	      "#export"
	      (do0
	       (setf d 12
		     R1 200
		     R2 -200
		     n0 1
		     n 1.5
		     f (/ 1d0
			  (* (/ (- n n0)
				n0)
			     (+ (/ 1d0 R1)
				(/ -1d0 R2)
				(/ (* (- n n0) d)
				   (* n R1 R2))))))
	       ,(let ((system-def `((1e9 (- (* 2 f) (/ d 2)) n0 30)
				    (R1 d n 30)
				    (R2 (- f (/ d 2)) n0 30)
				    (1e9 f n0 30 :STO True :comment (string "stop"))
				    (1e9 f n0 30 :STO True :comment (string "stop"))
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
	  (return (pd.DataFrame res))))
       ))

     (python
      (do0
       (comments "trace a ray through the full system") 
       (trace :df df
	      :ro (np.array (list -20 0 0))
	      :rd (np.array (list 1 .2 0)))))

  
     

     
     
     ))
  )



