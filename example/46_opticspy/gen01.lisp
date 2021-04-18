(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/46_opticspy")
  (defparameter *code-file* "run_01_lens_jax")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  (let* ((code
	  `(do0
	    (do0 
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
					;   scipy.optimize
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
			   ))
		 "from opticspy.ray_tracing.glass_function.refractiveIndex import *"

		 "from jax import grad, jit, jacfwd, jacrev, vmap, lax, random"
		 ,(format nil "from jax.numpy import ~{~a~^, ~}" `(sqrt newaxis sinc abs))
		 ,(format nil "from jax import ~{~a~^, ~}" `(grad jit jacfwd jacrev vmap lax random))
		 ,(format nil "from matplotlib.pyplot import ~{~a~^, ~}"
			  `(plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid
				 legend figure gcf xlim ylim))
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
		 
		 (do0
		  (setf l (opr.lens.Lens :lens_name (string "Triplet")
				     :creator (string "XF"))
			l.FNO 5)
		  (l.lens_info))
		 
		 #+nil (do0
		  (do0
		   (;def prof () ;
		     with (Timer (string "get 3 refractive indices"))
			(setf index
			      (opr.glass_funcs.glass2indexlist l.wavelength_list (string "S-BSM18_ohara")))))
		  index
		  ;(cProfile.run (string "prof()"))
		  )
		 #-nil (do0
		  ,(let ((system-def `((1e9 1e9 air)
				       (41.15909 6.09755 S-BSM18_ohara :comment (string "first"))
				       (-957.83146 9.349 air)
				       (-51.32 2.032 N-SF2_schott)
				       (42.378 5.996 air)
				       (1e9 4.065 air :STO True :comment (string "stop"))
				       (247.45 6.097 S-BSM18_ohara)
				       (-40.04 85.59 air :comment (string "last"))
				       (1e9 10 air)))
			 (l-wl `(656.3 587.6 486.1))
			 (l-wl-name `(red green blue))
			 (system-fn "system.csv"))
		     `(do0
		       (do0
			,@(loop for e in `((add_wavelength :wl ,l-wl)
					   (add_field_YAN :angle (0 14 20))
					   )
				appending
				(destructuring-bind (name param vals) e
				  (loop for v in vals collect
					`(dot l (,name ,param ,v)))))
			(l.list_wavelengths)
			(l.list_fields))
		       (do0
			(setf system_data (pathlib.Path (string ,system-fn)))
			(if (dot system_data
				 (exists))
			    (do0
			     (setf df (pd.read_csv (string ,system-fn))))
			    (do0
			     (with (Timer (string "system definition"))
				   (setf df (pd.DataFrame
					     (list
					      ,@(loop for e in system-def
						      and i from 1
						      collect
						      (destructuring-bind (radius thickness material &key (STO 'False) (comment 'None) (output 'True)) e
							`(dict ,@(loop for e in `(radius thickness material STO output comment)
								       and f in (list radius thickness `(string ,material) STO output comment)
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
		  )
		 (do0 (comments "accumulate thicknesses (replace first infinity with a finite number for plotting)")
		      (setf xstart -30)
		      (setf (aref df (string "thickness_cum0"))
			    (+ (list xstart)
					   ("list" (+ xstart
						      (numpy.cumsum (aref df.thickness.iloc "1:"))))))
		      (comments "shift thickness_cum0 to have first surface at 0")
		      (setf (aref df (string "thickness_cum"))
			    (- df.thickness_cum0 (dot (aref df.thickness_cum0 (== df.comment (string "first")))
						      (item))))
		      (setf (aref df (string "center_x"))
			    (+ df.radius df.thickness_cum))
		      (setf (aref df (string "aperture")) 20 ;40
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
			    
			     )))

		 (do0
		  (comments "https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/")
		  (def hit_sphere (ro rd sc sr)
		   (do0
		    (string3 "intersection of ray and a sphere, returns two ray parameters tau{1,2}"
			     "rd .. ray direction"
			     "ro .. ray origin, with p(tau) = ro + tau * rd"
			     "sc .. sphere center"
			     "sr .. sphere radius"
			     " a t^2 + b t + c = 0")
		    (setf oc (- ro sc))
		    (comments "b is somehow related to the intersection sequence. i should think more about that")
		    (setf a (np.dot rd rd)
			  b (* 2 (np.dot rd (- ro sc)))
			  c (- (np.dot oc oc)
			       (** sr 2))
			  discriminant (- (** b 2) (* 4 a c)))
		    (comments "two solutions when discriminant > 0 (intersection)")
		    (comments "sign of b controls which tau is the earlier hitpoint.")
		    (setf tau0 (/ (+ b (np.sqrt discriminant))
				  (* 2 a))
			  tau1 (/ (- b (np.sqrt discriminant))
				  (* 2 a)))
		    (return (np.array (list tau0 tau1)))
		    ))
		  (print
		   (hit_sphere (np.array (list 0 0 0))
			       (np.array (list 1 0 0))
			       (np.array (list 0 0 0))
			       1))
		  (def snell (rd n ni no)
		    ;;https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
		    (string3 "rd .. ray direction"
			     "n  .. surface normal"
			     "ni .. refractive index on input side"
			     "no .. refractive index on output side")
		    (setf u (/ n1 n2)
			  p (np.dot rd n))
		    (return (+ (* u
				  (- rd (* p n)))
			       (* (np.sqrt (- 1 (* (** u 2)
						   (- 1 (** p 2)))))
				  n)))))
		 (do0
		  (setf fig (figure :figsize (list 16 3))
			ax (fig.gca)
			)
		  (grid)
		  (ax.set_aspect (string "equal"))
		  (for ((ntuple idx row) (df.iterrows))
		       (do0 ;when (< row.radius 1e8)
			 (setf r (numpy.abs row.radius))
			 
			 (ax.add_patch (matplotlib.patches.Arc (tuple row.center_x 0)
							       (* 2 r) (* 2 r)
							       :angle 0
					    :theta1 row.theta1
					    :theta2 row.theta2
					      :alpha .8
					      :facecolor None
					    :edgecolor (string "k")))
			 (plt.text row.thickness_cum
				   0
			       (dot (string "{}")
				    (format idx)))
			 (do0
			  (setf avec (* (/ np.pi 180) row.theta1 ;30 ; (numpy.random.uniform 0 360)
				     ))
			  (setf dx (* r (numpy.cos avec))
				dy (* r (numpy.sin avec)))
			  (ax.add_patch (matplotlib.patches.Arrow
					 row.center_x
					 0
					 (+ dx row.center_x)
					 (+ dy 0)
					 :edgecolor (string "k"))
					))
			 (plt.text row.center_x
				   10
			       (dot (string "c{}")
				    (format idx)))))
		  #+nil(xlim (tuple (aref df.thickness_cum.iloc 0)
			       (aref df.thickness_cum.iloc -1))
			     )
		  (xlim (tuple -100 300))
		  (ylim (tuple -50 50)))
		#+nil (do0
		  (for ((ntuple idx row) (df.iterrows))
		       (l.add_surface
			 :number idx
			 :radius row.radius
			 :thickness row.thickness
			 :glass row.material
			 :output row.output
			 :STO row.STO)))

		 #+nil 
		 ((do0
		   (plt.figure)
		   (l.refresh_paraxial)
		   (setf d (opr.trace.trace_draw_ray l))
		   (opr.draw.draw_system l)
		   (plt.grid))
		  (do0
		  ;; grid n .. rays through y axis of entrance pupil
		  ;; circular n .. rays rings in entrance pupil
		  ;; random n .. rays through entrance pupil

		   (plt.figure)
		   (opr.field.grid_generator :n 6 :grid_type (string "circular")
					     :output True)
		   (plt.grid))
		  (do0
		   (plt.figure)
		   (opr.analysis.spotdiagram l (list 1 2 3)
					     (list 1 2 3)
					     :n 6
					     :grid_type (string "circular")))

		  (do0
		   (plt.figure)
		   (opr.analysis.Ray_fan l (list 1 2 3)
					 (list 1 2 3)
					 ))
		  (do0
		   (opr.trace.trace_one_ray l :field_num 2 :wave_num 1
					      :ray (list 0 -1)
					      :start 0
					      :end 0
					      :output True
					      :output_list (list ,@(loop for e in `(X Y Z K L M)
									 collect
									 `(string ,e)))))
		  (do0
		   ,@(loop for e in `((image_position)
				      (EFY)
				      (EFY 2 3)
				      (EP)
				      EPD
				      (EX)
				      (OAL))
			   collect
			   `(dot l ,e)))))
	    ))
	 
	 )
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



