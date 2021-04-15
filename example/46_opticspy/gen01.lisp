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
					;time
					;docopt
					;pathlib
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
			   (np numpy)
			   ;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
			  ; copy

			   (opr opticspy.ray_tracing)
			   ))
		 "from opticspy.ray_tracing.glass_function.refractiveIndex import *"
		 
		 (do0
		  (setf l (opr.lens.Lens :lens_name (string "Triplet")
				     :creator (string "XF"))
			l.FNO 5)
		  (l.lens_info))
		 (do0
		  ,@(loop for e in `((add_wavelength :wl (656.3 587.6 486.1))
				     (add_field_YAN :angle (0 14 20))
				     )
			  appending
			  (destructuring-bind (name param vals) e
			    (loop for v in vals collect
				  `(dot l (,name ,param ,v)))))
		  (l.list_wavelengths)
		  (l.list_fields))
		 (do0
		  (setf df (pd.DataFrame
			    (list
			     ,@(loop for e
				       in `((1e9 1e9 air)
					    (41.15909 6.09755 S-BSM18_ohara)
					    (-957.83146 9.349 air)
					    (-51.32 2.032 N-SF2_schott)
					    (42.378 5.996 air)
					    (1e9 4.065 air :STO True)
					    (247.45 6.097 S-BSM18_ohara)
					    (-40.04 85.59 air)
					    (1e9 0 air))
				       and i from 1
				     collect
				     (destructuring-bind (radius thickness material &key (STO 'False) (output 'True)) e
				       `(dict ,@(loop for e in `(radius thickness material STO output)
						      and f in (list radius thickness `(string ,material) STO output)
						      collect
						      
						      `((string ,e) ,f)))
				       
				       )))))
		  (df.to_csv (string "system.csv")))
		 (do0
		  (setf index
			(opr.glass_funcs.glass2indexlist l.wavelength_list (string "S-BSM18_ohara")))
		  index)
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



