(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/55_freecad_part")
  (defparameter *code-file* "run_00_part_bottle")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 
		 #+nil(do0
		  
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
			   ; scipy.sparse
			   ;scipy.sparse.linalg
					; jax
					;jax.random
					;jax.config
					;copy
					;subprocess
			   ;datetime
			   ;time
			   ))
		 (do0

		  (imports (Part math))
		  (imports-from (FreeCAD Base)))
		 
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
		  (setf w 50.0
			h 70.0
			thick 30.0)
		  ,@(loop for e in `(((/ w -2) 0 0)
				     ((/ w -2) (/ thick -4) 0)
				     (0        (/ thick -2) 0)
				     ((/ w 2) (/ thick -4) 0)
				     ((/ w 2) 0 0)
				     )
			  and i from 1
			  collect
			  `(setf ,(format nil "p~a" i)
				 (Base.Vector ,@e)))
		  (setf arc (Part.Arc p2 p3 p4)
			l1 (Part.LineSegment p1 p2)
			l2 (Part.LineSegment p4 p5))
		  (setf e1 (l1.toShape)
			e2 (arc.toShape)
			e3 (l2.toShape)
			wire (Part.Wire (list e1 e2 e3)))
		  (do0 (setf M (Base.Matrix))

		       (M.rotateZ math.pi))
		  (do0
		   (comments "mirror wire")
		   (setf wire_ (wire.copy))
		   (wire_.transformShape M)
		   (setf wire_profile (Part.Wire (list wire wire_))))
		  (do0
		   (setf face_profile (Part.Face wire_profile))
		   (do0
		    (setf prism (Base.Vector 0 0 h)
			  body (face_profile.extrude prism)
			  )
		    #+nil
		    (setf body (body.makeFillet (/ thick 12.0)
						body.Edges))
		    (setf neck_location (Base.Vector 0 0 h)
			  neck_normal (Base.Vector 0 0 1)
			  neck_r (/ thick 4)
			  neck_h (/ h 10)
			  neck (Part.makeCylinder neck_r
						  neck_h
						  neck_location
						  neck_normal))

		    (setf body (body.fuse neck))
		    (setf body (body.makeFillet (/ thick 12.0)
						body.Edges))
		    (Part.show body)
		    
		    ))
		 )))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



