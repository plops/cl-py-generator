(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/70_star_tracker")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (pd pandas)
      ;(xr xarray)
      matplotlib
      (s skyfield)
					;(ds dataset)
      cv2
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
		(do0
		      
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
					skimage.restoration
			  skimage.morphology
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
			  (sns seaborn)
			   skyfield.api
			  skyfield.data
			  skyfield.data.hipparcos
			  ))
		
		"from cv2 import *"
	      		(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
		 
		)
	       ))
      (python
       (do0
	"#export"
	(sns.set_theme)
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
	(class ArgsStub
	       ()
	       (setf filename  (string "/home/martin/ISS Timelapse - Stars Above The World (29 _ 30 Marzo 2017)-8fCLTeY7tQg.mp4.part")))))
      (python
       (do0
	"#export"
	(setf parser (argparse.ArgumentParser))
	(parser.add_argument (string "-i")
			     :dest (string "filename")
			     :required True
			     :help (string "input file")
			     :metavar (string "FILE"))
	(setf args (parser.parse_args))
	
	
	))
      (python
       (do0
	"#export"
	(with (as (open (string "/home/martin/stage/cl-py-generator/example/70_star_tracker/source/hip_main.dat"))
		  f)
	 (setf hip
	       (dot skyfield data hipparcos
		    (load_dataframe f))))
	(comments "ra .. 0 360"
		  "dec .. -90-90"
		  "magnitude .. -2-14, peak at 8"
		  "parallax .. -54-300 and few upto 800")
	(do0
	 (setf h (plt.hist2d hip.ra_degrees
			     hip.dec_degrees
			     :bins (list (np.linspace 0 360 (// 360 2))
					 (np.linspace -90 90 (// 180 2)))
			     :cmap (string "cubehelix")
			     :norm (dot matplotlib
					colors
					(LogNorm))
			     ))
	 (plt.colorbar   (aref h 3))
	 (do0
	  (xlabel (string "right ascension [degree]"))
	  (ylabel (string "declination [degree]"))
	  (do0
	   (xlim 0 360)
	   (ylim -90 90)
	   ;(grid)
	   (plt.axis (string "equal")))))
	(do0
	 (setf max_mag 6)
	 (dot (aref hip (< hip.magnitude max_mag))
	      plot (scatter :x (string "ra_degrees")
			    :y (string "dec_degrees")
			    :s 1
			    :marker (string ",")))
	 (do0
	  (xlabel (string "right ascension [degree]"))
	  (ylabel (string "declination [degree]"))
	  (title (dot (string "stars with magnitude < {}")
		      (format max_mag)))
	  ;(grid)
	  (do0
	   (xlim 0 360)
	   (ylim -90 90)
	   (plt.axis (string "equal")))))
	))
      (python
       (do0
	"#export"
	(setf cap (cv2.VideoCapture args.filename #+nil  (string
					;"ISS Timelapse - Stars Above The World (29 _ 30 Marzo 2017)-8fCLTeY7tQg.mp4.part"
							  "/home/martin/stars_XnRy3sJqfu4.webm"
				     )))
	(unless (cap.isOpened)
	  (print (string "error opening video stream or file")))
	(while (cap.isOpened)
	       (setf (ntuple ret frame)
		     (cap.read))
	       (if ret
		   (do0
		    (setf da (aref frame
				      (slice "" 512)
				      (slice 900 "")
				      1))
		    (cv2.imshow (string "frame")
				(* 255 (skimage.morphology.h_maxima da 20)))
		    (when (== (& (cv2.waitKey 25)
				 #xff  )
			      (ord (string "q")))
		      break))
		   break)
	       )
	(cap.release)
	(cv2.destroyAllWindows)
	))))))



