(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/71_datoviz")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (pd pandas)
					;(xr xarray)
					;matplotlib
					(s skyfield)
					;(ds dataset)
					; cv2
      ;datoviz
      ))
  (let ((nb-file "source/02_play.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play02"))
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
			   skyfield.api
			  skyfield.data
			  skyfield.data.hipparcos
			  ))
		
		;"from cv2 import *"
	      	#+nil
			(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))

		(imports-from (datoviz canvas  run colormap))
		 )
	       ))
      (python
       (do0
	"#export"
	
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
	       (setf filename (string "test.txt")))))
      (python
       (do0
	"#export"
	(setf parser (argparse.ArgumentParser))
	(parser.add_argument (string "-i")
			     :dest (string "filename")
			     ;:required True
			     :help (string "input file")
			     :metavar (string "FILE"))
	
	(setf args (parser.parse_args))
	(print args)
	
	
	))
      (python
       (do0
	"#export"
	
	(with (as (open (string "/home/martin/stage/cl-py-generator/example/70_star_tracker/source/hip_main.dat"))
		  f)
	      (setf df
		    (dot skyfield data hipparcos
			 (load_dataframe f))))
	(comments "ra .. 0 360"
		  "dec .. -90-90"
		  "magnitude .. -2-14, peak at 8"
		  "parallax .. -54-300 and few upto 800")
	))
      (python
       (do0
	(comments "for interactive IPython")
	"%gui datoviz"))
      (python
       (do0
	"#export"
	(setf c (canvas)
	      gui (c.gui (string "gui"))
	      s (c.scene)
	      ;; static panzoom axes arcball camera
	      p (s.panel :controller (string "arcball"))
	      v (p.visual (string "marker") :depth_test True))
	;; pos 3d
	;; ms marker size
	;; color values
	
	(setf 
	      rad 10
	      dec (* (/ np.pi 180) (+ 90  df.dec_degrees))
	      ra (* (/ np.pi 180) df.ra_degrees)
	      (aref df (string "x")) (* rad
					(np.sin dec)
					(np.cos ra))
	      (aref df (string "y")) (* rad
					(np.sin dec)
					(np.sin ra))
	      (aref df (string "z")) (* rad
					(np.cos dec)
					)
	      pos (dot (aref df (list (string "x")
				      (string "y")
				      (string "z")))
		      values)
	      
	      ms (- 16 df.magnitude.values)
	      color_values ms)
	(setf color (colormap
		     color_values
		     :vmin -2 :vmax 16
		     :alpha .2
		     :cmap (string "cubehelix")))

	(do0 
	 (setf sf (gui.control (string "slider_float")
			       (string "marker size")
			       :vmin .05 :vmax 3))
	 "@sf.connect"
	 (def on_change (value)
	   (v.data (string "ms")
		   (* ms value))))
	(do0 
	 (setf sf (gui.control (string "slider_float")
			       (string "alpha")
			       :vmin 0 :vmax 1))
	 "@sf.connect"
	 (def on_change (value)
	   (setf color (colormap
		     color_values
		     :vmin -2 :vmax 16
		     :alpha value
		     :cmap (string "cubehelix")))
	   (v.data (string "color")
		   color)))
	
	,@(loop for e in `(pos ms color)
		collect
		`(v.data (string ,e) ,e))
	#+nil (v.data (string "line_width")
		0s0)
	(comments "right click and move up/down left/right to scale axes")
	(run)
	))
      
      ))))



