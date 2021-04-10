(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/43_pysimplegui")
  (defparameter *code-file* "run_00_one_shot")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

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
		#+nil  (imports (		;os
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
			   (cv cv2)
					;(mp mediapipe)
					;jax
			   ; jax.random
			   ;jax.config
			   copy
			   
				 ))
		(imports ( (sg PySimpleGUI)))
		(do0
		 (setf layout (list (list (sg.Text (string "name:")))
				    (list (sg.Input))
				    (list (sg.Button (string "Ok")))))
		 (setf window (sg.Window (string "Window Title") layout))
		 (while True
			(setf (ntuple event values)
			      (window.read))
			(when (in event (tuple sg.WIN_CLOSED
					       (string "Cancel")))
			  break))
		 (window.close))
		))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)
    (write-source (format nil "~a/source/run_01_interactive" *path*)
		  `(do0
		    (imports ((sg
			       ;;PySimpleGUI
			       ;;PySimpleGUIQt
			       ;;PySimpleGUIWx
			       PySimpleGUIWeb
			       )))
		    (do0
		 (setf layout (list (list (sg.Text (string "name:")))
				    (list (sg.Input :key (string "-INPUT-")))
				    (list (sg.Text :size (tuple 40 1)
						   :key (string "-OUTPUT-")))
				    (list
				     (sg.Button (string "Ok"))
				     (sg.Button (string "Quit")))))
		 (setf window (sg.Window (string "Window Title") layout))
		 (while True
			(setf (ntuple event values)
			      (window.read))
			(when (in event (tuple sg.WIN_CLOSED
					       (string "Quit")))
			  break)
			(dot (aref window (string "-OUTPUT-"))
			     (update (+ (string "Hello ")
					(aref values (string "-INPUT-")
					      )
					(string "! Thanks for trying.")))))
		 (window.close))
		    ))
    (write-source (format nil "~a/source/run_02_pyplot" *path*)
		  `(do0
		    (imports ((sg
			       PySimpleGUIWeb
			       )
			      (np numpy)
			      matplotlib.backends.backend_tkagg
			      matplotlib.figure
			      matplotlib
			      
			      (plt matplotlib.pyplot)
			      io))
		    (do0
		 (setf layout (list 
				    (list (sg.Image :key (string "-IMAGE-")))
				    (list
				     (sg.Button (string "Draw"))
				     (sg.Button (string "Exit")))))
		 (setf window (sg.Window (string "plot example") layout))
		 (while True
			(setf (ntuple event values)
			      (window.read))
			(when (in event (tuple sg.WIN_CLOSED
					       (string "Exit")))
			  break)
			(when (== event (string "Draw"))
			  (do0
			   (plt.close (string "all"))
			   (setf fig (plt.figure :figsize (list 5 4)
						 :dpi 72)
				 x (np.linspace 0 3 100))
			   (dot fig (add_subplot 111)
				(plot x (np.sin (* 2 np.pi x)))))
			  (do0
			   (setf canv (matplotlib.backends.backend_tkagg.FigureCanvasAgg (plt.gcf))
				 buf (io.BytesIO))
			   (canv.print_figure buf :format (string "png"))
			   (when (is buf None)
			     (print "problem"))
			   (buf.seek 0)
			   (dot (aref window (string "-IMAGE-"))
				(update :data (buf.read))))))
		 (window.close))
		    
		    ))
    ))

