(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/42_android_repl")
  (defparameter *code-file* "main")
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
		(imports (time
			  subprocess
			  os))
		,@(loop for (pkg exprs) in `((kivy.app (App))
					;(kivy.uix.widget (Widget))
					     (kivy.uix.boxlayout (BoxLayout))
					     #+nil (kivy.properties (NumericProperty
							       ReferenceListProperty
							       ObjectProperty)
							      )
					     ;(kivy.vector (Vector))
					(kivy.clock (Clock))
					     ;(camera (Camera2))
					     )
			collect
			(format nil "from ~a import ~{~a~^, ~}" pkg exprs))
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
		 (class MainLayout (BoxLayout)
			pass)
		 (class MainApp (App)
		    	(def build (self)
			  (return (MainLayout)))
			(def on_start (self)
			  (Clock.schedule_once self.detect 5))
			(def detect (self nap)
			  (setf 
				ctime (aref (time.ctime)
					    "11:19")
				;self.root.ids.label.text
				#+nil (dot (string "{:s} {}x{} image")
					   (format ctime rows cols)))
			  (print (dot (string "{:s}")
				     (format ctime )))
			  (Clock.schedule_once self.detect 1)))
		 (def listfiles (folder)
		   (for ((ntuple root folders files)
			 (os.walk folder))
			(for (filename (+ folders files))
			     (yield (os.path.join root filename)))))
		 (when (== __name__ (string "__main__"))
		   (for (filename (listfiles (string "../")))
			(print filename))
		   (subprocess.Popen (string "jupyter notebook")
				     :shell True)
		   (setf app (MainApp))
		   (dot app
			(run)))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)
    
    (with-output-to-file (s (format nil "~a/source/main.kv" *path*)
			    :if-exists :supersede
			    :if-does-not-exist :create)
      (format s "#:kivy 1.0.9

<MainLayout>:
    orientation: 'vertical'
    padding: (36, 36)
 
"))))

