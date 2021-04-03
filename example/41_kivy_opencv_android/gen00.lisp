(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/41_kivy_opencv_android")
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
		(imports (time))
		,@(loop for (pkg exprs) in `((kivy.app (App))
					;(kivy.uix.widget (Widget))
					     (kivy.uix.boxlayout (BoxLayout))
					     #+nil (kivy.properties (NumericProperty
							       ReferenceListProperty
							       ObjectProperty)
							      )
					     ;(kivy.vector (Vector))
					(kivy.clock (Clock))
					     (camera (Camera2))
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
			  (setf image self.root.ids.camera.image
				(ntuple rows cols) (aref image.shape ":2")
				ctime (aref (time.ctime)
					    "11:19")
				;self.root.ids.label.text
				#+nil (dot (string "{:s} {}x{} image")
				     (format ctime rows cols)))
			  (Clock.schedule_once self.detect 1)))
		 
		 (when (== __name__ (string "__main__"))
		   (setf app (MainApp))
		   (dot app
			(run)))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)
    (write-source (format nil "~a/source/camera" *path*)
		  `(do0
		    (imports (kivy
			      (cv cv2)
			      (np numpy)
			      ))
		    ,@(loop for (pkg exprs) in
			    `((kivy.uix.camera (Camera))
			      (kivy.graphics.texture (Texture)))
			collect
			(format nil "from ~a import ~{~a~^, ~}" pkg exprs))
		    (class Camera2 (Camera)
			   (setf firstFrame None)
			   (def _camera_loaded (self *largs)
			     (if (== kivy.platform (string "android"))
				 (setf self.texture (Texture.create :size self.resolution
								    :colorfmt (string "rgb"))
				       self.texture_size (list self.texture.size))
				 (dot (super Camera2 self)
				      (_camera_loaded))))
			   (def on_tex (self *l)
			     (when (== kivy.platform (string "android"))
			       (setf buf (self._camera.grab_frame))
			       (unless buf
				 (return))
			       (setf frame (self._camera.decode_frame buf)
				     frame (self.process_frame frame)
				     self.image frame
				     buf (frame.tostring)
				     )
			       (self.texture.blit_buffer buf
							 :colorfmt (string "rgb")
							 :bufferfmt (string "ubyte")))
			     (dot (super Camera2 self)
				  (on_tex *l)))
			   (def process_frame (self frame)
			     (setf (ntuple r g b) (cv.split frame)
				   frame (cv.merge (tuple b g r))
				   (ntuple rows cols channel) frame.shape
				   M (cv.getRotationMatrix2D (tuple (/ cols 2)
								    (/ rows 2)
								    90 1))
				   dst (cv.warpAffine frame M (tuple cols rows))
				   frame (cv.flip dst 1))
			     (when (== self.index 1)
			       (setf frame (cv.flit dst -1)))
			     (return frame)))))
    (with-output-to-file (s (format nil "~a/source/main.kv" *path*)
			    :if-exists :supersede
			    :if-does-not-exist :create)
      (format s "#:kivy 1.0.9

<MainLayout>:
    orientation: 'vertical'
    padding: (36, 36)
    
    Camera2:
        index: 0
        resolution: (640,480)
        id: camera
        play: True

"))))

