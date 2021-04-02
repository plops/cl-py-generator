(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/39_opencv")
  (defparameter *code-file* "run_00_raw_cam")
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
			   (cv cv2)
					;(mp mediapipe)
					;jax
			   ; jax.random
			   ;jax.config
			   copy
			   
			   ))
		;"from mss import mss"
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
		 #+nil
		 (print (cv.getBuildInformation))
		 
		 #-nil (do0
			(setf cap (cv.VideoCapture (string "/dev/video0")))
			,@(loop for e in `((MODE 0)
					  
					   (FRAME_WIDTH 640)
					   (FRAME_HEIGHT 480)
					   (AUTO_EXPOSURE 1)
					   (EXPOSURE 50))
				collect
					  (destructuring-bind (name &optional value)
					      e
					    (let ((full-mode (format nil "cv.CAP_PROP_~a" name)))
					      `(do0
						
						(do0
						 (setf r
						       ,(if value
							    `(cap.set ,full-mode ,value)
							    `(cap.set ,full-mode )))
						 (unless r
						   (print (string ,(format nil "problem with ~a"  name))
							  )))
						(do0
						 (setf r
						       (cap.get ,full-mode ))
						 (print (dot  (string ,(format nil "~a={}"  name))
							      (format r))
							)))))))
		 
		 
		 (cv.namedWindow (string "image")
				 cv.WINDOW_NORMAL)
		 (cv.resizeWindow (string "image")
				  (// 1920 2) 1080)
		 (do0
		  (while True
			 (do0 (setf (ntuple ret image)
				    (cap.read))
			      (unless ret
				break))
			 
					;(setf debug_image (copy.deepcopy image))
			 (do0 ;(setf image (cv.cvtColor image cv.COLOR_YUYV2RGB))
			     ; (setf debug_image (copy.deepcopy image))

			      (setf image.flags.writeable False)
			      )

			
			 
			 (do0
			  (setf key (cv.waitKey 1))
			  (when (== 27 key)
			    break)
			  )
			 (cv.imshow (string "image")
				    image ))
		  )
		 (do0
		  (cap.release)
		  (cv.destroyAllWindows))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

