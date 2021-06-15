(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/57_android_automation")
  (defparameter *code-file* "run_01_android_ctl")
  (defparameter *source* (format nil "~a/source/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (let* ((code
	   `(do0
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
					subprocess
			   ;datetime
					;time
			   mss
			   cv2
			   ))
	      (imports-from (PIL Image)
			    (cv2 imshow destroyAllWindows
				 waitKey imwrite setMouseCallback)
			   )
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"))
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
	     (do0
	      (setf scrcpy
	       (subprocess.Popen (list ,@(loop for e in `(scrcpy --always-on-top
								 -m 640
								 -w
								 --window-x 0
								 --window-y 0
								 ;; --turn-screen-off
								 )
					       collect
					       `(string ,e))))))
	     (do0
	      (setf sct (mss.mss))
	      (setf x_start 0
		    y_start 0
		    x_end 0
		    y_end 0
		    cropping False)
	      (def mouse_crop (event x y flags param)
		"global x_start, y_start, x_end, y_end, cropping"
		(if (== event cv2.EVENT_LBUTTONDOWN)
		  (setf x_start x
			y_start y
			x_end x
			y_end y
			cropping True)
		    (if (== event cv2.EVENT_MOUSEMOVE)
			(when (== cropping True)
			  (setf x_end x
				y_end y))
			(when (== event cv2.EVENT_LBUTTONUP)
			  (setf x_end x
				y_end y
				cropping False)
			  (setf q (list (tuple x_start y_start)
						(tuple x_end y_end)))
			  (when (== (len q)
				    2)
			    (setf roi (aref img (slice (aref (aref q 0) 1)
						       (aref (aref q 1) 1))
					    (slice (aref (aref q 0) 0)
						   (aref (aref q 1) 0))))
			    (imshow (string "cropped") roi)
			    (imwrite (string "/dev/shm/crop.jpg")
				     roi)
			    )))))
	      (while True
		     (setf scr (sct.grab (dictionary :left 5 :top 22 :width 640 :height 384)))
		     (setf img (np.array scr))
		     (imshow (string "output")
			     img)
		     (when (== (& (cv2.waitKey 25) #xff)
			       (ord (string "q")))
		       (destroyAllWindows)
		       (setf running False)
		       break)
		     (when (== (& (cv2.waitKey 25) #xff)
			       (ord (string "m")))
		       (setMouseCallback (string "output") mouse_crop)
		       
		       ))
	      )
	     (scrcpy.terminate)
	)))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

