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
	      (setf sct (mss.mss)
		    )
	      (while True
		     (setf scr (sct.grab (dictionary :left 5 :top 22 :width 640 :height 384)))
		     (setf img (np.array scr))
		     (cv2.imshow (string "output")
				 img)
		     (when (== (& (cv2.waitKey 25) #xff)
			       (ord (string "q")))
		       (cv2.destroyAllWindows)
		       (setf running False)
		       break))
	      )
	     (scrcpy.terminate)
	)))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

