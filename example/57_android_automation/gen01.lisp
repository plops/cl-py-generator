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
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
		   (format (- (time.time) start_time)
			   ,@rest)))))
  (let* ((l-template `((SALE 1904 105  (608 37))
			 (back 80 74 (25.5 25))
			 (how_to_play 970 893 (312 288))
			 (hook1 292 1090 (91.5 351))
			 (pause)
			 (Play ))
		     )
	 (code
	   `(do0
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
				        scipy.optimize
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
			   threading
			   ;datetime
					;time
			   mss
			   cv2
			   time
			   ))
	      (imports-from (PIL Image)
			    (cv2 imshow destroyAllWindows imread
				 waitKey imwrite setMouseCallback circle matchTemplate minMaxLoc)
			    (ppadb.client Client)
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
	     (setf start_time (time.time)
		   debug True)
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
					       `(string ,e)))))
	      ,(lprint "start scrcpy"))

	     

	      (do0
	      (setf adb (Client :host (string "127.0.0.1")
				:port 5037)
		    devices (adb.devices))
	      (when (== 0 (len devices))c
		(print (string "no device attached"))
		(quit))
	      (setf device (aref devices 0))

	      (do0
	       (setf running True)
	       (def touch_collect_suns ()
		 "global running"
		 (while running
			(setf x1 150
			      y1 100
			      x2 630
			      y2 200)
			,(lprint "swipe" `(x1 y1 x2 y2))
			#+nil
			(device.shell (dot (string "input touchscreen swipe {} {} {} {} 10")
					   (format x1 y1 x2 y2)))))

	       (def tap_play ()
		 "global running"
		 (while running
			(setf x1 740
			      y1 870

			      )
			,(lprint "tap" `(x1 y1))
			#+nil
			(device.shell (dot (string "input touchscreen tap {} {}")
					   (format x1 y1)))))
	       
	       (do0 (setf thr (threading.Thread :target tap_play))
		    (thr.start)
		    (setf running False))))
	     ;; https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
	     (do0
	      (time.sleep 1)
	      ,(lprint "prepare screenshot capture")
	      (setf sct (mss.mss))

	      #+nil (do0 (setf x_start 0
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
				 ))))))
	      (do0 
	       (do0 (setf scr (sct.grab (dictionary :left 5 :top 22 :width 640 :height 384))
			  img (np.array scr))
		    (imshow (string "output")
			    img))
	       #+nil (setMouseCallback (string "output") mouse_crop))

	      (do0
	       ,(lprint "open templates")
	       (setf res (list))
	       ,@(loop for e in l-template
			 collect
			 (destructuring-bind (name &optional x y screen) e
			   `(do0
			     (setf fn  (string ,(format nil "img/~a.jpg" name)))
			     (setf ,name (imread fn
						 cv2.IMREAD_COLOR))
			     ,(if (and x y)
				  `(do0
				    (setf ,(format nil "~a_x" name) ,x)
				    (setf ,(format nil "~a_y" name) ,y)
				    (setf ,(format nil "~a_screen_x" name) ,(first screen))
				    (setf ,(format nil "~a_screen_y" name) ,(second screen))
				    (res.append (dictionary :x ,x
							    :y ,y
							    :sx ,(first screen)
							    :sy ,(second screen)
							    :name ,name
							    :fn fn)))
				  `(comments "no x y coords"))
			     )))
	       (setf dft (pd.DataFrame res))
	       )
	      (do0
	       ,(lprint "compute transform between scrcpy and device coordinates")
	       ,(let ((l `((offset_x 0)
			   (offset_y 0)
			   (scale_x 1)
			   (scale_y 1))))
		  `(do0
		    (def trafo (pars)
		      (setf (ntuple ,@(mapcar #'first l))
			    pars)
		      (return (np.concatenate (list (- dft.sx (* scale_x (- dft.x offset_x)))
				     (- dft.y (* scale_y (- dft.y offset_y))
					)))))
		    (setf sol
			  (scipy.optimize.least_squares trafo (tuple ,@(mapcar #'second l))
							:method (string "lm"))
			  )
		    (print sol))))
	      #+nil (do0 (setf pause (imread (string "img/pause.jpg") cv2.IMREAD_COLOR))
		   (setf Play (imread (string "img/Play.jpg") cv2.IMREAD_COLOR)))
	      
	      (while True
		     (do0 (setf scr (sct.grab (dictionary :left 5 :top 22 :width 640 :height 384))
				imga (np.array scr)
				img (aref imga 
					  ":" ":" (slice "" -1))
				)
			  
		          #+nil (do0
			   (setf res (matchTemplate img Play
						    cv2.TM_CCORR_NORMED
						    ;cv2.TM_SQDIFF
						    )
				 (ntuple w h ch) Play.shape
				 (ntuple mi ma miloc maloc ) (minMaxLoc res)
				 center (+ (np.array maloc)
					   (* .5 (np.array (list h w)))))
			   (circle imga  ("tuple" (dot center (astype int)))
				   20 (tuple 0 0 255) 5))
			  ,@(loop for e in l-template
				  collect
				  (destructuring-bind (name &optional x y screen) e
				    (if x
				      `(do0
				      (setf res (matchTemplate img ,name
							       cv2.TM_CCORR_NORMED)
					    (ntuple w h ch) (dot ,name shape)
					    (ntuple mi ma miloc maloc ) (minMaxLoc res)
					    center (+ (np.array maloc)
						      (* .5 (np.array (list h w)))))
				      (circle imga  ("tuple" (dot center (astype int)))
					      20 (tuple 0 0 255) 5)
				      ,(lprint (format nil "~a" name)
					       `(center)))
				      `(comment "no"))))

			  (imshow (string "output")
			    imga))
		     
		     (when (== (& (cv2.waitKey 25) #xff)
			       (ord (string "q")))
		       (destroyAllWindows)
		       (setf running False)
		       break
		       )
		     )
	      )
	     
	     (scrcpy.terminate)
	)))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

