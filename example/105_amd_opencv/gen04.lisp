(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "105_amd_opencv")
  (defparameter *idx* "04")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    msg
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    vars)))
                   (format  (- (time.time) start_time)
                            ,@vars)))))

  
  
  (let* ((notebook-name "opencv_mediapipe_face")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/python"
       (do0
	,(format nil "#|default_exp p~a_~a" *idx* notebook-name))
	 (do0
	  (comment "sudo pacman -S python-opencv rocm-opencl-runtime python-mss"
		   "python3 -m pip install --user mediapipe")
	  (imports (;	os
					;sys
			time
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
					; copy
					;re
					;json
					; csv
					;io.StringIO
					;bs4
					;requests
					mss
			
					;(np jax.numpy)
					;(mpf mplfinance)

					;argparse
					;torch
					(mp mediapipe)
					
			)))
	 
	 (setf start_time (time.time)
	       debug True)
	 (setf
	  _code_git_version
	  (string ,(let ((str (with-output-to-string (s)
				(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		     (subseq str 0 (1- (length str)))))
	  _code_repository (string ,(format nil
					    "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/"
					    *project*))
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
	  ,(lprint :vars `((cv.ocl.haveOpenCL)))
	  (setf loop_time (time.time))
	  (setf mp_drawing mp.solutions.drawing_utils
		mp_drawing_styles mp.solutions.drawing_styles
		mp_face_mesh mp.solutions.face_mesh
		)
	  (setf clahe (cv.createCLAHE :clipLimit 15.0
				      :tileGridSize (tuple 32 18)))
	  (setf drawing_spec (mp_drawing.DrawingSpec :thickness 1
						     :circle_radius 1))
	  (with (as (mp_face_mesh.FaceMesh
		     :static_image_mode  False
		     :max_num_faces 1
		     :refine_landmarks True
		     :min_detection_confidence .15
		     :min_tracking_confidence .15)
		    face_mesh)
		(comments "https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md"
			  "select the attention model using refine_landmarks option to home improve accuracy around lips, eyes and irises")
		(with (as (mss.mss)
			  sct)
		      (setf loop_start (time.time))
		      (while True
			     (do0
			      
			      (setf img (np.array (sct.grab (dictionary :top 160
									:left 0
									:width (// 1920 2)
									:height (// 1080 2))))
				    )
			      
			
			      (setf timestamp_ms (int (* 1000 (- (time.time)
								 loop_start))))
			
			
			      (setf imgr (cv.cvtColor img 
						      cv.COLOR_BGR2RGB))
			      (setf results (face_mesh.process imgr))
			      (setf lab (cv.cvtColor img 
						     cv.COLOR_RGB2LAB)
				    lab_planes (cv.split lab)
				    lclahe
				    (clahe.apply (aref lab_planes 0))
				    lab (cv.merge (list lclahe 
							(aref lab_planes 1)
							(aref lab_planes 2)))
				    imgr (cv.cvtColor lab cv.COLOR_LAB2RGB))
			      (when results.multi_face_landmarks
			       (for (face_landmarks results.multi_face_landmarks)
				    ,@(loop for e in
					    `((:connections FACEMESH_TESSELATION :spec get_default_face_mesh_tesselation_style)
					      (:connections FACEMESH_CONTOURS :spec get_default_face_mesh_contours_style)
					      (:connections FACEMESH_IRISES :spec get_default_face_mesh_iris_connections_style)
					      )
					    collect
					    (destructuring-bind (&key connections spec) e
						
						`(mp_drawing.draw_landmarks
						 :image imgr
						 :landmark_list face_landmarks
						 :landmark_drawing_spec None
						 :connections (dot mp_face_mesh ,connections)
						 :connection_drawing_spec
						 (dot 
						  mp_drawing_styles (,spec)))))
				   ))
			      (cv.imshow (string "screen")
					 imgr) 
			
			
			      (do0
			       (setf delta (- (time.time)
					      loop_time))
			       (setf target_period (- (/ 1 60.0)
						      .0001))
			       (when (< delta target_period)
				 (time.sleep (- target_period delta)
					     ))
			       (setf fps (/ 1 delta)
				     fps_wait (/ 1 (- (time.time)
						      loop_time)))
			       (setf loop_time (time.time))
			       (when (== 0 (% timestamp_ms 2000))
				 ,(lprint :vars `(fps fps_wait))))
			      (when (== (ord (string "q"))
					(cv.waitKey 1))
				(cv.destroyAllWindows)
				break))))))))))

