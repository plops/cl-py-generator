(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/38_mediapipe")
  (defparameter *code-file* "run_00_start")
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
			   (mp mediapipe)
			  ; jax
			   ;jax.random
					;jax.config
			   copy
			   
			   ))
		"from mss import mss"
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

		 (do0
		  (def calc_bounding_rect (image landmarks)
		    (setf (ntuple height width)
			  (aref image.shape "0:2")
			  )
		    (setf landmark_array (np.empty (tuple 0 2)
						   int))
		    (for (landmark
			  landmarks.landmark)
			 (setf lx (min (int (* landmark.x width))
				       (- width 1)))
			 (setf ly (min (int (* landmark.y height))
				       (- height 1)))
			 (setf landmark_point (list (np.array (tuple lx ly))))
			 (setf landmark_array (np.append
					       landmark_array
					       landmark_point
					       :axis 0)))
		    (setf (ntuple x y w h)
			  (cv.boundingRect landmark_array))
		    (return (list x
				  y
				  (+ x w)
				  (+ y h))))
		  (def draw_bounding_rect (image brect)
		    (cv.rectangle image
				  (tuple (aref brect 0)
					 (aref brect 1))
				  (tuple (aref brect 2)
					 (aref brect 3))
				  (tuple 0 255 0)
				  2)
		    (return image)
		    )
		  (def draw_landmarks (image landmarks)
		    (setf (ntuple height width)
			  (aref image.shape "0:2")
			  )
		    (do0
		     ;(setf landmark_point (list))
		     (for (l landmarks.landmark)
			  (when (or (< l.visibility 0)
				    (< l.presence 0))
			    continue)
			  (do0 (setf lx (min (int (* l.x width))
					    (- width 1)))
			       (setf ly (min (int (* l.y height))
					     (- height 1))))
			  ;(landmark_point.append (tuple lx ly))
			  (cv.circle image (tuple lx ly)
				     1 (tuple 0 255 0)
				     1)))
		    (return image)))
		 #+nil (do0
		  (setf cap (cv.VideoCapture (string "/dev/video0"))))
		 (do0
		  (setf bbox (dict ((string "top") 180)
				   ((string "left") 10)
				   ((string "width") 512)
				   ((string "height") 512))
			sct (mss)))
		 (do0
		  
		  (setf mp_face_mesh mp.solutions.face_mesh
			mp_drawing mp.solutions.drawing_utils)
		  (setf face_mesh (dot mp_face_mesh
				       (FaceMesh
					:max_num_faces 3
					:min_detection_confidence .7
					:min_tracking_confidence .5))))
		 (do0
		  (setf drawing_spec (mp_drawing.DrawingSpec :thickness 1 :circle_radius 1))
		 )
		 (do0
		  (while True
			 #+nil
			 (do0 (setf (ntuple ret image)
				    (cap.read))
			      (unless ret
				break))
			 (do0
			  (setf sct_img (sct.grab bbox))
			  (setf image (np.array sct_img)))
			 ;(setf debug_image (copy.deepcopy image))
			 (do0 (setf image (cv.cvtColor image cv.COLOR_BGR2RGB))
			      (setf debug_image (copy.deepcopy image))
			      (setf image.flags.writeable False)
			      (setf results (face_mesh.process image)))

			 (do0
			  (when results.multi_face_landmarks
			    (for (face_landmarks results.multi_face_landmarks)
				 (mp_drawing.draw_landmarks
				  :image debug_image
				  :landmark_list face_landmarks
				  :connections mp_face_mesh.FACE_CONNECTIONS
				  :connection_drawing_spec drawing_spec))))
			 #+nil
			 (do0
			  (unless (is results.multi_face_landmarks
				      None)
			    (for (face_landmarks results.multi_face_landmarks)
				 (setf brect (calc_bounding_rect
					      debug_image
					      face_landmarks))
				 (setf debug_image
				       (draw_landmarks debug_image face_landmarks))
				 #+nil (setf debug_image
				       (draw_bounding_rect
					debug_image brect)))))
			 (do0
			  (setf key (cv.waitKey 1))
			  (when (== 27 key)
			    break)
			  )
			 (cv.imshow (string "face mesh")
				    debug_image))
		  )
		 (do0
		  (cap.release)
		  (cv.destroyAllWindows))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

