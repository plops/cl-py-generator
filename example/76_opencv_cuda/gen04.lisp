(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/76_opencv_cuda")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `((np numpy)
      (cv cv2)
      (pd pandas)
      (xr xarray)
      lmfit
      ;rawpy
      ))
  (defun lprint (&optional rest)
    `(print (dot (string ,(format nil "~{~a={}~^\\n~}" rest))
                 (format  ;(- (time.time) start_time)
                  ,@rest))))
  (let ((nb-file "source/04_load_jpg.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play04_jpg"
	       (comments "pip3 install --user opencv-python opencv-contrib-python tqdm xarray pandas h5netcdf lmfit")))
      (python (do0
	       "#export"
	       (do0
					
		#-nil (do0
			;"%matplotlib notebook"
		      	(imports (matplotlib))
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
					pathlib
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
			  ;argparse
			  ;(sns seaborn)
					;(jnp jax.numpy)
					;jax.config
					;jax.scipy.optimize
					;jax.experimental.maps
					;jax.numpy.linalg
					;jax.nn
			  cv2.aruco
			  tqdm
			  decimal
			  ))
		#-nil
			(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
		#+nil
		(imports-from (jax.experimental.maps
			       xmap))

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
	"#export"
	(comments "https://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/")
	,(let* ((screen-fac 1)
		(screen-w (* screen-fac 1920))
		(screen-h (* screen-fac 1080))
		(squares-fac 3)
		(squares-x (* squares-fac 16))
		(squares-y (* squares-fac 9))
		;; add one row and column for shifting
		(square-a (/ screen-w squares-x))
		(screen-w2 (+ screen-w square-a))
		(screen-h2 (+ screen-h square-a))
		(squares-x2 (+ 1 squares-x))
		(squares-y2 (+ 1 squares-y))
		(n-squares (/ (* squares-x2
				 squares-y2)
			      2))) 
	   `(do0
	     (setf aruco_dict (cv.aruco.getPredefinedDictionary (dot cv aruco
							    ,(format nil "DICT_4X4_~a"
								     (cond ((< n-squares 50)
									    50)
									   ((< n-squares 100)
									    100)
									   ((< n-squares 250)
									    250)
									   ((< n-squares 1000)
									    1000)
									   (t (break "too many"))))))
		   squares_x ,squares-x2
		   squares_y ,squares-y2
		   square_length 2 ;; in m, i should measure this after displaying the pattern
		   marker_length 1 ;; in m
		   board (cv.aruco.CharucoBoard_create squares_x squares_y square_length marker_length aruco_dict)
		   out_size (tuple ,screen-w2 ,screen-h2)
		   board_img (board.draw out_size)
		   steps_x 5
		   steps_y 5)
	     (print (dot (string "aruco dictionary can correct at most {} bits")
			 (format aruco_dict.maxCorrectionBits)))
	     #+nil (cv.imwrite (string "charuco1.png")
			       board_img)
	     #+nil (do0 (setf w (string "board"))
		  (cv.namedWindow w1
				  cv.WINDOW_NORMAL)
		  (cv.setWindowProperty
		   w cv.WND_PROP_FULLSCREEN
		   cv.WINDOW_FULLSCREEN)
		  #+nil
		  (cv.resizeWindow w ,screen-w ,screen-h)
		  (for(y  (dot (np.round (np.linspace 0 ,(- square-a 1) steps_y)) (astype int)))
		   (for (x (dot (np.round (np.linspace 0 ,(- square-a 1) steps_x)) (astype int)))
			(do0
					;(cv.moveWindow w (* 5 x) 5)
			 (cv.imshow w
				    (aref board_img (slice y (- ,screen-h y))
					  (slice x (+ ,screen-w x))))
			 (cv.waitKey 0)))))))
	#+nil (do0 (cv.waitKey 500)
	      (cv.destroyAllWindows))
	))

      (python
       (do0
	"#export"
	(do0
	 (setf xs_fn  (string "calib2/checkerboards.nc") )
	 (if (dot pathlib (Path xs_fn) (exists))
	     (do0
	      (setf start (time.time))
	      (setf xs (xr.open_dataset xs_fn))
	      (print (dot (string "duration loading from netcdf {:4.2f}s")
			  (format (- (time.time) start)))))
	     (do0
	      (do0
	      (setf start (time.time))
	      )
	   (setf fns ("list"
		      (dot pathlib
			   (Path (string "calib2/"))
			   (glob (string "*.jpg")))))
	   (do0
	    (setf res (list))
	    (for (fn (tqdm.tqdm fns))
		 
		 (setf rgb (cv.imread (str fn)))
		 (setf gray (cv.cvtColor rgb cv.COLOR_BGR2GRAY))
		 (res.append gray))
	    (setf
	     data (np.stack res 0)
	     xs (xr.Dataset
		 (dictionary
		  :cb (xr.DataArray
		   :data data
		   ;; 10 3024 4032 3
		   :dims (list (string "frame")
			       (string "h")
			       (string "w")
			       ;(string "ch")
			       )
		   :coords (dictionary
			    :frame (np.arange (aref data.shape 0))
			    :h (np.arange (aref data.shape 1))
			    :w (np.arange (aref data.shape 2))
			    ;:ch (np.arange (aref data.shape 3))
			    )))))
	    (xs.to_netcdf xs_fn))
	   (do0
		   (print (dot (string "duration loading from jpg and saving netcdf {:4.2f}s")
			       (format (- (time.time) start))))))))))

      (python
       (do0
	"#export"
	(comments "this will be used by interpolateCornersCharuco"
		  "initially with out camera matrix ApproxCalib will be run"
		  "you can execute the next cell again after camera_matrix has been found to run LocalHom")
	(setf camera_matrix None
	      distortion_params None)))
      (python
       (do0
	"#export"
	(comments "flags for camera calibration that influence the model (fix fx=fy, number of coefficents for distortion)")
	(comments "fix K1,K2 and K3 to zero (no distortion)")
	(setf calibrate_camera_flags_general
	      (logior
	       cv.CALIB_ZERO_TANGENT_DIST
	       cv.CALIB_FIX_ASPECT_RATIO
	       
	     ;  cv.CALIB_FIX_K1
	       cv.CALIB_FIX_K2
	       cv.CALIB_FIX_K3))))
      
      (python
       (do0
	(comments "use this for the second run to make use of existing camera matrix")
	(setf calibrate_camera_flags (logior cv.CALIB_USE_INTRINSIC_GUESS
					     calibrate_camera_flags_general))))
      (python
       (do0
	"#export"
	(comments "use these flags for the first run")
	(setf calibrate_camera_flags (logior
				      
				      calibrate_camera_flags_general))))

      
      (python
       (do0
	"#export"
	(comments "collect corners for each frame")
	(setf do_plot
					;True
					False
					)
	(setf save_figure False)
	(setf w (string "cb"))

	(when do_plot
	 (do0 (cv.namedWindow w
			      cv.WINDOW_NORMAL ;AUTOSIZE
			      )
	      (cv.resizeWindow w 1600 900)))
	(do0
	 ; (setf decimator 0)
	 (setf all_corners (list)
	       all_ids (list)
	       all_rejects (list))
	 (setf aruco_params (cv.aruco.DetectorParameters_create))
	
	 (for (frame (tqdm.tqdm (range (len xs.frame))))
	      (do0
	       (setf gray (dot xs
			       (aref cb frame "...")
			       values)
					;gray (cv.cvtColor rgb cv.COLOR_BGR2GRAY)
		     )
	       (comments "rejected_points[NR-1].shape = 1 4 2, NR=566")
	       (setf (ntuple corners ids rejected_points)
		     ;; markers
		     (cv.aruco.detectMarkers :image gray
					     :dictionary aruco_dict
					     :parameters aruco_params
					     :cameraMatrix camera_matrix
					     :distCoeff distortion_params))
	       
	       (when (< 0 (len corners))
		 (comments "corners[N-1].shape = 1 4 2, for each of N markers provide 4 corners"
			   "ids[0] = 653, id for this particular marker"
			   "cameraMatrix (optional) [fx 0 cx; 0 fy c0; 0 0 1]"
			   "distCoeffs (optional 4,5,8 or 12 elements) k1 k2 p1 p1 [k3 [k4 k5 k6] [s1 s2 s3 s4]]"
			   "minMarkers (optional) number of adjacent markers that must be detected to return corner"
			   )
		 (setf (ntuple charuco_retval int_corners int_ids)
		       (cv.aruco.interpolateCornersCharuco :markerCorners corners
							   :markerIds ids
							   :image gray
							   :board board
							   :cameraMatrix camera_matrix) )
		 (when (< 20 charuco_retval)
		   (comments "found at least 20 squares")
		   (all_corners.append int_corners)
		   (all_ids.append int_ids)
		   (all_rejects.append rejected_points))
		 (when save_figure
		     (do0 
		      (comments "image 16 and 25 have the most recognized markers (i think)"
				"blue .. markers, index fixed to board, starts from top left, increases towards right"
				"green .. corners, fixed to board, starts from bottom left increases towards right")
		      
		      (setf img (cv.aruco.drawDetectedCornersCharuco
				 :image (cv.cvtColor gray cv.COLOR_GRAY2RGB)
				 :charucoCorners int_corners ;; sub pixel corner positions in image of checkerboard intersections
				 :charucoIds int_ids ;; corner index to draw the id (optional)
				 :cornerColor
				 (tuple 255 255 0)))
		      (cv.aruco.drawDetectedMarkers img
						    corners
						    ids
						    (tuple 0 255 0))
		      (cv.imwrite (dot (string "/dev/shm/{:02d}.jpg")
				       (format frame))
				  img)))
		 (when do_plot
		   (do0
		    

		   
		   (do0 
		    (cv.imshow
		     w
		     #+nil img
		     #-nil (aref gray (slice "" "" 4)
				 (slice "" "" 4)))
		    (cv.setWindowTitle w
				       (dot (string "frame {}")
					    (format frame)))
		    (cv.waitKey 1)))))
	       ;(incf decimator)
	       )))
	(do0
	 (comments "all_corners[0].shape = 295 1 2"
		   "all_ids[0].shape     = 295 1"))
	
	(do0
	 (try (setf (ntuple calibration
			    camera_matrix
			    distortion_params
			    rvecs
			    tvecs)
		    (cv.aruco.calibrateCameraCharuco
		     :charucoCorners all_corners
		     :charucoIds all_ids
		     :board board
		     :imageSize gray.shape
		     :cameraMatrix camera_matrix
		     :distCoeffs distortion_params
		     :flags calibrate_camera_flags))
	      ("Exception as e"
	       (print e)
	       pass))
	 (print camera_matrix)
	 (print distortion_params))

	#+nil
	(do0
	 (comments "once we have an estimate of camera matrix and distortion,"
		   "we can use refineDetectedMarkers to look for the missing markers"
		   "FIXME: subsequent calibration is not working. i read about issues with the order of ids returned by refineDetectedMarkers, might be the cause")
	 (setf res (list))
	 (for ((ntuple int_corners int_ids rejected_points)
	       (tqdm.tqdm (zip all_corners
			       all_ids
			       all_rejects)))
	      (setf (ntuple ref_corners
			    ref_ids
			    ref_rejects
			    ref_recovered_ids)
	       (cv.aruco.refineDetectedMarkers :image gray
					       :board board
					       :detectedCorners int_corners
					       :detectedIds int_ids
					       :rejectedCorners rejected_points
					       :cameraMatrix camera_matrix
					       :distCoeffs distortion_params))
	      (res.append (dictionary
			   :corner ref_corners
			   :id ref_ids
			   :reject ref_rejects
			   :recovered_id ref_recovered_ids))
	      ))
	(do0
	 (comments "board.chessboardCorners.shape 1296 3")
	 (comments
	  "all marker corners on the board"
	  "board.objPoints[685].shape 4 3;  coordinates of 4 points in CCW order,  z coordinate 0"
		   ))

	#+nil
	(do0
	 (try (setf (ntuple calibration2
			    camera_matrix2
			    distortion_params2
			    rvecs2
			    tvecs2)
		    (cv.aruco.calibrateCameraCharuco
		     :charucoCorners ref_corners
		     :charucoIds ref_ids
		     :board board
		     :imageSize gray.shape
		     :cameraMatrix camera_matrix ;None
		     :distCoeffs distortion_params ;None
		     ))
	      ("Exception as e"
	       (print e)
	       pass))
	 (print camera_matrix2)
	 (print distortion_params2))
	
	(when do_plot
	 (cv.destroyAllWindows))))
      #+nil
      (python
       (do0
	 (try (setf (ntuple calibration2
			    camera_matrix2
			    distortion_params2
			    rvecs2
			    tvecs2)
		    (cv.aruco.calibrateCameraCharuco
		     :charucoCorners ref_corners
		     :charucoIds ref_ids
		     :board board
		     :imageSize gray.shape
		     :cameraMatrix camera_matrix
		     :distCoeffs distortion_params))
	      ("Exception as e"
	       (print e)
	       pass))
	 (print camera_matrix2)
	 (print distortion_params2)))

      
      (python
       (do0
	(comments "calibration step by itself, also print fit errors for the parameters")
	(comments "intrinsics: fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,τx,τy"
		  "extrinsics: R0,T0,…,RM−1,TM−1"
		  "M .. number of frames"
		  "R_i, T_i .. concatenated 1x3 vectors")
	 (try (setf (ntuple calibration3
			    camera_matrix3
			    distortion_params3
			    rvecs3
			    tvecs3
			    intrinsic_err
			    extrinsic_err
			    view_err)
		    (cv.aruco.calibrateCameraCharucoExtended
		     :charucoCorners all_corners
		     :charucoIds all_ids
		     :board board
		     :imageSize gray.shape
		     :cameraMatrix camera_matrix
		     :distCoeffs distortion_params
		     :flags calibrate_camera_flags))
	      ("Exception as e"
	       (print e)
	       pass))
	 (print camera_matrix3)
	 (print distortion_params3)
	 (do0
	  (setf M camera_matrix3
		D distortion_params3)
	  ,(let ((l `(fx fy cx cy k1 k2 p1 p2 k3;  k4 k5 k6 s1 s2 s3 s4 τx τy
			 )))
	   `(for ((ntuple idx (tuple name val))
		 (enumerate (zip (list ,@(mapcar #'(lambda (x) `(string ,x))
						 l))
				 (list (aref M 0 0) ;; fx
				       (aref M 1 1) ;; fy
				       (aref M 0 2) ;; cx
				       (aref M 1 2) ;; cy
				       (aref D 0 0)   ;; k1
				       (aref D 0 1)   ;; k2
				       (aref D 0 2) ;; p1
				       (aref D 0 3) ;; p2
				       (aref D 0 4) ;; k3
				       ))))
	       
		(print (dot (string "{} = {}±{} ({:2.1f}%)")
			    (format name
				    (dot decimal (Decimal (dot (string "{:.4g}")
							       (format val)))
					 (normalize)
					 (to_eng_string))
				    (dot decimal (Decimal (dot (string "{:.1g}")
							       (format (dot (aref intrinsic_err idx)
									    (item)))))
					 (normalize)
					 (to_eng_string))
				    (np.abs (* 100 (/ 
					     (dot (aref intrinsic_err idx)
						  (item))
					     val)))))
		       ))))))

      
      (python
       (do0
	(comments "collect the data, so that i can implement the fit myself")
	(comments "function calibrateCameraCharuco https://github.com/opencv/opencv_contrib/blob/a26f71313009c93d105151094436eecd4a0990ed/modules/aruco/src/charuco.cpp")
	(assert (< 0 (len all_ids)))
	(assert (== (len all_ids)
		    (len all_corners)))
	(setf res (list))
	(for ((ntuple i ids) (enumerate all_ids))
	     (setf n_corners (len ids)
		   corners (aref all_corners i))
	     (assert (< 0 n_corners))
	     (assert (== n_corners (len corners)))
	     (for (j (range n_corners))
		  (setf point_id (aref ids j))
		  (assert (<= 0 point_id))
		  (assert (< point_id (len board.chessboardCorners)))
		  (res.append
		   (dictionary :frame_idx i
			       :corner_idx j
			       :point_id (dot point_id (item))
			       ;; checkerboard
			       :x (aref (aref board.chessboardCorners point_id) 0 0)
			       :y (aref (aref board.chessboardCorners point_id) 0 1)
			       ;; camera
			       :u (aref corners j 0 0)
			       :v (aref corners j 0 1)
			       ))))
	(setf df (pd.DataFrame res))
	df))
      
      (python
       (do0
	(comments "plot the coordinates")
	(plt.scatter df.x df.y)
	(plt.xlim 0 (* 2 squares_x))
	(plt.ylim 0 (* 2 squares_y))
	(grid)))
      
      (python
       (do0
	(comments "heatmap of point density/coverage of the camera")
	(setf fac 3)
	(plt.hist2d df.u df.v :bins (list (np.linspace 0 (+ -1 (dot xs w (max) (item))) (* 16 fac))
					  (np.linspace 0 (+ -1 (dot xs h (max) (item))) (* 9 fac)))
		    :cmap (string "cubehelix"))
	(plt.colorbar)
	(do0
	 (plt.xlim 0 (+ -1 (dot xs w (max) (item))))
	 (plt.ylim 0 (+ -1 (dot xs h (max) (item)))))
	(grid)))


      
      ,@(let (
	     ;; first list scalar parameters
	    #+nil (merit-params `(;; intrinsics (same for all views)
			     (:name fx :start 3336)
			     (:name fy :start 3336)
			     (:name cx :start 2008)
			     (:name cy :start 1436)
			     (:name k1 :start  .0504)
			     (:name k2 :start -.2428)
			     ;(:name p1 :start -.00016)
			     ;(:name p2 :start  .00032)
			     (:name k3 :start .311)
			     ;; unknowns per frame / view
			     (:name rvecs :start .1 :shape (n_frames 2))
			     (:name tvecs :start .1 :shape (n_frames 2))
			     ))
	     #+nil (merit-fun `(do0
			  ;; object coordinate Q on the checkerboard
			  (setf Q (np.array (list (list X)
						  (list Y)
						  (list 1))))
			  ;; (setf q (np.matmul H Q)) ;; homography matrix H=MW
			  
			  ;; camera matrix M
			  (setf M (np.array (list (list fx 0 cx)
						  (list 0 fx cy)
						  (list 0 0 1))))
			  ;; physical transformation matrix W (rotation and translation)
			  (setf W (np.hstack r1 r2 trans))
			  
			  ;; image coordinate q in the perfect pinhole camera q=(x_p,y_p)
			  (setf q (np.matmul M (np.matmul W Q)))
			  ;; image coordinates q_d=(x_d,y_d) in the camera with distortions
			  (setf q (+ (* (+ 1
					   (* k1 (** r 2))
					   (* k2 (** r 4))
					   (* k3 (** r 6)))
					(np.array (list x_d y_d)))
				    #+nil  (np.array (list (+ (* 2 p1 x_d y_d)
							(* p2 (+ (** r 2)
								 (* 2 (** x_d 2)))))
						     (+ (* p1 (+ (** r 2)
								 (* 2 (** y_d 2))))
							(* 2 p2 x_d y_d))))))
			  
			  ))
	     )
	  `((python
	     (do0
	      (comments "try the transform with a single corner")
	      (setf frame_idx 24)
	      (setf rvec (aref rvecs frame_idx)
		    tvec (aref tvecs frame_idx))
	      (setf (ntuple R3 R3jac) (cv.Rodrigues :src rvec))
	      (setf W (np.hstack (list (aref R3 ":" (slice 0 2))
				       tvec)))
	      (do0
	       (setf Mcam camera_matrix3
		     D distortion_params3)
	       ,(let ((l `(fx fy cx cy k1 k2 p1 p2 k3;  k4 k5 k6 s1 s2 s3 s4 τx τy
			      ))
		      (ll `((aref Mcam 0 0)		    ;; fx
			    (aref Mcam 1 1)		    ;; fy
			    (aref Mcam 0 2)		    ;; cx
			    (aref Mcam 1 2)		    ;; cy
			    (aref D 0 0)		    ;; k1
			    (aref D 0 1)		    ;; k2
			    (aref D 0 2)		    ;; p1
			    (aref D 0 3)		    ;; p2
			    (aref D 0 4)		    ;; k3
			    )))
		  `(do0
		    ,@(loop for name in l and val in ll
			    collect
			    `(setf ,name ,val)))))
	      (comments "https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d Detailed description explains how to compute r"
			"documentation of undistortPoints explains how modify coordinates: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga55c716492470bfe86b0ee9bf3a1f0f7e ")
	      (setf M (np.array (list (list fx 0 cx)
						  (list 0 fx cy)
						  (list 0 0 1))))
	      
	      (setf df_ (aref df (== df.frame_idx frame_idx)))
	      (setf row (dot df_
			     (aref iloc 120)))
	      (setf res (list))
	      (for ((ntuple idx row) (df_.iterrows))

	       (do0
		(setf Q (np.array (list (list row.x)
					(list row.y)
					(list 1))))
		(setf WQ (np.matmul W Q))
		(setf MWQ (np.matmul M WQ))
	       
		(comments "divide by 3rd coordinate, this will return x_prime y_prime")
		(setf mwq (/ MWQ (aref MWQ 2)))
		(setf uv (np.array (list (list row.u )
					 (list row.v )))
		      center (np.array (list (list cx)
					     (list cy))))
		(setf uv_pinhole (cv.undistortPoints :src uv
						     :cameraMatrix camera_matrix
						     :distCoeffs distortion_params))
		;(setf uvc (- uv center))
	        #+nil
		(do0 (setf r (np.linalg.norm (aref uvc (slice "" 2))))
		    	(setf uv_pinhole (* (+ 1
						       (* k1 (** r 2))
						       (* k2 (** r 4))
						       (* k3 (** r 6)))
						    uv)))
					;,(lprint `(Q W WQ M MWQ mwq uv r uv_pinhole))
		(res.append
		 (dictionary
		  :mwq mwq
		  :uv uv
		  ;:uvc uvc
		  ;:r r
		  :uv_pinhole uv_pinhole
		  ))
		))
	      (setf dft (pd.DataFrame res))
	      dft))
	    
	    ,@(loop for e in `((:col uv :x u :y v)
			       ;(:col uvc :x u-c :y v-c)
			       (:col mwq :x x_prime :y y_prime)
			       (:col uv_pinhole :x x_pprime :y y_pprime)
			       )
		    collect
		    (destructuring-bind (&key col x y) e
		      `(python
		       (do0
			(plt.scatter (dot np (stack (dot dft ,col values))
					  (aref (squeeze) ":" 0))
				     (dot np (stack (dot dft ,col values))
					  (aref (squeeze) ":" 1))
				     :s 1)
			(grid)
			(do0
		(plt.xlim 0 (+ -1 (dot xs w (max) (item))))
		(plt.ylim 0 (+ -1 (dot xs h (max) (item)))))
			(xlabel (string ,x))
			(ylabel (string ,y))))
		      ))
	    
	    (python
	     (do0
	      ,@(loop for (col marker) in `((uv "+")
					    (mwq "x"))
		      collect
		      `(plt.scatter (dot np (stack (dot dft ,col values))
					 (aref (squeeze) ":" 0))
				    (dot np (stack (dot dft ,col values))
					 (aref (squeeze) ":" 1))
				    :marker (string ,marker)))
			
	      (grid)
	       (do0
		(plt.xlim 0 (+ -1 (dot xs w (max) (item))))
		(plt.ylim 0 (+ -1 (dot xs h (max) (item)))))
			(xlabel (string "x"))
			(ylabel (string "y"))))
	    

	    
	    ,@(let ((coord-pairs `((uv mwq)
				   (uv uv_pinhole)
				   (mwq uv_pinhole))))
		(loop for (A B) in coord-pairs
			collect
			`(python
			  (do0
			   (comments "quiver plot to show mismatch between camera coordinates and transformed object coordinates")
			   (setf x0  (dot np (stack (dot dft ,A values))
					  (aref (squeeze) ":" 0))
				 y0 (dot np (stack (dot dft ,A values))
					 (aref (squeeze) ":" 1))
				 x1  (dot np (stack (dot dft ,B values))
					  (aref (squeeze) ":" 0))
				 y1  (dot np (stack (dot dft ,B values))
					  (aref (squeeze) ":" 1))
				 dx (- x0 x1) 
				 dy (- y0 y1)
				 s 1)
			   (plt.quiver x0
				       y0
				       (* s dx)
				       (* s dy)
				       )
			   (plt.scatter (list cx)
					(list cy)
					:marker (string "x")
					:color (string "r"))
			   (grid)
			   (title (string ,(format nil "compare ~a and ~a" A B)))

			   #-nil (do0
				  (plt.xlim 0 (+ -1 (dot xs w (max) (item))))
				  (plt.ylim 0 (+ -1 (dot xs h (max) (item))))
				  
				  (plt.axis (string "equal")))
					;(dot plt (gca) (set_aspect (string "auto")))
			   (do0
			    (xlabel (string "x"))
			    (ylabel (string "y")))))))

	    
	    (python
	     (do0
	      (comments "show distortion for the corners in this particular frame")
	      (setf r (np.sqrt (+ (** (- x0 cx) 2 )
				  (** (- y0 cy) 2 ))))
	      (plt.scatter r (np.sqrt (+ (** dx 2)
					 (** dy 2))))
	             (do0
	
		      (do0
		       (xlim 0 2500)
		       (grid)
		(xlabel (string "r"))
		(ylabel (string "dr"))))
	      ))
	    
	    (python
	     (do0
	      (comments "distortion should be monotonic (barrel distortion decreases). if not then consider calibration a failure")
	      (setf r (np.linspace 0 1))
	      (plot r (+ 1
				    (* k1 (** r 2))
				    (* k2 (** r 4))
				    (* k3 (** r 6))))
	      (grid)
	      (xlabel (string "r"))
	      (ylabel (string "distortion factor"))))
	    #+nil
	    (python
	   (do0
	    (def merit (params grid_x grid_y cam_x cam_y)
	      (setf n_frames (len grid_x)
		    count_start ,(loop named outer
				       for e in merit-params
				       and i from 0
				       do
					  (destructuring-bind (&key name start (shape 1)) e
					    (unless (eq shape 1)
					      (return-from outer i)))))
	      ,@(loop for e in merit-params
		      and i from 0
		      collect
		      (destructuring-bind (&key name start (shape 1)) e
			`,(if (eq shape 1)
			      `(setf ,name (aref params ,i))
			      `(do0
				(setf ,name
				      (dot (aref params (slice count_start (+ count_start (* ,@shape))))
					   (reshape (list ,@shape))))
				(incf count_start  (* ,@shape)))
			     
			      )))
	      ,merit-fun)
	    (do0
	     (setf grid_x df.x
		   grid_y df.y
		   cam_x df.u
		   cam_y df.v)
	     (setf n (len grid_x))
	     ,(let ((scalar-params
		      (loop for e in merit-params
			    and i from 0
			    collect
			    (destructuring-bind (&key name start (shape 1)) e
			      (if (eq shape 1)
				  start
				  (loop-finish))))))
		`(setf params0 (+  (list ,@scalar-params)
				   ,@(loop for e in (subseq merit-params (length scalar-params))
					   collect
					   (destructuring-bind (&key name start (shape 1)) e
					     `(* (list ,start) (* ,@shape)))
					   ))))
	     (setf fitter (lmfit.Minimizer merit params0)))
	    ))))
      
      ))))



 
