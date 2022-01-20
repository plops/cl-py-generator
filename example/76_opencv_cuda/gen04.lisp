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
      rawpy
      ))
  (let ((nb-file "source/04_load_jpg.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play04_jpg"
	       (comments "pip3 install --user opencv-python opencv-contrib-python rawpy tqdm")))
      (python (do0
	       "#export"
	       (do0
					
		#+nil (do0
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
			  ))
		#+nil
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
		   square_length 2 ;; in m
		   marker_length 1 ;; in m
		   board (cv.aruco.CharucoBoard_create squares_x squares_y square_length marker_length aruco_dict)
		   out_size (tuple ,screen-w2 ,screen-h2)
		   board_img (board.draw out_size)
		   steps_x 5
		   steps_y 5)
	     #+nil (cv.imwrite (string "charuco1.png")
			       board_img)
	     #+nil (do0 (setf w (string "board"))
		  (cv.namedWindow w
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
	 (setf xs_fn  (string "calib/checkerboards.nc") )
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
			   (Path (string "/home/martin/stage/cl-py-generator/example/76_opencv_cuda/source/calib/"))
			   (glob (string "APC*.dng")))))
	   (do0
	    (setf res (list))
	    (for (fn (tqdm.tqdm fns))
		 (with (as (rawpy.imread (str fn))
			   raw)
		       (res.append (raw.postprocess))))
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
			       (string "ch"))
		   :coords (dictionary
			    :frame (np.arange (aref data.shape 0))
			    :h (np.arange (aref data.shape 1))
			    :w (np.arange (aref data.shape 2))
			    :ch (np.arange (aref data.shape 3)))))))
	    (xs.to_netcdf xs_fn))
	   (do0
		   (print (dot (string "duration loading from dng and saving netcdf {:4.2f}s")
			       (format (- (time.time) start))))))))))
      (python
       (do0
	"#export" 
	(setf w (string "cb"))
	(cv.namedWindow w
			cv.WINDOW_NORMAL ;AUTOSIZE
			)
	(cv.resizeWindow w 800 600)
	(do0
	 (setf decimator 0)
	 (setf all_corners (list)
		     all_ids (list))
	 (for (frame (range (len xs.frame)))
	      (do0
	       (setf rgb (dot xs
			      (aref cb frame "...")
			      values)
		     gray (cv.cvtColor rgb cv.COLOR_BGR2GRAY))
	       (setf markers (cv.aruco.detectMarkers gray aruco_dict))
	       
	       (when (< 0 (len (aref markers 0)))
		 (setf (ntuple corners ids num)
		       (cv.aruco.interpolateCornersCharuco (aref markers 0)
							      (aref markers 1)
							      gray board) )
		 (when (and (is corners "not None")
				 (is ids "not None")
				 (< 3 corners)
				 (== 0 (% decimator 3)))
			(all_corners.append corners)
			(all_ids.append ids))
		 (cv.aruco.drawDetectedMarkers gray
					       (aref markers 0)
					       (aref markers 1)))

	      
	       (do0 
		(cv.imshow
		 w
		 gray)
		(cv.setWindowTitle w
				   (dot (string "frame {}")
					(format frame)))
		(cv.waitKey 20))
	       (incf decimator))))
	(try (setf cal (cv.aruco.calibrateCameraCharuco
			      all_corners all_ids board gray.shape
			      None None))
		   ("Exception as e"
		    (print e)
		    pass))
	(cv.destroyAllWindows)
	))))))



 
