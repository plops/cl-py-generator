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
      
      ;jax
      ;gr
					;(xr xarray)
      ;;matplotlib
					;(s skyfield)
      ;;(ds dataset)
					; cv2
      ;datoviz
      ))
  (let ((nb-file "source/01_play.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play01"))
      
      (python (do0
	       
	       "#export"
	       (do0
		
					;"%matplotlib notebook"
		 #+nil (do0
		      
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
			  ;skyfield.api
			  ;skyfield.data
					;skyfield.data.hipparcos
			  ;(jnp jax.numpy)
			  ;jax.config
			  ;jax.scipy.optimize
			  ;jax.experimental.maps
			  ;jax.numpy.linalg
			  ;jax.nn
			  
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
	(comments "https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html")
	(setf fns ("list"
<<<<<<< HEAD
		   (dot (pathlib.Path (string "/home/martin/src/opencv-4.5.5/samples/data/"))
			(glob (string "left*.jpg")))))
	(display fns)
=======
		   (dot (pathlib.Path (string "data/"
					      #+nil "/home/martin/src/opencv-4.5.5/samples/data/"))
			(glob (string "left*.jpg")))))
	(print fns)
>>>>>>> 769798e8f26ac2d0e15b22b75e16181b557d2d9b

	)
       )

      (python
       (do0
	"#export"
	(setf criteria
	      (tuple
	       (+ cv.TERM_CRITERIA_EPS
		  cv.TERM_CRITERIA_MAX_ITER
		  )
	       30
	       .001))
	(setf objp (np.zeros (list (* 6 7) 3)
			     np.float32)
	      (aref objp ":" (slice "" 2))
	      (dot np (aref mgrid (slice 0 7)
			     (slice 0 6))
		   T (reshape -1 2))
	      objpoints (list)
	      imgpoints (list))
	(for (fn fns)
	     (setf img (cv.imread (str fn))
		   gray (cv.cvtColor img cv.COLOR_BGR2GRAY))
	     (setf (ntuple ret corners)
		   (cv.findChessboardCorners gray
					     (list 7 6)
					     None))
	     (when ret
	       (objpoints.append objp)
	       (setf corners2 (cv.cornerSubPix gray
					       corners
					       (list 11 11)
					       (list -1 -1)
					       criteria))
	       (imgpoints.append corners)
	       (cv.drawChessboardCorners img (list 7 6)
					 corners2 ret)
	       (cv.imshow (string "img") img)
	       (cv.waitKey 500)))
	(cv.destroyAllWindows))
       )

      (python
       (do0
	"#export"
	(setf (ntuple ret mtx dist rvecs tvecs)
	      (cv.calibrateCamera
	       objpoints
	       imgpoints
	       (aref gray.shape (slice "" "" -1))
	       None None
	       ))
<<<<<<< HEAD
	(display mtx)))
=======
	(print mtx)))
>>>>>>> 769798e8f26ac2d0e15b22b75e16181b557d2d9b

      (python
       (do0
	"#export"
	(setf img (cv.imread (str (aref fns 0)))
	      (ntuple h w) (dot img (aref shape (slice "" 2)))
	      (ntuple new_mtx roi)
	      (cv.getOptimalNewCameraMatrix mtx
					    dist
					    (list w h)
					    1 (list w h)))
<<<<<<< HEAD
	(display new_mtx)))
=======
	(print new_mtx)))
>>>>>>> 769798e8f26ac2d0e15b22b75e16181b557d2d9b

      (python
       (do0
	"#export"
	(setf dst (cv.undistort img mtx dist None new_mtx)
	      (ntuple x y w h) roi
	      dst (aref dst (slice y (+ y h))
			(slice x (+ x w))))
	(do0
	 (cv.imshow (string "dst") dst)
<<<<<<< HEAD
	 (cv.waitKey 5000)
	 (cv.destroyAllWindows))))
=======
	 (do0 (cv.waitKey 5000)
	      (cv.destroyAllWindows)))))
>>>>>>> 769798e8f26ac2d0e15b22b75e16181b557d2d9b

      (python
       (do0
	"#export"
	(setf res 0.0)
	(for (i (range (len objpoints)))
	     (setf (ntuple imgpoints2 _)
		   (cv.projectPoints
		    (aref objpoints i)
		    (aref rvecs i)
		    (aref tvecs i)
		    mtx
		    dist))
	     (setf err (/ (cv.norm
			   (aref imgpoints i)
			   imgpoints2
			   cv.NORM_L2)
			  (len imgpoints2)))
	     (incf res err))
	(print (dot (string "mean reprojection error: {:5.3f}px")
		    (format (/ res
			       (len objpoints)))))))
      
      ))))



