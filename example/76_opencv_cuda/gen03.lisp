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
      rawpy
      ))
  (let ((nb-file "source/03_load_dng.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play03_dng"
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
      #+nil
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
	     (setf d (cv.aruco.getPredefinedDictionary (dot cv aruco
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
		   board (cv.aruco.CharucoBoard_create squares_x squares_y square_length marker_length d)
		   out_size (tuple ,screen-w2 ,screen-h2)
		   board_img (board.draw out_size)
		   steps_x 5
		   steps_y 5)
	     #+nil (cv.imwrite (string "charuco1.png")
			       board_img)
	     (do0 (setf w (string "board"))
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
	(do0 (cv.waitKey 500)
	      (cv.destroyAllWindows))
	))

      (python
       (do0
	"#export"
	(setf fns ("list"
		   (dot pathlib
			(Path (string "/home/martin/stage/cl-py-generator/example/76_opencv_cuda/source/calib/"))
			(glob (string "APC*.dng")))))
	(do0
	 (setf res (list))
	 (for (fn (tqdm.tqdm fns))
	  (with (as (rawpy.imread (str fn))
		    raw)
		(res.append (raw.postprocess)))))))))))



