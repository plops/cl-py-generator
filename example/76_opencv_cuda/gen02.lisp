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
      ))
  (let ((nb-file "source/02_show_checkerboard.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp play02"
	       (comments "pip3 install --user opencv-python opencv-contrib-python")))
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
	(setf d (cv.aruco.getPredefinedDictionary cv.aruco.DICT_4X4_250)
	      squares_x 16
	      squares_y 9
	      square_length 2 ;; in m
	      marker_length 1 ;; in m
	      board (cv.aruco.CharucoBoard_create squares_x squares_y square_length marker_length d)
	      out_size (tuple (// 1920 2) (// 1080 2))
	      board_img (board.draw out_size))
	(cv.imwrite (string "charuco1.png")
		    board_img)
	(cv.imshow (string "board")
		   (np.tile board_img
			    (list 1 1 3)))
	))))))



