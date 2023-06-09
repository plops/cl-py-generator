(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "109_mediapipe_segment")
  (defparameter *idx* "01")
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

  
  (defun doc (def)
    `(do0
      ,@(loop for e in def
	      collect
	      (destructuring-bind (&key name val (unit "-") (help name)) e
		`(do0
		  (comments ,(format nil "~a (~a)" help unit))
		  (setf ,name ,val))))))
  
  (let* ((notebook-name "segment")
	 #+nil (cli-args `(
			   (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (do0
	)
       (do0
	(comments "python3 -m pip install --user mediapipe"
		  "wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
		  "16 MB download")
	#+nil
	(do0
	 
	 (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
	 (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation)
					;(xrp xarray.plot)
		   ))

					;(plt.ion)
	 (plt.ioff)
	 ;;(setf font (dict ((string size) (string 6))))
	 ;; (matplotlib.rc (string "font") **font)
	 )
	#+nil 
	(imports-from  (matplotlib.pyplot
			plot imshow tight_layout xlabel ylabel
			title subplot subplot2grid grid text
			legend figure gcf xlim ylim)
		       )
	(imports (			;	os
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
					;(np numpy)
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
					;mss
			
					;(np jax.numpy)
					;(mpf mplfinance)
					;(fft scipy.fftpack)
					;argparse
					;torch
		  (mp mediapipe)
		  
		  )))
       (imports-from (mediapipe.tasks python)
		     (mediapipe.tasks.python vision))
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
	(setf BaseOptions mp.tasks.BaseOptions
	      ImageSegmenter mp.tasks.vision.ImageSegmenter
	      ImageSegmenterOptions mp.tasks.vision.ImageSegmenterOptions
	      VisionRunningMode mp.tasks.vision.RunningMode)

	(def print_result (result output_image timestamp_ms)
	  (declare (type "List[Image]" result)
		   (type Image output_image)
		   (type int timestamp_ms))
	  (print (dot (string "segmented mask size: {}")
		      (format (len result)))))

	(setf options (ImageSegmenterOptions
		       :base_options (BaseOptions
				      :model_asset_path (string "selfie_multiclass_256x256.tflite")
				      :running_mode VisionRunningMode.VIDEO
				      :output_category_mask True)))
	(with (as (ImageSegmenter.create_from_options options)
		  segmenter)
	      (setf mp_image (mp.Image
			      :image_format mp.ImageFormat.SRGB
			      :data numpy_frame_from_opencv))
	      (segmenter.segment_async
	       mp_image
	       frame_timestamp_ms)))

       
       ))))

