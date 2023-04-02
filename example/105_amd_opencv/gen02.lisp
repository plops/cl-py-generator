(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "105_amd_opencv")
  (defparameter *idx* "02")
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

  
  
  (let* ((notebook-name "opencv_mediapipe_segment")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
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
					mediapipe.tasks
					mediapipe.tasks.python
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
	  
	  (setf model_path (string "/home/martin/Downloads/deeplabv3.tflite"))
	  ,@(loop for e in `((base-options mp.tasks )
			     (image-segmenter mp.tasks.vision)
			     (image-segmenter-options mp.tasks.vision)
			     (vision-running-mode mp.tasks.vision running-mode))
		  collect
		  (destructuring-bind (var pre &optional (name var)) e
		   `(setf ,(cl-change-case:pascal-case (format nil "~a" var))
			  (dot ,pre
			       ,(cl-change-case:pascal-case (format nil "~a" name))))))
	  (do0
	   (setf
	    
		 gResult None
		 oldResult None)
	   (def print_result (result output_image timestamp_ms )
	     (declare (type "list[mp.Image]" result)
		      (type mp.Image output_image)
		      (type int timestamp_ms))
	     ,(lprint :msg "result"
		      :vars `((len result))
		      )
	     (do0
	      
	      "global gResult"
	      (setf gResult result)
	      ))
	   )
	  (comments "output can be category .. single uint8 for every pixel"
		    "or confidence_mask .. several float images with range [0,1]")
	  (setf options (ImageSegmenterOptions
			 :base_options
			 (BaseOptions :model_asset_path model_path)
			 :running_mode VisionRunningMode.LIVE_STREAM
			 :output_type ImageSegmenterOptions.OutputType.CATEGORY_MASK
			 :result_callback print_result)))

	 (do0
	  #+cat
	  (setf category_color (np.random.randint 0 255
						  :size (tuple (+ 255 1)
							       3)
						  :dtype np.uint8))
	  (def view (category imgr)
	    #+cat (setf category_image (aref category_color category))
	    #-cat
	    (do0 (setf category_edges (cv.Canny :image category
				   :threshold1 10
				       :threshold2 1))
		 )
	    (setf (aref imgr (!= category_edges 0))
		  (list 0 0 255))
	    #+nil 
	    (do0 (setf alpha .5)
		 (setf merged_image (cv.addWeighted imgr
						    alpha
						    #-cat (np.array (list category_edges
									  category_edges
									  category_edges))
						    #+cat category_image
						    (- 1 alpha)
						    0)))
	    (cv.imshow (string "screen")
		       imgr)
	    ))
	 (do0
	  ,(lprint :vars `((cv.ocl.haveOpenCL)))
	  (setf loop_time (time.time))
	  (setf clahe (cv.createCLAHE :clipLimit 15.0
				      :tileGridSize (tuple 32 18)))
	  (with (as (ImageSegmenter.create_from_options options)
		    segmenter)
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
			      (setf mp_image (mp.Image :image_format mp.ImageFormat.SRGB
						       :data img))
			      (setf timestamp_ms (int (* 1000 (- (time.time)
								 loop_start))))
			      (segmenter.segment_async mp_image
						       timestamp_ms)
			      #+nil
			      (setf imgr (cv.cvtColor img ;cv.COLOR_RGB2BGR
						      cv.COLOR_RGB2GRAY))
			      (setf lab (cv.cvtColor img 
						     cv.COLOR_RGB2LAB)
				    lab_planes (cv.split lab)
				    lclahe
				    (clahe.apply (aref lab_planes 0))
				    lab (cv.merge (list lclahe 
							(aref lab_planes 1)
							(aref lab_planes 2)))
				    imgr (cv.cvtColor lab cv.COLOR_LAB2RGB))
			      (if (is gResult None)
				  (do0
				   (if (is oldResult None)
				       (cv.imshow (string "screen")
						  imgr)
				       (view (dot (aref oldResult 0)
						  (numpy_view))
					     imgr)))
				  (do0
				   (view (dot (aref gResult 0)
					      (numpy_view))
					 imgr)
				   
				   (setf oldResult gResult
					 gResult None)))
			      

			      
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

