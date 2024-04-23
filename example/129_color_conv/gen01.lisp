(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "129_color_conv")
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

  
  
  (let* ((notebook-name "conv")
	 (cli-args `(#+nil (:short "c" :long "chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     #+nil (:short "p" :long "prompt" :type str
		      :default (string "Summarize the following video transcript as a bullet list.")
		      :help "The prompt to be prepended to the output file(s)."))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (imports (os
		 time
		 (np numpy)
		 (cv cv2)
		 (pd pandas)
		 lmfit))
       
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

       #+nil (do0 (setf bgr (cv.Mat (list (list (list 10 120 13)))))
	    (cv.cvtColor bgr cv.COLOR_BGR2YCrCb))

       (def bgr_to_ycbcr_model (bgr coeff_matrix offsets gains gamma)
	 (string3 "Model for BGR to YCbCr color transformation with adjustable parameters.

  Args:
    bgr: A numpy array of shape (3,) representing B, G, R values.
    coeff_matrix: A 3x3 numpy array representing the transformation coefficients.
    offsets: A numpy array of shape (3,) representing the offsets for each channel.
    gains: A numpy array of shape (3,) representing the gains for each channel.
    gamma: The gamma correction value.

  Returns:
    A numpy array of shape (3,) representing the Y, Cb, Cr values.")

	 (setf bgr_gamma (np.power (/ bgr 255s0)
				   (/ 1s0 gamma)))
	 (setf ycbcr (+ (* (np.dot coeff_matrix bgr_gamma) gains )
			offsets))
	 (return ycbcr))

       (def fit_bgr_to_ycbcr (df)
	 (string3 "Fits the BGR to YCbCr model to data using lmfit.

  Args:
    df: A pandas DataFrame with columns 'B', 'G', 'R', 'Y', 'Cb', 'Cr'.

  Returns:
    An lmfit ModelResult object containing the fitted parameters.")
	 (setf model (lmfit.Model bgr_to_ycbcr_model)
	       params (model.make_params
		       :coeff_matrix (np.identity 3)
		       :offsets (np.zeros 3)
		       :gains (np.ones 3)
		       :gamma 2.2
		       )
	       result (model.fit (dot (aref df (list (string "Y")
						  (string "Cb")
						  (string "Cr")))
				      values)
				 params
				 :bgr (dot (aref df (list (string "B")
						  (string "G")
						  (string "R")))
				      values)))
	 (return result))

       (do0
	(setf num_colors 100)
	(setf bgr_colors (np.random.randint 0 256 :size (tuple num_colors 3)))
	(setf res (list))
	(for (bgr bgr_colors)
	     (setf ycbcr (aref (cv.cvtColor (np.uint8 (list (list bgr)))
				       cv.COLOR_BGR2YCrCb)
			       0 0))
	     (res.append (dictionary :B (aref bgr 0)
				     :G (aref bgr 1)
				     :R (aref bgr 2)
				     :Y (aref ycbcr 0)
				     :Cb (aref ycbcr 1)
				     :Cr (aref ycbcr 2)
				     
				     )))
	
	(setf df (pd.DataFrame res))
	(setf result (fit_bgr_to_ycbcr df))
	(print (result.fit_report)))

       
       ))))

