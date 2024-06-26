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
		      :help "The prompt to be prepended to the output file(s).")))
	 		       
	 (l-coef `((:name coeff_matrix :value (1 0 0
						 0 1 0
						 0 0 1) :vary True :dim (3 3) ; :mi 0 :ma 300
						 )
		   (:name offsets :value (0 128 128) :dim (3) :vary False ;:mi -100 :ma 100
			  )
		   #+nil (:name gains :value (1 1 1) :dim (3) :mi 0 :ma 3 :vary False
			  )
		   (:name gamma_bgr :value (1) :dim (1) :mi .1 :ma 3  :vary False
			  )
		   
		   #+nil((:name offsets_y :value (0 0 0) :dim (3) :vary False ;:mi -100 :ma 100
			   )
		    (:name gains_y :value (1 1 1) :dim (3) :vary False ; True :mi .01 :ma 30s0
			   )
		    (:name gamma_y :value (1 1 1) :dim (3) :mi .1 :ma 3  :vary False
			   )))))
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

       #+nil(do0
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
			    (- tz))))))

       #+nil (do0 (setf bgr (cv.Mat (list (list (list 10 120 13)))))
	    (cv.cvtColor bgr cv.COLOR_BGR2YCrCb))
       " "
       (def bgr_to_ycrcb_model (bgr ,@(let ((count 0))
					 (loop for e in l-coef
					       appending
					       (destructuring-bind (&key name value dim (vary 'True) mi ma) e
						 (loop for v in value
						       collect
						       (prog1
							   (intern (string-upcase (format nil "~a~a" name count)))
							 (incf count)))
						 ))))
	 (string3 "Model for BGR to Ycrcb color transformation with adjustable parameters.

  Args:
    bgr: A numpy array of shape (3,) representing B, G, R values.
    coeff_matrix: A 3x3 numpy array representing the transformation coefficients.
    offsets: A numpy array of shape (3,) representing the offsets for each channel.
    gains: A numpy array of shape (3,) representing the gains for each channel.
    gamma: The gamma correction value.

  Returns:
    A numpy array of shape (3,) representing the Y, Cb, Cr values.")

	 (setf params (np.array (list ,@(let ((count 0))
					 (loop for e in l-coef
					       appending
					       (destructuring-bind (&key name value dim (vary 'True) mi ma) e
						 (loop for v in value
						       collect
						       (prog1
							   (intern (string-upcase (format nil "~a~a" name count)))
							 (incf count)))
						 ))))))
	 ,@(let ((count 0))
	     (loop for e in l-coef
		   collect
		   (destructuring-bind (&key name value dim (vary 'True) mi ma) e
		     (let ((inc (length value)))
		       (prog1
			   `(setf ,name (dot (aref params (slice ,count ,(+ count inc)))
					     (reshape (tuple ,@dim))))
			 (incf count inc))))))
	 (setf bgr_gamma (np.power (/ bgr 255s0)
				   (/ 1s0 gamma_bgr)))
	 #+nil (setf ycrcb (+ (* (np.dot bgr_gamma coeff_matrix.T) gains )
			offsets))
	  (setf ycrcb (+ (np.dot  bgr_gamma coeff_matrix.T)
			offsets))
	 #+noil (setf ycrcb
	       (+ offsets_y (* gains_y 255s0
			       (np.power (/ ycrcb 255s0)
					 (/ 1s0 gamma_y)))))
	 
	 
	 (return ycrcb))
       " "
       

       (do0
	" "
	(setf num_colors 1000)
	(setf bgr_colors (np.random.randint 0 256 :size (tuple num_colors 3)))
	(setf res (list))
	(for (bgr bgr_colors)
	     (setf ycrcb (aref (cv.cvtColor (np.uint8 (list (list bgr)))
				       cv.COLOR_BGR2YCrCb)
			       0 0))
	     (res.append (dictionary :B (aref bgr 0)
				     :G (aref bgr 1)
				     :R (aref bgr 2)
				     :Y (aref ycrcb 0)
				     :Cr (aref ycrcb 1)
				     :Cb (aref ycrcb 2)
				     
				     )))
	" "

	(setf df (pd.DataFrame res))

	(do0 
	    ; def fit_bgr_to_ycrcb (df)
	    (string3 "Fits the BGR to Ycrcb model to data using lmfit.

  Args:
    df: A pandas DataFrame with columns 'B', 'G', 'R', 'Y', 'Cb', 'Cr'.

  Returns:
    An lmfit ModelResult object containing the fitted parameters.")
	 (setf model (lmfit.Model bgr_to_ycrcb_model)
	       params (lmfit.Parameters)
	       #+nil
	       (model.make_params
		:coeff_matrix (np.identity 3)
		:offsets (np.zeros 3)
		:gains (np.ones 3)
		:gamma "2.2"
		)
	       )
	 ,@(let ((count 0))
	     (loop for e in l-coef
		  appending
		   (destructuring-bind (&key name value dim (vary 'True) mi ma) e
		     (loop for v in value
			   collect
			   (prog1
			       `(params.add
				 (string ,(format nil "~a~a" name count))
				 :value ,v :vary ,vary :min ,(if mi mi '-np.inf) :max ,(if ma ma 'np.inf))
			     (incf count)))
		     )))
	 (setf result (model.fit (dot (aref df (list (string "Y")
						     (string "Cb")
						     (string "Cr")))
				      values)
				 params
				 :bgr (dot (aref df (list (string "B")
							  (string "G")
							  (string "R")))
					   values)))
	 ;(return result)
	 )

	;(setf result (fit_bgr_to_ycrcb df))
	(print (result.fit_report)))

       
       ))))

