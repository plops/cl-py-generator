(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "130_torch_optim")
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

  
  
  (let* ((notebook-name "optim")
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
       (comments "python -m venv ~/pytorch_env"
		 ". ~/pytorch_env/bin/activate"
		 "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu11")
       (imports (os
		 time
		 torch
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
       ,(let ((l-model `((:name rgb_to_srgb_matrix :torch (eye 3) :dim (3 3))
			 (:name brightness :torch (tensor 1s0) :dim (1))
			 (:name offset :torch (zeros 3) :dim (3))
			 (:name hue_angle :torch (tensor 0s0) :dim (1))
			 (:name gamma :torch (tensor 2.2s0) :dim (1)))))
	  `(do0
	    (class ImageModel (torch.nn.Module)
		   (def __init__ (self)
		     (dot (super ImageModel self)
			  (__init__))
		     ,@(loop for e in l-model
			     collect
			     (destructuring-bind (&key name torch dim) e
			      `(setf (dot self ,name)
				     (dot torch nn (Parameter (dot torch ,torch)))))
			     )
		     (setf self.rgb_to_yuv_matrix (torch.tensor
						   (list (list  0.299 0.587 0.114)
							 (list -0.14713 -0.28886 0.436)
							 (list  0.615 -.51499 -.10001)))))
		   (def forward (self x)
		     ,@(loop for e in `((torch.matmul x self.rgb_to_srgb_matrix)
					(torch.pow x (/ 1s0 self.gamma))
					(* x self.brightness)
					(+ x self.offset)
					(torch.matmul x rgb_to_yuv_matrix)
					(torch.matmul x (torch.tensor (list (list 1 0 0)
									    (list 0 (torch.cos self.hue_angle) (* -1 (torch.sin self.hue_angle)))
									    (list 0 (torch.sin self.hue_angle) (torch.cos self.hue_angle))
									    ))))
			     collect
			     `(setf x ,e)
			     )
		     #+nil (do0
			    (comments "apply rotation only to U and V")
			    (setf (aref x ":" (slice 1 ""))
				  (torch.matmul (aref x ":" (slice 1 "")))))
		     (return x)))

	    (setf rgb_data (torch.rand 100 3))

	    (setf model (ImageModel))

	    (setf initial_yuv (model rgb_data))
	    (setf target_yuv (+ initial_yuv (* (torch.randn_like initial_yuv) .1)))

	    (def objective (params)
	      (comments "update model parameters from individual lmfit parameters")
	      ,@(loop for e in l-model
			     collect
			     (destructuring-bind (&key name torch dim) e
			       (cond 
				 ((equal dim `(1))
				  `(setf (dot model ,name)
					 (aref params (string ,name))))
				 ((equal dim `(3))
				  `(setf (dot model ,name)
					 (torch.tensor (list (for-generator
							      (i (range 3))
							      (aref params (fstring ,(format nil "~a_{i}" name))))))))
				 ((equal dim `(3 3))
				  `(setf (dot model ,name)
					 (torch.tensor
					  (list (for-generator
						 (i (range 3))
						 (list (for-generator
							(j (range 3))
							(aref params (fstring ,(format nil "~a_{i}{j}" name))))))))))
				 (t (break (string "unhandled condition")))))
			     )
	      (for (param (model.parameters))
		   (setf param.requires_grad True))
	      (setf yuv_predicted (model rgb_data))
	      (setf loss (torch.nn.functional_mse_loss yuv_predicted target_yuv))
	      (loss.backward)
	      (setf grads (curly (for-generator ((ntuple name parm)
						 (model.named_parameters))
						(space "name:"
						       (dot param
							    grad
							    (detach)
							    (numpy))))))
	      (for (param (model.parameters))
		   (setf param.requires_grad False))
	      (return (ntuple loss grads)))
	    (do0
	     (comments "define lmfit parameters")
	     (setf params (lmfit.Parameters))
	     )))
       
       ))))

