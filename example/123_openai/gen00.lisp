(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "123_openai")
  (defparameter *idx* "00")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    cmd
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    rest)))
                   (format  (- (time.time) start_time)
                            ,@rest)))))

  
  
  (let* ((notebook-name "use_openai")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       (do0
	,(format nil "#|default_exp p~a_~a" *idx* notebook-name))
       (comments "python -m venv ~/bardapi_venv")
       (comments "python -m venv ~/bardapi_env; . ~/bardapi_env/bin/activate; pip install openai toml"
		 ""
		 "deactivate")
       (comments "https://platform.openai.com/docs/api-reference/authentication"
		 "https://github.com/openai/openai-python")
       (comments "env.toml contains this:"
		 "[keys]"
		 "key_token = \" ... \" ")
       (do0
	  
	(imports (	os
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
					;(np jax.numpy)
					;(mpf mplfinance)
					;argparse
			toml
			)))
       (imports-from (openai OpenAI))
	  
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
	  (comments "either set cookies here or read them from env.toml file")
	  (setf config_path (string "env.toml"))
	  (if (os.path.exists config_path)
	      (do0
		   (with (as (open config_path
				   (string "r"))
			     f)
			 (setf data (toml.load f)))
		   (setf keys (aref data (string "keys")))
		  #+nil ,(lprint :vars `(keys))
		   )
	      (do0
	       (print (string "Warning: No env.toml file found."))
	       ))
	  (setf client (OpenAI :api_key (aref keys (string "key_token"))))
	  (setf chat_completion (client.chat.completions.create
				 :messages (list (dictionary
						  :role (string "user")
						  :content (string "Say this is a test")))
				 :model (string "gpt-3.5-turbo")))

	 ;; openai.RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}

	  #+nil (setf bard (BardCookies :cookie_dict cookies))

	  #+nil (with (as (open (string "transcript")
			  (string "r"))
		    f)
		(setf text (f.read)))
	  )))))

