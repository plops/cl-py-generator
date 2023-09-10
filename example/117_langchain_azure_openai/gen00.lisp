(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "117_langchain_azure_openai")
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

  
  
  (let* ((notebook-name "use_gpt")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       (do0
	,(format nil "#|default_exp p~a_~a" *idx* notebook-name))
       (comments "python -m venv ~/llm_env; . ~/llm_env/bin/activate; pip install langchain"
		 ""
		 "deactivate")
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
			datetime
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
			langchain
			langchain.chat_models
			langchain.schema
			langchain.llms
			openai
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

	 #+nil
	 (do0
	  (setf response (openai.Completion.create :engine (string "gpt-35")
						   :prompt (string "This is a test")
						   :max_tokens 5))
					
	  )
	 (setf chatgpt_deployment_name (string "gpt-35")
	       chatgpt_model_name (string "gpt-35-turbo")
	       openai.api_type (string "azure")
	       openai.api_key (os.getenv (string "OPENAI_API_KEY"))
	       openai.api_base (os.getenv (string "OPENAI_API_BASE"))
	       openai.api_version (os.getenv (string "OPENAI_API_VERSION")))

	 #+nil
	 (do0 (setf response (openai.ChatCompletion.create :engine chatgpt_deployment_name
							   :messages (list (dictionary :role (string "system")
										       :content (string "You are an angry assistent."))
									   (dictionary :role (string "user")
										       :content (string "Who won the world series in 2020.")))))
	      (print response))
	 
	 (do0
	  (setf llm (langchain.llms.AzureOpenAI :deployment_name chatgpt_deployment_name
						:model_name chatgpt_model_name))

	  (llm (string "Tell me a joke")))
	 #+nil
	 
	 (do0
	  (setf chat (langchain.chat_models.ChatOpenAI :temperature 1))
	  (setf user_input (input (string "Ask me a question: ")))

	  (setf messages (list ;(langchain.schema.SystemMessage :contents (string "You are an angry assistant"))
			  (langchain.schema.HumanMessage :content user_input)))

	  (print (dot (chat messages)
		      content)))

	))))

