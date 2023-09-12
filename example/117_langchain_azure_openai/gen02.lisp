(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "117_langchain_azure_openai")
  (defparameter *idx* "02")
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

  
  
  (let* ((notebook-name "use_qdrant")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
     (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       (do0
	,(format nil "#|default_exp p~a_~a" *idx* notebook-name))
       
       (comments "python -m venv ~/llm_env; . ~/llm_env/bin/activate; source ~/llm_environment.sh;"
		 "pip install qdrant-client langchain[llms] openai sentence-transformers"
		 "deactivate")
       (do0
	  
	(imports (	os
					;sys
			time
					;docopt
					;pathlib
			(np numpy)
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
					langchain
					qdrant_client
					;langchain.chat_models
					; langchain.schema
					; langchain.llms
					openai
			))
	(imports-from (langchain.vectorstores Qdrant)
		      (langchain.embeddings HuggingFaceEmbeddings)
		      (langchain.chat_models AzureChatOpenAI)
		      (langchain.chains RetrievalQA)
		      (qdrant_client QdrantClient)
		     ))
	  
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

	 (setf langchain.debug True)
	 
	 (setf chatgpt_deployment_name (string "gpt-35")
	       chatgpt_model_name (string "gpt-35-turbo")
	       openai.api_type (string "azure")
	       openai.api_key (os.getenv (string "OPENAI_API_KEY"))
	       openai.api_base (os.getenv (string "OPENAI_API_BASE"))
	       openai.api_version (os.getenv (string "OPENAI_API_VERSION")))


	 (setf llm (AzureChatOpenAI :temperature 0
				    :model_name chatgpt_model_name
				    :deployment_name chatgpt_deployment_name)
	       embeddings (HuggingFaceEmbeddings
			   :model_name (string "sentence-transformers/msmarco-MiniLM-L-6-v3")))

	 (setf 
	       client (QdrantClient :host (string "localhost")
				    :port 6333
				    :prefer_grpc False))

	 
	 
	 (setf COLLECTION_NAME (string "aiw")
	       )

	 (setf qdrant (Qdrant :client client
			      :collection_name COLLECTION_NAME
			      :embeddings embeddings
			      :metadata_payload_key (string "payload")))

	 (setf retriever (qdrant.as_retriever))
	 (setf qa (RetrievalQA.from_chain_type :llm llm
					       :chain_type (string "stuff")
					       :retriever retriever))
	 (setf question (string "What is the cause of long covid?"))
	 (setf answer (qa.run question))
	 (print answer)))))

;; Alice in Wonderland
;; How many people are killed by the queen?
;; ValueError: Azure has not provided the response due to a content filter being triggered

;; Twiv Transcripts
;; What is the cause of long covid?
;; The cause of long COVID, also known as post-COVID syndrome, is not fully understood. However, it is believed to be a result of the damage caused by the initial COVID-19 infection. Long COVID can affect various organs and systems in the body, including the lungs, heart, and nervous system. It is important to note that more research is needed to fully understand the underlying mechanisms of long COVID.

