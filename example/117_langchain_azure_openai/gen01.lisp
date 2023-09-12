(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "117_langchain_azure_openai")
  (defparameter *idx* "01")
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
       
       (comments "python -m venv ~/llm_env; . ~/llm_env/bin/activate; source ~/llm_environment.sh;"
		 "pip install qdrant-client langchain[llms] openai sentence-transformers"
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
					;langchain
					;langchain.chat_models
					; langchain.schema
			; langchain.llms
			openai
			))
	(imports-from (langchain.vectorstores Qdrant)
		      (langchain.embeddings HuggingFaceEmbeddings)
		      (langchain.text_splitter RecursiveCharacterTextSplitter)
		      (qdrant_client QdrantClient)
		      (qdrant_client.http models)
		      (qdrant_client.http.models CollectionStatus)
		      (qdrant_client.models PointStruct Distance VectorParams)
		      (sentence_transformers SentenceTransformer)
		      (tqdm.notebook tqdm)))
	  
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

	 
	 (setf chatgpt_deployment_name (string "gpt-35")
	       chatgpt_model_name (string "gpt-35-turbo")
	       openai.api_type (string "azure")
	       openai.api_key (os.getenv (string "OPENAI_API_KEY"))
	       openai.api_base (os.getenv (string "OPENAI_API_BASE"))
	       openai.api_version (os.getenv (string "OPENAI_API_VERSION")))

	 (comments "cd ~/src; git clone --depth 1 https://github.com/RGGH/LangChain-Course")
	 
	 (comments "cd ~/Downloads ; wget https://github.com/qdrant/qdrant/releases/download/v1.5.1/qdrant-x86_64-unknown-linux-gnu.tar.gz")
	 (comments "mkdir q; cd q; tar xaf ../q/qdrant*.tar.gz")
	 (comments "cd ~/Downloads/q; ./qdrant")
	 (setf COLLECTION_NAME (string "aiw")
	       TEXTS (list (string "/home/martin/src/LangChain-Course/lc5_indexes/text/aiw.txt"))
	       vectors (list)
	       batch_size 512
	       batch (list))

	 (setf model (SentenceTransformer (string "msmarco-MiniLM-L-6-v3"))
	       client (QdrantClient :host (string "localhost")
				    :port 6333
				    :prefer_grpc false))

	 (def make_collection (client collection_name)
	   (declare (type str collection_name))
	   (client.recreate_collection
	    :collection_name COLLECTION_NAME
	    :vectors_config (models.VectorParams :size 384
						 :distance models.Distance.COSINE)))

	 (def make_chunks (input_text )
	   (declare (type str input_text))
	   (setf text_splitter (RecursiveCharacterTextSplitter :separators (string "\\n")
							       :chunk_size 1000
							       :chunk_overlap 20
							       :length_function len))
	   (with (as (open input_text) f)
		 (setf alice (f.create)))
	   (setf chunks (text_splitter.create_documents (list alice)))
	   (return chunks))

	 (setf texts (make_chunks (aref TEXTS 0)))
	 
	 #+nil
	 (do0
	  (comments "this works")
	  (setf chat (langchain.chat_models.AzureChatOpenAI
		      :deployment_name chatgpt_deployment_name
		      :model_name chatgpt_model_name
		      :temperature 1))
	  (setf user_input (input (string "Ask me a question: ")))

	  (setf messages (list ;(langchain.schema.SystemMessage :contents (string "You are an angry assistant"))
			  (langchain.schema.HumanMessage :content user_input)))

	  (print (dot (chat messages)
		      content)))

	))))

