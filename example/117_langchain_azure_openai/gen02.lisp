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
	 
	 (setf chatgpt_deployment_name (string "gpt-4")
	       chatgpt_model_name (string "gpt-4")
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
	 (setf question (string "Tell me about passive immunity."))
	 (setf answer (qa.run question))
	 (print answer)))))

;; GPT-3.5 Turbo

;; Alice in Wonderland
;; How many people are killed by the queen?
;; ValueError: Azure has not provided the response due to a content filter being triggered

;; Twiv Transcripts
;; What is the cause of long covid?
;; The cause of long COVID, also known as post-COVID syndrome, is not fully understood. However, it is believed to be a result of the damage caused by the initial COVID-19 infection. Long COVID can affect various organs and systems in the body, including the lungs, heart, and nervous system. It is important to note that more research is needed to fully understand the underlying mechanisms of long COVID.

;; How does a COVID infection typically progress?
;; A COVID infection can progress differently for each individual, but generally, it starts with mild symptoms such as fever, cough, and fatigue. In some cases, the symptoms may worsen, leading to difficulty breathing and pneumonia. Severe cases can result in acute respiratory distress syndrome (ARDS), organ failure, and even death. It's important to note that some individuals may experience long-term effects, such as lung damage and heart damage, even after recovering from the initial infection.

;; What happens in the second week of a COVID infection?
;; I don't have enough information to answer that question.

;; What are treatments for COVID beside Paxlovid?
;; I don't know.

;; How does Paxlovid work?
;; I don't know.

;; Tell me in which contexts they mention migriane.
;; There is no mention of migraines in the given context.

;; What is IgG?
;; I don't know.

;; GPT-4
;; What is IgG?
;; I'm sorry, but the provided context does not contain information about IgG.

;; Tell me about tosyllismab.
;; I'm sorry, but the provided context does not contain any information about tosyllismab.

;;    "System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\nnext thing I guess I'll mention is that this is passive immunity. So when you give someone this\n\n23:56.000 --> 24:02.480\n infusion or you do the subcutaneous injection, then you're not going to see an immune response.\n\n24:02.480 --> 24:06.240\n You're not going to see lymph nodes. You're not going to see soreness at the site.\n\n24:06.240 --> 24:11.120\n About 2% of the time people get a little bit itchy. So that's why we sort of observed them.\n\n24:11.120 --> 24:14.480\n Get a little bit itchy. We give them Benadryl has been an issue to date. So\n\n24:15.600 --> 24:23.040\n making some progress there. The early inflammatory phase. People should go back and listen to\n\n24:23.040 --> 24:29.920\n immune 39, a Tonka truck full of salt. And they have a discussion right at about 57 minutes about\n\n24:29.920 --> 24:38.160\n the cytokine storm. And the whole early inflammatory response is really a sophisticated, complicated\n\n24:38.160 --> 24:45.200\n\nyou know, we only do well visits. We don't let people come in if they're sick and not sure if\n\n04:45.520 --> 04:51.200\n this really applies to me. And so I responded back and said, you think they're well visits.\n\n04:51.200 --> 04:56.320\n But you know, what percent of those people are pre-symptomatic or asymptomatic? And that's\n\n04:56.320 --> 05:00.720\n really the thing. If you get a call the next day and they say, you know what, I wasn't feeling 100%\n\n05:00.720 --> 05:04.880\n when I saw you. And today I just got a test. And you were not wearing your eye protection.\n\n05:05.920 --> 05:11.440\n That's a potential exposure. So, you know, this is really moving into a universal precautions\n\n05:11.440 --> 05:17.840\n approach, at least for the foreseeable future here. You know, for front facing healthcare workers,\n\n05:17.840 --> 05:23.120\n we'll talk about why there is a finite period of time that we're going to need to be practicing\n\n05:23.120 --> 05:29.040\n\noxygen. We still do medications to prevent people against cloth. So a lot of supportive care.\n\n25:31.120 --> 25:36.160\n Secondary infection phase. We're actually, I think I mentioned last time, but we're starting to see\n\n25:36.160 --> 25:42.000\n more issues here with resistant organisms because of sort of overuse of antibiotics early on. So\n\n25:42.000 --> 25:46.880\n let's really be careful sort of a reminder there to all our clinicians. If you see these people in\n\n25:46.880 --> 25:53.120\n the urgent care, you see people in the outpatient setting, what do you do? You might get them hooked\n\n25:53.120 --> 25:57.840\n up with a pulse oximeter. You might monitor them. You want to make that decision. Do they need to\n\n25:57.840 --> 26:03.840\n come in the hospital? You might consider aspirin, things like that. But no antibiotics is a viral\n\n26:03.840 --> 26:08.880\n illness. You want to be really careful. If you're giving more than 10% of your COVID patients\n\n26:08.880 --> 26:14.240\n\nand getting tested. A lot of people with asymptomatic and presymptomatic infections were being picked\n\n07:39.520 --> 07:45.120\n up, diagnosed, sort of pulled out, contact tracing, and then those people were being picked out. And\n\n07:45.120 --> 07:48.800\n I really think, you know, we were, we were saving lives here. We were detecting people\n\n07:48.800 --> 07:53.200\n before they went to those gatherings. We were picking up who else was infected that they were\n\n07:53.200 --> 07:59.440\n interacting with. And so this is really, really taken on and people are realizing the utility.\n\n08:00.000 --> 08:05.680\n You know, two days before symptoms, five days of symptoms, the rapid test gives you a result in\n\n08:05.680 --> 08:11.360\n about 15 minutes has really been tremendous. This is free, right? This is something that we,\n\n08:11.360 --> 08:15.120\n you know, as a society are picking up. This is not something we charged people for.\n\n08:15.760 --> 08:20.160\nHuman: Tell me about tosyllismab."


;; Tell me about passive immunity.
;; Passive immunity refers to the process where a person is given antibodies, rather than their body producing them. This can be done through an infusion or a subcutaneous injection. However, because the antibodies are given directly, you won't see an immune response such as lymph nodes or soreness at the site. About 2% of the time, people might get a little bit itchy from this process.
