(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_gen"
						 "example/162_genai/"))
 `(do0
   #+nil
   (do0
    (comments "export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_gen.py")
    (imports (matplotlib))
    (matplotlib.use (string "qtagg"	; "webagg"
			    ))
    (imports ((plt matplotlib.pyplot)))
    (plt.ion)
    )

   (imports ((np numpy)
	     (pd pandas)
	     sys
	     os
	     yaml
	     time))
    
   (imports-from (sqlite_minutils *)
		 (loguru logger)
		 (google genai)
		 (google.genai types))

    (do0
     (logger.remove)
     (logger.add
      sys.stdout
      :format (string "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>")
      :colorize True
      :level (string "DEBUG")
      ;:utc True
      ))

    (logger.info (string "Logger configured"))

   (do0
    
    (setf client (genai.Client :api_key (os.environ.get (string "GEMINI_API_KEY"))
					;:http_options (types.HttpOptions :api_version (string "v1alpha"))
			       ))
    (setf model (string "gemini-flash-latest"))
    (setf contents (list (types.Content
			  :role (string "user")
			  :parts (list (types.Part.from_text :text (rstring3 "make a summary about the most recent news about any vaccine developments against herpes viruses (mrna based or any other).")))
			  )))
    (setf tools (list (types.Tool :googleSearch (types.GoogleSearch))))

    (setf think_max_budget_flash 24576
	  think_auto_budget -1
	  think_off 0)
    (setf generate_content_config (types.GenerateContentConfig
				   :thinking_config (types.ThinkingConfig
						     :thinkingBudget think_auto_budget
						     :include_thoughts True
						     )
				   :tools tools
				   :response_mime_type (string
								    "text/plain"
						)
				   
				   ))
    (setf 
     thoughts (string "")
     answer (string "")
     responses (list)
     t_submit (time.monotonic)
     first_thought_time None
     last_thought_time None
     first_answer_time None
     final_answer_time None)
    
    (for (chunk (client.models.generate_content_stream
		 :model model
		 :contents contents
		 :config generate_content_config))
	 (for (part (dot chunk (aref candidates 0) content parts))
	      (responses.append chunk)
	      (print chunk)
	      (cond ((not part.text)
		     continue)
		    (part.thought
		     (do0
		      (setf now (time.monotonic))
		      (when (is first_thought_time None)
			(logger.info (string "first thought"))
			(setf first_thought_time now))
		      (setf last_thought_time now))
		     (print part.text)
		     (incf thoughts part.text))
		    (t
		     (do0
		      (setf now (time.monotonic))
		      (when (is first_answer_time None)
			(logger.info (string "first answer"))
			(setf first_answer_time now))
		      (setf final_answer_time now))
		     (print part.text)
		     (incf answer part.text))))
	 )
    (do0
     (comments "persist raw responses")
     (with (as (open (string "out.yaml")
		     (string "w")
		     :encoding (string "utf-8"))
	       f)
	   (yaml.dump responses f :allow_unicode True :indent 2)))

    (do0
     (comments "helper to find the first non-null value across all responses using a provided extractor")
     (def find_first (responses extractor)
       (for (r responses)
	    (try
	     (setf v (extractor r))
	     (Exception
	      (setf v None)))
	    (unless (is v None)
	      (return v)))
       (return None))

     (def find_last (responses extractor)
       (for (r (reversed responses))
	    (try
	     (setf v (extractor r))
	     (Exception
	      (setf v None)))
	    (unless (is v None)
	      (return v)))
       (return None))
     (do0 (comments "find the last response that contains usage metadata (for the aggregated token counts)")
	  (setf last_with_usage None
		)
	  (for (resp (reversed responses))
	       (when (is (getattr resp (string "usage_metadata") None)
			 "not None")
		 (setf last_with_usage resp)
		 break)))
     (when (is last_with_usage
	       "not None")
       (setf um last_with_usage.usage_metadata)
       (setf d "{}")
       ,@(loop for e  in `(candidates_token_count
			   prompt_token_count
			   thoughts_token_count
			   )
	       collect
	       `(setf (aref d (string ,e))
		      (getattr um (string ,e) None)))
       ,@(loop for e in `(
			  response_id
			  model_version
			  
			  )
	       collect
	       `(setf (aref d (string ,e))
		      (find_first responses (lambda (r)
						    (getattr r (string ,e)
							     None))))
	       )


       (setf totals (list (for-generator (r responses)
					 (getattr (getattr r (string "usage_metadata")
							   None)
						  (string "total_token_count")
						  None)))
	     valid_totals (list (for-generator (tot totals)
					       (? (isinstance tot (tuple int float))
						  tot
						  None)))
	     (aref d (string "total_token_count")) (? valid_totals
						      (max valid_totals)
						      None))
       (setf (aref d (string "finish_reason"))
	     (find_last responses
			 (lambda (r)
			   (or (getattr r (string "finish_reason")
					None)
			       (? (getattr r (string "candidates") None)
				  (getattr (aref (getattr r (string "candidates") (list None)) 0
						 )
					   (string "finish_reason")
					   None)
				  None)))))

       (setf (aref d (string "sdk_date"))
	     (find_first responses (lambda (r)
				     (and
				      (getattr r (string "sdk_http_response") None)
				      (dot (getattr r.sdk_http_response (string "headers") "{}")
					   (get (string "date")))))))
       (setf (aref d (string "prompt_parsing_time")) (- first_thought_time t_submit)
	     (aref d (string "thinking_time")) (- last_thought_time
						  first_thought_time)
	     (aref d (string "answer_time")) (- final_answer_time last_thought_time))))
    
    (do0
     (logger.info (fstring "thoughts: {thoughts}"))
     (logger.info (fstring "answer: {answer}"))
     (logger.info (fstring "{d}"))
     )
    )
   ))


