(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)


(defparameter *source* "example/165_fasthtml_sse_genai")
(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p02_impl"
						 *source*))
 `(do0
   (imports-from (__future__ annotations))
   (imports (os time yaml))
   
   (imports-from
    (dataclasses dataclass field asdict)
    (typing List Callable Any Optional Dict)
    
    (loguru logger)
    (google genai)
    (google.genai types))

   (do0
    @dataclass
    (class GenerationConfig ()
	   "prompt_text:str"
	   (setf "model:str" (string "gemini-flash-latest")
		 "output_yaml_path:str" (string "out.yaml")
		 "use_search:bool" True
		 "think_budget:int" -1
		 "include_thoughts:bool" True
		 "api_key_env:str" (string "GEMINI_API_KEY"))
	   ))


   (do0
    @dataclass
    (class StreamResult ()
	   (setf "thoughts:str" (string "")
		 "answer:str" (string "")
		 "responses:List[Any]" (field :default_factory list)
		 )))
   
   (class GenAIJob ()
	  (def __init__ (self config	;&key (logger_configured True)
			      )
	    (declare (type GenerationConfig config)
					;(type bool logger_configured)
		     )
	    (setf self.config config)
	    #+nil (unless logger_configured
		    (logger.remove)
		    (logger.add ))
	    (setf self.client (genai.Client :api_key (os.environ.get config.api_key_env))))
	  (def _build_request (self)
	    (declare (values "Dict[str,Any]"))
	    (setf tools (? self.config.use_search
			   (list (types.Tool :googleSearch (types.GoogleSearch)))
			   (list)))
	    (setf safety (list
			  ,@(loop for e in `(HARASSMENT HATE_SPEECH SEXUALLY_EXPLICIT DANGEROUS_COUNTENT)
				  collect
				  `(types.SafetySetting
				    :category (string ,(format nil "HARM_CATEGORY_~a" e))
				    :threshold (string "BLOCK_NONE")))))
	    (setf generate_content_config (types.GenerateContentConfig
					   :thinking_config (types.ThinkingConfig
							     :thinkingBudget self.config.think_budget
							     :include_thoughts self.config.include_thoughts 
							     )
					   :safety_settings safety
					   :tools tools
					;:response_mime_type (string "text/plain")
					   )
		  contents (list (types.Content :role (string "user")
						:parts (list (types.Part.from_text :text self.config.prompt_text)))))
	    (return (dictionary :model self.config.model
				:contents contents
				:config generate_content_config)))
	  (def run (self)
	    (declare (values StreamResult))
	    (setf req (self._build_request)
		  result (StreamResult :submit_time (time.monotonic)))
	    (logger.debug (string "Starting streaming generation"))
	    (setf error_in_parts False)
	    (for (chunk (self.client.models.generate_content_stream **req))
		 (result.responses.append chunk)
		 (try (setf parts (dot chunk (aref candidates 0)
				       content parts))
		      (Exception
		       continue))
		 (try
		  (for (part parts)
		       (when (getattr part (string "text") None)
			 (if (getattr part (string "thought") False)
			     (do0 (setf now (time.monotonic))
				  (when (is result.first_thought_time
					    None)
				    (logger.debug (string "First thought received"))
				    (setf result.first_thought_time now))
				  (setf result.last_thought_time now)
				  (incf result.thoughts part.text))
			     (do0
			      (setf now (time.monotonic))
			      (when (is result.first_answer_time None)
				(logger.debug (string "First answer chunk received"))
				(setf result.first_answer_time now))
			      (setf result.final_answer_time now)
			      (incf result.answer part.text)))))
		  (Exception
		   (setf error_in_parts True)
		   pass)))
	    (self._persist_yaml result error_in_parts)
	    (logger.debug (fstring "Thoughts: {result.thoughts}"))
	    (logger.debug (fstring "Answer: {result.answer}"))
	    
	    (setf result.usage_summary (UsageAggregator.summarize result))
	    (setf u (or result.usage_summary
			"{}"))
	    (logger.debug (fstring "Usage: {result.usage_summary}")) 
	    (setf price (PricingEstimator.estimate_cost
			 :model_version (u.get (string "model_version"))
			 :prompt_tokens (u.get (string "prompt_token_count"))
			 :thought_tokens (u.get (string "thoughts_token_count"))
			 :output_tokens (u.get (string "candidates_token_count"))
			 :grounding_used self.config.use_search
			 ))
	    (logger.debug (fstring "Price: {price}"))
	    (setf (aref result.usage_summary (string "price"))
		  price)
	    
	    (return result))
	  (def _persist_yaml (self result error_in_parts)
	    (declare (type StreamResult result))
	    (setf path self.config.output_yaml_path)
	    (when error_in_parts
	      (setf path (fstring "error_{path}")))
	    (try
	     (do0
	      (with (as (open path (string "w")
			      :encoding (string "utf-8"))
			f)
		    (yaml.dump result.responses
			       f
			       :allow_unicode True
			       :indent 2))
	      (logger.info (fstring "Wrote raw responses to {path}")))
	     ("Exception as e"
	      (logger.error (fstring "Failed to write YAML: {e}")))))
	  (def to_dict (self result)
	    (declare (type StreamResult result)
		     (values "Dict[str,Any]"))
	    (return (dictionary
		     :config (asdict self.config)
		     :thoughts result.thoughts
		     :answer result.answer
		     :usage result.usage_summary))
	    ))
   
   
   (setf __all__ (list ,@(loop for e in `(GenerationConfig
					  StreamResult
					  GenAIJob
					  UsageAggregator
					  PricingEstimator)
			       collect
			       `(string ,e))))
   ))

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_top"
						 *source*))
 `(do0
   (imports (random time asyncio))
   (imports-from (loguru logger)
		 (fasthtml.common *))
          
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


   (setf hdrs (tuple (Script :src (string "https://unpkg.com/htmx-ext-sse@2.2.3/sse.js")))
	 (ntuple app rt) (fast_app :hdrs hdrs))

   (do0
    @rt
    (def index ()
      (return (ntuple
	       (Titled (string "SSE Random Number Generator")
		       (P (string "Generate pairs of random numbers, as the list grows scroll downwards."))
		       (Div :hx_ext (string "sse")
			    :sse_connect (string "/number-stream")
			    :hx_swap (string "beforeend show:bottom")
			    :sse_swap (string "message")))))))

   (setf shutdown_event (signal_shutdown))
   (space async (def number_generator ()
		  (while (not (shutdown_event.is_set))
			 (setf data (Div (Article (random.randint 1 100))
					 (Article (random.randint 1 100))))
			 (yield (sse_message data))
			 (await (asyncio.sleep 1)))))

   (@rt (string "/number-stream"))
   (space async (def get ()
		  (return (EventStream (number_generator)))))
   (serve)
   ))


