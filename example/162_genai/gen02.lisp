(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p02_top"
						 "example/162_genai/"))
 `(do0
   
   (comments "export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_top.py")
   
   
   
   (imports (sys))
    
   (imports-from
    (sqlite_minutils *)
    (loguru logger)
    )

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


   (imports-from (p02_impl GenerationConfig GenAIJob))

   (setf cfg (GenerationConfig
	      :prompt_text (string "make a summary of the imminent government shutdown in the US. show historical parallels.")
	      :model (string "gemini-flash-latest")
	      :output_yaml_path (string "out.yaml")
	      :use_search True
	      :think_budget -1
	      :include_thoughts True))
   (setf job (GenAIJob cfg))
   (setf result (job.run))
   (logger.info (fstring "thoughts: {result.thoughts}"))
   (logger.info (fstring "answer: {result.answer}"))
   (logger.info (fstring "usage: {result.usage_summary}"))
   
   ))

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p02_impl"
						 "example/162_genai/"))
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


   ,(let ((l `(first_thought_time
	       last_thought_time
	       first_answer_time
	       final_answer_time
	       submit_time)))
      `(do0
	@dataclass
	(class StreamResult ()
	       (setf "thoughts:str" (string "")
		     "answer:str" (string "")
		     "responses:List[Any]" (field :default_factory list)
		     "usage_summary:Dict[str,Any]" (field :default_factory dict))
	       ,@(loop for e in l
		       collect
		       `(setf ,(format nil "~a:Optional[float]" e) None))
	       (def timing_metrics (self)
		 (declare (values "Dict[str,float]"))
		 (unless (and ,@(loop for e in l collect `(dot self ,e)))
		   (return "{}"))
		 (return (dictionary
			  :prompt_parsing_time (- self.first_thought_time
						  self.submit_time)
			  :thinking_time (- self.last_thought_time
					    self.first_thought_time)
			  :answer_time (- self.final_answer_time
					  self.last_thought_time)))))))

   (class UsageAggregator ()
	  ,@(loop for (name loop-list) in `((_first responses)
					   (_last (reversed responses)))
	     collect
	     `(do0 @staticmethod
		  (def ,name (responses extractor)
		    (declare (type "Callable[[Any],Any]" extractor))
		    (for (r ,loop-list)
			 (try
			  (setf v (extractor r))
			  (Exception
			   (setf v None)))
			 (unless (is v None)
			   (return v)))
		    (return None))))
	  @classmethod
	  (def summarize (cls result)
	    (declare (type StreamResult result)
		     (values "Dict[str,Any]" ))
	    (setf responses result.responses
		  last_with_usage None)
	    (for (r (reversed responses))
		 (unless (is (getattr r (string "usage_metadata") None)
			     None)
		   (setf last_with_usage r)
		   break))
	    (setf "summary: Dict[str,Any]" "{}")
	    (when last_with_usage
	      (setf um last_with_usage.usage_metadata
		    )
	      ,@(loop for e in `(candidates_token_count
				 prompt_token_count
				 thoughts_token_count)
		      collect
		      `(setf (aref summary (string ,e))
			     (getattr um (string ,e) None))))
	    ,@(loop for e in `(response_id
			       model_version)
		      collect
		      `(setf (aref summary (string ,e))
			     (cls._first responses (lambda (r) (getattr r (string ,e) None)))))

	    (setf totals (list
			  (for-generator (r responses)
					 (getattr
					  (getattr r (string "usage_metadata")
						   None)
					  (string "total_token_count")
					  None))))
	    (setf numeric_totals (list (? (isinstance tot (tuple int float))
					  (for-generator (tot totals)
							 tot))))
	    (setf (aref summary (string "total_token_count"))
		  (? numeric_totals (max numeric_totals) None))
	    (setf (aref summary (string "finish_reason"))
		  (cls._last responses
			     (lambda (r)
			       (or (getattr r (string "finish_reason")
					    None)
				   (? (getattr r (string "candidates") None)
				    (getattr (aref (getattr r (string "candidates") (list None))
						   0)
					     (string "finish_reason") None)
				    None)))))
	    (comments "merge timing metrics")
	    (summary.update (result.timing_metrics))
	    (return summary)))

   ;; Insert PricingEstimator class (cost estimator)
   (class PricingEstimator ()
	  (comments "Estimates API costs based on token usage and model version.")
	  (setf self.PRICING "{}")
	  (setf (aref self.PRICING (string "gemini-2.5-pro"))
		 (dictionary
		  :input_low 1.25
		  :input_high 2.5
		  :output_low 10.0
		  :output_high 15.0
		  :threshold 200000)
		(aref self.PRICING (string "gemini-2.5-flash"))
		 (dictionary
		  :input_low 0.30
		  :input_high 0.30
		  :output_low 2.50
		  :output_high 2.50
		  :threshold 200000)
		(aref self.PRICING (string "gemini-2.5-flash-lite"))
		 (dictionary
		  :input_low 0.10
		  :input_high 0.10
		  :output_low 0.40
		  :output_high 0.40
		  :threshold 200000)
		(aref self.PRICING (string "gemini-2.0-flash"))
		 (dictionary
		  :input_low 0.30
		  :input_high 0.30
		  :output_low 2.50
		  :output_high 2.50
		  :threshold 200000))
	  (setf self.GROUNDING_PRICING
		(dictionary
		 :google_search 35.0
		 :web_grounding_enterprise 45.0
		 :google_maps 25.0
		 :grounding_with_data 2.5))
	  (setf self.FREE_TIER_LIMITS
		(dict
		 ((string "gemini-2.5-pro") (dictionary :google_search 10000 :google_maps 10000))
		 ((string "gemini-2.5-flash") (dictionary :google_search 1500 :google_maps 1500))
		 ((string "gemini-2.5-flash-lite") (dictionary :google_search 1500 :google_maps 1500))
		 ((string "gemini-2.0-flash") (dictionary :google_search 1500 :google_maps 1500))))
	  @classmethod
	  (def _normalize_model_name (cls model_version)
	    (declare (values "Optional[str]"))
	    (unless model_version
	      (return None))
	    (setf m (dot model_version lower))
	    (if (m.contains  (string "2.5-pro"))
		(return (string "gemini-2.5-pro")))
	    (if (m.contains  (string "2.5-flash-lite"))
		(return (string "gemini-2.5-flash-lite")))
	    (if (m.contains (string "2.5-flash"))
		(return (string "gemini-2.5-flash")))
	    (if (m.contains (string "2.0-flash"))
		(return (string "gemini-2.0-flash")))
	    (return None))
	  @class3method
	  (def estimate_cost (cls model_version prompt_tokens thought_tokens output_tokens &key (grounding_used False) (grounding_type (string "google_search")))
	    (declare (values "Dict[str,Any]"))
	    (setf model_name (cls._normalize_model_name model_version))
	    (unless (and model_name (getattr cls PRICING None))
	      (return (dictionary
		       :error (fstring "Unknown model: {model_version}")
		       :model_detected model_name
		       :total_cost_usd 0.0)))
	    (setf pricing (getattr cls PRICING model_name)
		  threshold (getattr pricing (string "threshold") None))
	    (setf use_high_tier (> prompt_tokens threshold))
	    (setf input_rate (? use_high_tier (getattr pricing (string "input_high")) (getattr pricing (string "input_low"))))
	    (setf output_rate (? use_high_tier (getattr pricing (string "output_high")) (getattr pricing (string "output_low"))))
	    (setf input_cost (* (/ prompt_tokens 1000000.0) input_rate)
		  thought_cost (* (/ thought_tokens 1000000.0) output_rate)
		  output_cost (* (/ output_tokens 1000000.0) output_rate))
	    (setf total_token_cost (+ input_cost thought_cost output_cost))
	    (setf grounding_cost 0.0
		  grounding_info "{}")
	    (when grounding_used
	      (setf grounding_rate (getattr cls.GROUNDING_PRICING grounding_type 35.0)
		    grounding_cost (/ grounding_rate 1000.0)
		    grounding_info (dictionary
				   :grounding_type grounding_type
				   :grounding_prompts 1
				   :grounding_cost_usd (round grounding_cost 6)
				   :grounding_rate_per_1k grounding_rate
				   :note (string "Free tier limits apply (not calculated here)"))))
	    (setf total_cost (+ total_token_cost grounding_cost))
	    (setf result (dictionary
			  :model_version model_version
			  :model_detected model_name
			  :pricing_tier (? use_high_tier (string "high") (string "low"))
			  :input_tokens prompt_tokens
			  :thought_tokens thought_tokens
			  :output_tokens output_tokens
			  :total_output_tokens (+ thought_tokens output_tokens)
			  :input_cost_usd (round input_cost 6)
			  :thought_cost_usd (round thought_cost 6)
			  :output_cost_usd (round output_cost 6)
			  :total_token_cost_usd (round total_token_cost 6)
			  :total_cost_usd (round total_cost 6)
			  :rates_per_1m (dictionary :input input_rate :output output_rate)))
	    (unless (== grounding_info "{}")
	      (setf (aref result (string "grounding")) grounding_info))
	    (return result)))
   (class GenAIJob ()
	  (def __init__ (self config ;&key (logger_configured True)
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
	    (setf generate_content_config (types.GenerateContentConfig
					   :thinking_config (types.ThinkingConfig :thinkingBudget self.config.think_budget
										  :include_thoughts self.config.include_thoughts 
							     )
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
	    (for (chunk (self.client.models.generate_content_stream **req))
		 (result.responses.append chunk)
		 (try (setf parts (dot chunk (aref candidates 0)
				       content parts))
		      (Exception
		       continue))
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
			     (incf result.answer part.text))))))
	    
	    (logger.debug (fstring "Thoughts: {result.thoughts}"))
	    (logger.debug (fstring "Answer: {result.answer}"))
	   
	    (setf result.usage_summary (UsageAggregator.summarize result))
	     (setf u result.usage_summary)
	    (setf e (PricingEstimator))
	    (setf price (e.estimate_cost :model_version u.model_version
					:prompt_tokens u.input_tokens
					:thought_tokens u.thought_tokens
					:output_tokens u.output_tokens
					:grounding_used True
					))
	    (logger.debug (fstring "Price: {price}2"))
	    (logger.debug (fstring "Usage: {result.usage_summary}")) 
	    (self._persist_yaml result)
	    (return result))
	  (def _persist_yaml (self result)
	    (declare (type StreamResult result))
	    (setf path self.config.output_yaml_path)
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
