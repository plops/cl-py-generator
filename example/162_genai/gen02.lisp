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
   
   
   
   (imports ((np numpy) (pd pandas) sys))
    
   (imports-from
    (sqlite_minutils *)
    (loguru logger)
    (google genai)
    (google.genai types)
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
		     "responses:list[Any]" (field :default_factory list)
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
			  :answer_time (- final_answer_time
					  last_thought_timeq)))))))

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
	    (result.usage_summary (UsageAggregator.summarize result))
	    (self._persist_yaml result)
	    (return result))
	  (def _persist_yaml (self result)
	    (declare (type StreamResult result))
	    (setf path self.config.output_yaml_path)
	    (try
	     (do0
	      (with (as (open pathname (string "w")
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
					  UsageAggregator)
			       collect
			       `(string ,e))))
   ))


