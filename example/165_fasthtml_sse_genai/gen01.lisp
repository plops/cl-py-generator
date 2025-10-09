(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)
;; https://htmx.org/extensions/sse/                                HTMX with server side events
;; https://www.fastht.ml/docs/ref/response_types.html#eventstream  FastHTML with server side events
(progn

  (defparameter *source* "example/165_fasthtml_sse_genai/")
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p02_impl"
						   *source*))
   `(do0
     (imports-from (__future__ annotations))
     (imports (os time #+yaml yaml
		       asyncio))
     
     (imports-from
      (dataclasses dataclass field asdict)
      (typing List
					;Callable
	      Any
					;Optional
	      Dict)
      
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
	    (def __init__ (self config)
	      (declare (type GenerationConfig config))
	      (setf self.config config)
	      
	      (setf self.client (genai.Client :api_key (os.environ.get config.api_key_env))))
	    (def _build_request (self)
	      (declare (values "Dict[str,Any]"))
	      (setf tools (? self.config.use_search
			     (list (types.Tool :googleSearch (types.GoogleSearch)))
			     (list)))
	      (setf safety (list
			    ,@(loop for e in `(HARASSMENT HATE_SPEECH SEXUALLY_EXPLICIT DANGEROUS_CONTENT)
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
					     :tools tools)
		    contents (list (types.Content :role (string "user")
						  :parts (list (types.Part.from_text :text self.config.prompt_text)))))
	      (logger.debug (fstring "_build_request {self.config.prompt_text}"))
	      (return (dictionary :model self.config.model
				  :contents contents
				  :config generate_content_config)))
	    (space async
		   (def run (self)
		     (declare (values StreamResult))
		     (setf req (self._build_request)
			   result (StreamResult))
		     (logger.debug (string "Starting streaming generation"))
		     (setf error_in_parts False)

		     (try
		      (space async
			     (for (chunk (self.client.models.generate_content_stream **req))
				  (result.responses.append chunk)
				  (logger.debug (string "received chunk"))
				  (try (setf parts (dot chunk (aref candidates 0)
							content parts))
				       (Exception
					continue))
				  (try
				   (for (part parts)
					(when (getattr part (string "text") None)
					  (if (getattr part (string "thought") False)
					      (do0
					       (incf result.thought part.text)
					       (yield (dictionary :type (string "thought")
								  :text part.text)))
					      (do0
					       (incf result.answer part.text)
					       (yield (dictionary :type (string "answer")
								  :text part.text))))))
				   (Exception
				    (setf error_in_parts True)
				    pass))))
		      ("Exception as e"
		       (logger.error (fstring "genai {e}"))
		       (yield (dictionary :type (string "error")
					  :message (str e)))))
		     #+yaml
		     (self._persist_yaml result error_in_parts)

		     (logger.debug (fstring "Thoughts: {result.thoughts}"))
		     (logger.debug (fstring "Answer: {result.answer}"))
		     
		     (yield (dictionary :type (string "complete")
					:thought result.thought
					:answer result.answer))))
	    #+nil
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
		       ))
	      ))
     
     
     (setf __all__ (list ,@(loop for e in `(GenerationConfig
					    StreamResult
					    GenAIJob
					    )
				 collect
				 `(string ,e))))
     ))

  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p01_top"
						   *source*))
   `(do0
     (comments "export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_top.py")

     (imports (				;random time
	       asyncio))
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
       )

      (logger.info (string "Logger configured")))
     
     (comments "import after logger exists")
     (imports-from (p02_impl GenerationConfig GenAIJob))

     #+yaml
     (do0 
      (comments "UTC timestamp for output file")
      (setf timestamp (dot datetime datetime (now datetime.UTC)
			   (strftime (string "%Y%m%d_%H_%M_%S"))))
      (setf yaml_filename (fstring "out_{timestamp}.yaml")))

     

     #+nil
     (do0
      (setf job (GenAIJob cfg))
      (setf result (job.run))
      (logger.info (fstring "thoughts: {result.thoughts}"))
      (logger.info (fstring "answer: {result.answer}")))
     
     
     


     (setf hdrs (tuple (Script :src (string "https://unpkg.com/htmx-ext-sse@2.2.3/sse.js")))
	   (ntuple app rt) (fast_app :hdrs hdrs :live True))

     (do0
      @rt
      (def index ()
	(return (ntuple
		 (Titled (string "SSE AI Responder")
			 (P (string "See the response to the prompt"))
			 (Div :hx_ext (string "sse")
			      :sse_connect (string "/response-stream")
			      :hx_swap (string "beforeend show:bottom")
			      :sse_swap (string "message")))))))

     

     (@rt (string "/response-stream"))
     (space async (def get ()
		    (setf config (GenerationConfig
				  :prompt_text (rstring3
						"Make a list of european companies like Bosch, Siemens, group by topic, innovation and moat"
						)
				  :model (string "gemini-flash-latest")
				  #+yaml :output_yaml_path #+yaml yaml_filename
				  :use_search False	  ; True
				  :think_budget 0	  ;-1
				  :include_thoughts False ; True
				  ))
		    (setf job (GenAIJob config))
		    (space async
			   (for (msg (job.run))
				(cond ((== (aref msg (string "type"))
					   (string "thought"))
				       (yield (sse_message (Div (fstring "Thought: {msg['text']}")))))
				      ((== (aref msg (string "type"))
					   (string "answer"))
				       (yield (sse_message (Div (fstring "Answer: {msg['text']}")))))
				      ((== (aref msg (string "type"))
					   (string "complete"))
				       (yield (sse_message (Div (fstring "Final Answer: {msg['answer']}"))))
				       break)
				      ((== (aref msg (string "type"))
					   (string "error"))
				       (yield (sse_message (Div (fstring "Error: {msg['message']}"))))
				       break))))))
     (serve)
     )))


