(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator))

(in-package #:my-py-project)
;; https://htmx.org/extensions/sse/                                HTMX with server side events
;; https://www.fastht.ml/docs/ref/response_types.html#eventstream  FastHTML with server side events

(setf *features* (union *features* '(:log ;; logger
				     )))
(setf *features* (set-difference *features* '(;:log
					      )))

(progn

  (defun lprint (&key msg vars (level "info"))
    `(do0
      #+log (dot logger
		 (,level
		  ,(if vars
		       `(dot (string ,(format nil "~a ~{~a={}~^ ~}"
					      msg
					      (mapcar (lambda (x)
							(emit-py :code x))
						      vars)))
			     (format
			      ,@vars))
		       `(string ,(format nil "~a"
					 msg)))))))
  (let ((helper-classes
	  `(do0
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
		    (setf "thought:str" (string "")
			  "answer:str" (string "")
					;"responses:List[Any]" (field :default_factory list)
			  )))

	    (class GenAIJob ()
		   (def __init__ (self config)
		     (declare (type GenerationConfig config))
		     ,(lprint :msg "GenAIJob.__init__" :level "trace" )
		     (setf self.config config)

		     (setf self.client (genai.Client :api_key (os.environ.get config.api_key_env))))
		   (def _build_request (self)
		     (declare (values "Dict[str,Any]"))
		     ,(lprint :level "trace" :msg "GenAIJob._build_request")

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
		     ,(lprint :level "debug"
			      :msg "_build_request"
			      :vars `(self.config.prompt_text))
		     (return (dictionary :model self.config.model
					 :contents contents
					 :config generate_content_config)))
		   (space async
			  (def run (self)
					;(declare (values StreamResult))
			    (setf req (self._build_request)
				  result (StreamResult))
			    ,(lprint :level "debug" :msg "Starting streaming generation")
			    (setf error_in_parts False)

			    (try
			     (for (chunk (self.client.models.generate_content_stream **req))
				  #+nil (result.responses.append chunk)
				  ,(lprint :level "debug" :msg "received chunk")
				  (try (setf parts (dot chunk (aref candidates 0)
							content parts))
				       ("Exception as e"
					,(lprint :level "debug" :msg "exception when accessing chunk:" :vars `(e))
					continue))
				  (try
				   (for (part parts)
					(when (getattr part (string "text") None)
					  ,(lprint :level "trace" :msg "" :vars `(part))
					  (if (getattr part (string "thought") False)
					      (do0
					       (incf result.thought part.text)
					       (yield (dictionary :type (string "thought")
								  :text part.text)))
					      (do0
					       (incf result.answer part.text)
					       (yield (dictionary :type (string "answer")
								  :text part.text))))))
				   ("Exception as e"
				    (setf error_in_parts True)
				    ,(lprint :level "warning" :msg "genai" :vars `(e))
					;pass
				    )))
			     ("Exception as e"
			      ,(lprint :level "error" :msg "genai" :vars `(e))
			      (yield (dictionary :type (string "error")
						 :message (str e)))
			      return))
			    #+yaml
			    (self._persist_yaml result error_in_parts)

			    ,(lprint :level "debug" :msg "Thought:" :vars `(result.thought))
			    ,(lprint :level "debug" :msg "Answer:" :vars `(result.answer))

			    (yield (dictionary :type (string "complete")
					       :thought result.thought
					       :answer result.answer
					       :error error_in_parts))))
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
		       ,(lprint :msg "Wrote raw responses to" :vars `(path)))
		      ("Exception as e"
		       ,(lprint :level "error" :msg "Failed to write YAML:" :vars `(e)))))
		   #+nil
		   (def to_dict (self result)
		     (declare (type StreamResult result)
			      (values "Dict[str,Any]"))
		     (return (dictionary
			      :config (asdict self.config)
			      :thought result.thought
			      :answer result.answer
			      ))
		     ))



	    )))
    (defparameter *source* "example/165_fasthtml_sse_genai/")
    #+nil
    (write-source
     (asdf:system-relative-pathname 'cl-py-generator
				    (merge-pathnames #P"p02_impl"
						     *source*))
     `(do0
       (do0 (imports-from (__future__ annotations))
	    (imports (os time #+yaml yaml
			      asyncio))

	    (imports-from
	     (dataclasses dataclass field asdict)
	     (typing			;List
					;Callable
	      Any
					;Optional
	      Dict)

	     (loguru logger)
	     (google genai)
	     (google.genai types)))
       ,helper-classes
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
       (imports-from (__future__ annotations))
       (imports (			;random time

		 datetime
		 argparse))
       (imports-from
	(fasthtml.common
	 Script
	 fast_app
	 Titled
	 Form
	 Fieldset
	 Legend
	 Div
	 Label
	 Textarea
	 Button
	 Request
	 #+timer signal_shutdown
	 #+timer Article
	 sse_message

	 EventStream
	 serve

	 ))

       (do0
	(imports (os  sys #+yaml yaml
			  asyncio))

	(imports-from
	 (dataclasses dataclass		;field ;asdict
		      )
	 (typing			;List
					;Callable
	  Any
					;Optional
	  Dict)


	 (loguru logger)
	 (google genai)
	 (google.genai types)
	 (urllib.parse quote_plus)))

       (do0
	(comments "Parse command-line arguments")
	(setf parser (argparse.ArgumentParser :description (string "Run the SSE AI Responder website")))
	(parser.add_argument (string "-v")
			     (string "--verbose")
			     :action (string "count")
			     :default 0
			     :help (string "Increase verbosity: -v for DEBUG, -vv for TRACE"))
	(setf args (parser.parse_args))

	(do0
	 (comments "Determine log level based on verbosity")
	 (cond ((== args.verbose 1)
		(setf log_level (string "DEBUG"))
		)
	       ((>= args.verbose 2)
		(setf log_level (string "TRACE")))
	       (t
		(setf log_level (string "INFO")))))

	)
       (do0
	(logger.remove)
	(logger.add
	 sys.stdout
	 :format (string "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>")
	 :colorize True
	 :level log_level
	 :enqueue True ;; enable logging in async environment without blocking the eventloop
					;:utc True
	 )

	,(lprint :msg "Logger configured"))

       (comments "import after logger exists")

       #+nil
       (imports-from (p02_impl GenerationConfig GenAIJob))
       ,helper-classes

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
	,(lprint :msg "thought:" :vars `(result.thought))
	,(lprint :msg "answer:" :vars `(result.answer)))





       (setf hdrs (tuple (Script :src (string "https://unpkg.com/htmx-ext-sse@2.2.3/sse.js")))
	     (ntuple app rt) (fast_app :hdrs hdrs
					; :live False
				       ))

       #+nil "If the submit button is pressed, the prompt-text shall be submitted as
a GenAIJob. the thought and answer outputs of its job method shall be
appended to corresponding HTML elements via HTMX with server side
events until the final answer is complete or an error has occured"
       (do0
	@rt
	(def index ()
	  (return (ntuple
		   (Titled (string "SSE AI Responder")
			   (Form
			    (Fieldset
			     (Legend (string "Submit a prompt for the AI to respond to"))
			     (Div
			      (Label (string "Enter your prompt here (e.g. Make a list of european companies like Bosch, Siemens, group by topic, innovation and moat.)")
				     :_for (string "prompt_text"))
			      (Textarea
			       :placeholder (string "Enter prompt text here")
			       :style (string "height: 300px; width: 60%;")
			       :id (string "prompt_text")
			       :name (string "prompt_text"))
			      (Button (string "Submit"))))
			    :data_hx_post (string "/process_transcript")
			    :data_hx_swap (string "afterbegin")
			    :data_hx_target (string "#response-list"))
			   #+timer
			   (Div :data_hx_ext (string "sse")
				:data_sse_connect (string "/time-sender")
				:data_hx_swap (string "innerHTML") ; (string "beforeend show:bottom")
				:data_sse_swap (string "message")
				:data_sse_close (string "close"))
			   #+nil
			   (Div :data_hx_ext (string "sse")
				:data_sse_connect (string "/response-stream")
				:data_hx_swap (string "beforeend show:bottom")
				:data_sse_swap (string "message"))
			   (Div :id (string "response-list")))))))


       (do0
	(@app.post (string "/process_transcript"))
	(def process_transcript (prompt_text
				 request)
	  (declare (type str prompt_text)
		   (type Request request))
	  (comments "Return a new SSE Div with the prompt in the connect URL")
	  (setf id_str (dot datetime datetime (now) (timestamp))
		uid (fstring "id-{id_str}"))
	  ,(lprint :level "trace" :msg "POST process_transcript" :vars `(request.client.host prompt_text))
	  (return (Div
					;(Article (fstring "Prompt: {prompt_text}"))
		   (Div (string "Thoughts:")
			(Div :id (fstring "{uid}-thoughts")))
		   (Div (string "Answer:")
			(Div :id (fstring "{uid}-answer")))
		   (Div :id (fstring "{uid}-error"))
		   :data_hx_ext (string "sse")
		   :data_sse_connect (fstring "/response-stream?prompt_text={quote_plus(prompt_text)}&uid={uid}")
					; :data_hx_swap (string "beforeend show:bottom")
		   :data_sse_swap (string "thought,answer,final_answer,error")
		   :data_hx_swap_oob (string "true")
		   :data_hx_target (string "response-list")
		   :data_sse_close (string "close")))))

       #+timer
       (do0
	(setf event (signal_shutdown))
	(space async (def time_generator ()
		       ,(lprint :level "trace" :msg "time_generator init")
		       (setf count 0)
		       (while (not (or (event.is_set)
				       (< 7 count)))
			      (incf count)
			      (setf time_str (dot datetime
						  datetime (now)
						  (strftime (string "%H:%M:%S"))))
			      ,(lprint :level "trace" :msg "time_generator sends" :vars `(time_str))
			      (yield (sse_message (Article time_str)
						  :event (string "message")))
			      (await (asyncio.sleep 1)))
		       (yield (sse_message (Article time_str)
					   :event (string "close")))
		       ,(lprint :level "trace" :msg "time_generator shutdown")))
	(do0
	 (@app.get (string "/time-sender"))
	 (space async (def time_sender ()
			,(lprint :level "trace" :msg "GET time-sender")
			(return (EventStream (time_generator)))))))



       (@app.get (string "/response-stream"))
       (space async
	      (def  response_stream (prompt_text uid)
		(declare (type str prompt_text uid))
		(space async
		       (def gen ()

			 ,(lprint :level "trace" :msg "GET response-stream" :vars `(prompt_text))
			 (setf include_thought False)
			 (setf config (GenerationConfig
				       :prompt_text prompt_text
				       :model (string "gemini-flash-latest")
				       #+yaml :output_yaml_path #+yaml yaml_filename
				       :use_search False ; True
				       :think_budget (? include_thought -1 0)
				       :include_thoughts include_thought
				       ))
			 ,(lprint :level "trace" :msg "created a genai configuration")
			 (setf job (GenAIJob config))
			 ,(lprint :level "trace" :msg "configured genai job")
			 (space async
				(for (msg (job.run))
				     ,(lprint :level "trace" :msg "genai.job async for" :vars `(msg))
				     (cond ((and include_thought
						 (== (aref msg (string "type"))
						     (string "thought")))
					    (yield (sse_message (Div (fstring "{msg['text']}")
								     :id (fstring "{uid}-thoughts")
								     :data_hx_swap_oob (string "beforeend"))
								:event (string "thought"))))
					   ((== (aref msg (string "type"))
						(string "answer"))
					    (yield (sse_message (Div (fstring "{msg['text']}")
								     :id (fstring "{uid}-answer")
								     :data_hx_swap_oob (string "beforeend")
								     )
								:event (string "answer"))))
					   ((== (aref msg (string "type"))
						(string "complete"))
					    (yield (sse_message (Div (fstring "Final Answer: {msg['text']}")
								     :id (fstring "{uid}-answer")
								     :data_hx_swap_oob (string "innerHTML")
								     )
								:event (string "final_answer")))
					    break)
					   ((== (aref msg (string "type"))
						(string "error"))
					    (yield (sse_message (Div (fstring "Error: {msg['text']}")
								     :id (fstring "{uid}-error")
								     :data_hx_swap_oob (string "innerHTML")
								     )
								:event (string "error")))
					    break))))
			 (yield (sse_message (string "")
					     :event (string "close")))))
		(return (EventStream (gen)))))
       (serve)
       ))))
