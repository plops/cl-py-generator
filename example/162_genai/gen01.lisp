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
	     ))
    
    
    
    (imports-from (sqlite_minutils *)
		  (loguru logger)
		  (google genai)
		  (pydantic BaseModel)
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
    (class Recipe (BaseModel)
	   "title: str"
	   "summary: list[str]")
    (setf client (genai.Client :api_key (os.environ.get (string "GEMINI_API_KEY"))
					;:http_options (types.HttpOptions :api_version (string "v1alpha"))
			       ))
    (setf model (string "gemini-flash-latest"))
    (setf contents (list (types.Content
			  :role (string "user")
			  :parts (list (types.Part.from_text :text (rstring3 "make a summary about the most recent news about trump")))
			  )))
    (setf tools (list (types.Tool :googleSearch (types.GoogleSearch))))
    
    (setf generate_content_config (types.GenerateContentConfig
				   :thinking_config (types.ThinkingConfig :thinkingBudget 24576)
				   :tools tools
				   :response_mime_type (string "application/json")
				   :response_schema (space "list" (list Recipe))))
    (for (chunk (client.models.generate_content_stream
		 :model model
		 :contents contents
		 :config generate_content_config))
	 (print chunk.text :end (string "")))
    
    )
   ))


