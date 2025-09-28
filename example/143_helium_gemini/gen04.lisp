(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

;; TODO:
;; [X] add choice for output language
;; [ ] show spinner
;; [X] allow to store the original youtube link
;; [X] optional timestamps
;; [X] optional user comments
;; [X] optional glossary
;; [X] allow copying the prompt (in case resources are exhausted the user can go to a different provider)
;; [ ] allow user to enter a modified or alternative summary (with comment)
;; [ ] find better examples for comments, or glossary
;; [ ] communicate gemini's evaluation of the content (e.g. harassment) to the user
;; [ ] generate transcript from audio channel
;; [ ] allow editing of the models output
;; [ ] allow local LLM
;; [X] get transcript from link
;; [X] allow concurrent transcript downloads
;; [ ] OAuth
;; [X] Add example with abstract infront of summary
;; [ ] Store the thought process of the model in the database
;; [.] Download main language of the transcript
;; [X] Browsers sometimes translate the model selector, this might mess up the model lookup
;; [X] Log to file with timestamps

;; Tests of individual steps
;; [X] Validate YouTube URL
;; [X] Bring Timestamps into HH:MM:SS format
;; [X] Deduplicate Transcript
;; [X] Convert formatting of the summary from markdown to YouTube
;; [ ] Price estimation

(setf *features* (union *features* '(:example ;; store long example text in python (enable for production, disable for debugging)
				     :emulate ;; don't actually make the calls to gemini api (enable for debugging)
				     :dl ;; download transcript from link
				     :simple ;; use very few gui elements
				     :auth ;; oauth login (requires AUTH_CLIENT_{ID,SECRET} in env)
				     :optional-abstract ;; user can disable abstract generation
				     :copy-prompt ;; transmit the entire prompt (makes page a lot bigger)
				     :show-ip ;; display the origin ip of the request
				     )))
(setf *features* (set-difference *features* '(;:example
					      :emulate
					      ;:simple
					      ;:dl
					      :auth
					      :optional-abstract
					      :copy-prompt
					      ;:show-ip
					      )))

(progn
  (defparameter *project* "143_helium_gemini")

  (load "gen04_data.lisp")
  ;; name input-price output-price context-length harm-civic
  
  (let ((iflash .1)
	(oflash .4)
	(iflashl .075)
	(oflashl .3)
	(ipro 1.25)
	(opro 5)
	(iflash25 .15)
	(oflash25think 3.5)
	(oflash25nothink .6))
    (defparameter *models* `((:name gemini-2.5-flash-preview-09-2025 :input-price .3 :output-price 2.5 :context-length 128_000 :harm-civic nil)
			     (:name gemini-2.5-pro :input-price 1.25 :output-price 10 :context-length 200_000 :harm-civic nil)
			     (:name gemini-2.5-flash-lite-preview-09-2025 :input-price .1 :output-price .4 :context-length 128_000 :harm-civic nil)
			     #+nil
			     ((:name gemini-2.5-flash :input-price .3 :output-price 2.5 :context-length 128_000 :harm-civic nil)
			     (:name gemini-2.5-flash-lite :input-price .1 :output-price .4 :context-length 128_000 :harm-civic nil)

			      (:name gemini-2.5-flash-lite-preview-06-17 :input-price .1 :output-price .4 :context-length 128_000 :harm-civic nil)
			      (:name gemini-2.5-flash-preview-05-20 :input-price ,iflash25 :output-price ,oflash25think :context-length 128_000 :harm-civic nil)
			      
			      (:name gemma-3n-e4b-it :input-price -1 :output-price -1 :context-length 128_000 :harm-civic nil)
			      (:name gemini-2.5-flash-preview-04-17 :input-price ,iflash25 :output-price ,oflash25think :context-length 128_000 :harm-civic nil)
			      (:name gemini-2.5-pro-preview-05-06 :input-price 1.25 :output-price 10.0 :context-length 128_000 :harm-civic nil)
			      (:name gemini-2.5-pro-exp-03-25 :input-price 1.25 :output-price 10.0 :context-length 128_000 :harm-civic nil)
			      (:name gemini-2.0-flash :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-2.0-flash-lite :input-price ,iflashl :output-price ,oflashl :context-length 128_000 :harm-civic t)
			      (:name gemini-2.0-flash-thinking-exp-01-21 :input-price ,iflashl :output-price ,oflashl :context-length 128_000 :harm-civic t)
			      
			      (:name gemini-2.0-flash-exp :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-2.0-pro-exp-02-05 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      (:name gemini-1.5-pro-exp-0827 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      (:name gemini-2.0-flash-lite-preview-02-05 :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-2.0-flash-thinking-exp-01-21 :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-2.0-flash-001 :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t) ;; the price is now fixed at $.1 for input and $.4 for output
			      (:name gemini-exp-1206 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      (:name gemini-exp-1121 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      (:name gemini-exp-1114 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      (:name learnlm-1.5-pro-experimental :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      (:name gemini-1.5-flash-002 :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-1.5-pro-002 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			      
			      (:name gemini-1.5-pro-exp-0801 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic nil)
			      (:name gemini-1.5-flash-exp-0827 :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-1.5-flash-8b-exp-0924 :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemini-1.5-flash-latest :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
			      (:name gemma-2-2b-it :input-price -1 :output-price -1 :context-length 128_000 :harm-civic nil)
			      (:name gemma-2-9b-it :input-price -1 :output-price -1 :context-length 128_000 :harm-civic nil)
			      (:name gemma-2-27b-it :input-price -1 :output-price -1 :context-length 128_000 :harm-civic nil)
			      (:name gemma-3-27b-it :input-price -1 :output-price -1 :context-length 128_000 :harm-civic nil)
			      (:name gemini-1.5-flash :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic nil)
			      (:name gemini-1.5-pro :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic nil)
			      (:name gemini-1.0-pro :input-price .5 :output-price 1.5 :context-length 128_000 :harm-civic nil))
			     )))
  (defparameter *languages* `(en ;de fr ch nl pt cz it jp ar
			      ))
  (defparameter *idx* "04") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format		;(- (time.time) start_time)
                   ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  (let* ((notebook-name "host")
	 (example-output-nocomments  *example-output*)
	 (example-output-comments  *example-output-with-comments*)
	 (example-input *example-input*)
	 (db-cols `((:name identifier :type int)
		    (:name model :type str)
		    (:name transcript :type str :no-show t)
		    (:name host :type str :no-show t)
		    (:name original_source_link :type str)
		    (:name include_comments :type bool :no-show t)
		    (:name include_timestamps :type bool :no-show t)
		    (:name include_glossary :type bool :no-show t)
		    #+optional-abstract (:name generate_abstract :type bool)
		    (:name output_language :type str :no-show t)
		    ,@(loop for e in `(summary timestamps)
			    appending
			    `((:name ,e :type str :no-show t)
			      ,@(loop for (f f-type) in `((done bool)
							  (input_tokens int)
							  (output_tokens int)
							  (timestamp_start str)
							  (timestamp_end str))
				      collect
				      `(:name ,(format nil "~a_~a" e f) :type ,f-type :no-show t))))
		    (:name timestamped_summary_in_youtube_format :type str :no-show t)
		    (:name cost :type float :no-show t)
		    (:name embedding :type bytes)
		    (:name full_embedding :type bytes)
		    ))
	 )

    (let* ((l-steps0 `((:step-name validate_youtube_url
			:code (do0
			       "#!/usr/bin/env python3"
			       (imports (re))
			       (def validate_youtube_url (url)
				 (string3 "Validates various YouTube URL formats. Returns normalized YouTube video identifier as a string where unneccessary information has been removed. False if the Link doesn't match acceptable patterns.")
				 (setf patterns
				       (list
					 ;; standard watch link
					(rstring3 "^https://(www\\.)?youtube\\.com/watch\\?v=([A-Za-z0-9_-]{11}).*")
					;; live stream link
					(rstring3 "^https://(www\\.)?youtube\\.com/live/([A-Za-z0-9_-]{11}).*")
					;; shortened link
					(rstring3 "^https://(www\\.)?youtu\\.be/([A-Za-z0-9_-]{11}).*")
					))
				 (for (pattern patterns)
				      (setf match (re.match pattern url))
				      (when match
					;; (print (match.groups))
					(return (aref (match.groups) 1))))
				 (print (string "Error: Invalid YouTube URL"))
				 (return False)))
			:test (do0
			       
			       (assert
				(== (string "0123456789a")
				    (validate_youtube_url (string "https://www.youtube.com/live/0123456789a"))))
			       (assert
				(== (string "0123456789a")
				    (validate_youtube_url (string "https://www.youtube.com/live/0123456789a&abc=123"))))
			       (assert
				(== (string "_123456789a")
				    (validate_youtube_url (string "https://www.youtube.com/watch?v=_123456789a&abc=123"))))
			       (assert
				(== (string "_123456789a")
				    (validate_youtube_url (string "https://youtube.com/watch?v=_123456789a&abc=123"))))
			       (comments "FIXME: I'm not sure if a Youtube-ID that starts with a - character is handled correctly in the downstream pipeline")
			       (assert
				(== (string "-123456789a") 
				    (validate_youtube_url (string "https://www.youtu.be/-123456789a&abc=123"))))
			       (assert
				(== (string "-123456789a")
				    (validate_youtube_url (string "https://youtu.be/-123456789a&abc=123"))))
			       
			       (assert
				(== False
				    (validate_youtube_url (string "http://www.youtube.com/live/0123456789a"))))))
		       (:step-name parse_vtt_file
			:code (do0
			       "#!/usr/bin/env python3"
			       (imports ( ;re
					 webvtt))
			       (def parse_vtt_file (filename)
				 (rstring3 "load vtt from <filename>. Returns deduplicated transcript as string with second-granularity timestamps")
				 (do0
				  
				  (setf old_text (list (string "__bla__"))
					old_time (string "00:00:00"))
				  (setf out (list (dictionary :text (string ""))))
				  (comments "collect transcript. perform deduplication")
				  (for (c (webvtt.read filename))
				       (when (!= (aref (aref out -1) (string "text"))
						 (aref old_text -1))
					 (out.append (dictionary :text (aref old_text -1)
								 :time old_time)))
				       ;;,(lprint :vars `(c.start (c.text.__repr__) old_text out))
				       (setf old_text (dot c.text (split (string "\\n")))
					     old_time c.start))
				  (setf ostr (string ""))
				  (comments "skip the first two entries of out (they are left over from the initialization)")
				  (for (time_str (aref out (slice 2 "")))
				       (comments "cut away the milliseconds from the time stamps")
				       (setf tstamp_fine (aref time_str (string "time"))
					     tstamp (dot tstamp_fine (aref (split (string ".")) 0))
					     caption (aref time_str (string "text")))
				       (incf ostr (fstring "{tstamp} {caption}\\n")))
				  (return ostr))))
			:test (do0
			       
			       (assert
				(==
				 (rstring3 "00:00:00 [Music]
00:00:00 welcome to BASF we create chemistry so
00:00:05 it makes sense that we should
00:00:06 familiarize you with the basic chemistry
00:00:08 taught in our poly urethanes Academy
00:00:11 we're going to simplify things a bit in
00:00:13 this video and at the same time cover a
00:00:16 lot of topics so let's get started first
00:00:18 let's introduce you to two of our
00:00:21 leading characters by societies or ISO
00:00:25 and resin let's talk about ISO first
00:00:29 when we make ISO we do so in very large
00:00:32 quantities for our purposes today there
00:00:35 are only a few types of eye soaps pure
00:00:38 MD eyes and TV eyes that's their
00:00:40 nicknames form long and squiggly
00:00:43 chemical structures that's because they
00:00:45 have fewer places to connect to they are
00:00:47 generally used to make flexible products
00:00:50 like seat cushions mattresses and
00:00:52 sealants polymeric MD eyes have many
00:00:55 more places to plug into which creates
00:00:58 more of the structure they are generally
00:01:00 used to make you guessed it rigid
00:01:03 products like picnic coolers foam
00:01:05 insulation and wood boards now when our
00:01:09 customers make a resin they create a
00:01:11 custom formula of additives that include
00:01:14 polygons also supplied by BASF which are
00:01:18 the backbone of the mix polyols make the
00:01:21 majority of the mix kind of like flour
00:01:24 is to a cake batter
00:01:25 polyols determine the physical
00:01:27 properties of the product like how soft
00:01:30 or hard the product is
00:01:32 catalysts they control the speed of the
00:01:35 chemical reaction and how quickly it
00:01:37 cures surfactants determine the cell
00:01:40 structure and influence the flow
00:01:42 pigments determine the color flame
00:01:45 retardants make it savory adhesion
00:01:48 promoters make it stickier and finally
00:01:50 blowing agents help determine the
00:01:53 density and foaming action
00:01:55 at BASF we're proud to supply raw
00:01:58 materials that help our customers
00:02:00 innovate and succeed on ISOs and polyols
00:02:04 combined to make custom formulas for our
00:02:06 customers custom formulas that produce
00:02:09 unique products that are flexible rigid
00:02:33 just the way end-users like them so
00:02:33 there you have it the basics of
00:02:36 polyurethanes from BASF we create
")
				 (parse_vtt_file (string "cW3tzRzTHKI.en.vtt"))))

			       ))
		       
		       (:step-name convert_markdown_to_youtube_format
			:code (do0
			       "#!/usr/bin/env python3"
			       (imports (re))
			       (def convert_markdown_to_youtube_format (text)
			 (rstring3 "In its comments YouTube only allows *word* for bold text, not **word**. Colons or comma can not be fat (e.g. *Description:* must be written as *Description*: to be formatted properly. YouTube comments containing links seem to cause severe censoring. So we replace links.")
			 (do0
			  (comments "adapt the markdown to YouTube formatting")
			  (setf text (text.replace (string "**:")
						   (string ":**")))
			  (setf text (text.replace (string "**,")
						   (string ",**")))
			  (setf text (text.replace (string "**.")
						   (string ".**")))

			  (setf text (text.replace (string "**")
						   (string "*")))

			  (comments "markdown title starting with ## with fat text")
			  (setf text (re.sub (rstring3 "^##\\s*(.*)")
					     (rstring3 "*\\1*")
					     text))


			  (comments "find any text that looks like an url and replace the . with -dot-")

	      
			  ;; text = re.sub(r"((?:https?://)?(?:www\.)?[^\s]+)\.((?:com|org|de|us|gov|net|edu|info|io|co\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr))", r"\1-dot-\2", text)

			  (setf text (re.sub (rstring3 "((?:https?://)?(?:www\\.)?\\S+)\\.(com|org|de|us|gov|net|edu|info|io|co\\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr)")
					     (rstring3 "\\1-dot-\\2")
					     text))
			  (return text))))
			:test (do0
			       (setf r (convert_markdown_to_youtube_format
				     (rstring3 "**Title:**
Let's **go** to http://www.google.com/search?q=hello.")))
			       (assert
				(== (rstring3 "*Title:*
Let's *go* to http://www.google-dot-com/search?q=hello.")
				    r))

			       ))))
	   (l-steps (loop for e in l-steps0
			  and step-index from 1
			  collect
			  (destructuring-bind (&key step-name code test) e
			    `(:step-index ,step-index
			      :step-name ,step-name
			      :code ,code
			      :test ,test
			      )))))
      (loop for e in l-steps
	    collect
	    (destructuring-bind (&key step-index step-name code test) e
	      (write-source
	       (format nil "~a/source04/tsum/s~2,'0d_~a" *path* step-index step-name)
	       code)
	      (write-source
	       (format nil "~a/source04/tsum/t~2,'0d_~a" *path* step-index step-name)
	       `(do0
		 (imports-from 
		  (,(format nil "s~2,'0d_~a" step-index step-name) *))
		 ,test))))
      
      (write-source
       (format nil "~a/source04/tsum/p~a_~a" *path* *idx* notebook-name)
       `(do0
	 "#!/usr/bin/env python3"
	 (comments "Alternative 1: running with uv: GEMINI_API_KEY=`cat api_key.txt` uv run uvicorn p04_host:app --port 5001")
	 (comments "Alternative 2: install dependencies with pip: pip install -U google-generativeai python-fasthtml markdown")
	 #+dl (comments "Alternative 3: install dependencies with micromamba: micromamba install python-fasthtml markdown yt-dlp uvicorn numpy; pip install  webvtt-py")
	 (imports ((genai google.generativeai)
					;google.generativeai.types.answer_types
		   #+auth os
		   google.api_core.exceptions
		   ;re
		   markdown
					; uvicorn
		   sqlite_minutils.db
		   datetime
		   ;difflib
		   #+dl subprocess
		   ;#+dl webvtt
		   time
		   glob
		   (np numpy)
		   os
		   logging))
	 (imports-from 
		       (google.generativeai types))

	 #+nil
	 (imports-from (threading Thread)
		       (functools wraps))

	 #-emulate (imports-from (google.generativeai.types HarmCategory HarmBlockThreshold))

	 (imports-from (fasthtml.common *)
		       #+auth (fasthtml.oauth OAuth GoogleAppClient)
		       (s01_validate_youtube_url *)
		       (s02_parse_vtt_file *)
		       (s03_convert_markdown_to_youtube_format *)
		       (s04_convert_html_timestamps_to_youtube_links *))

	 (do0
	  (comments "Configure logging with UTC timestamps and file output")
	  (class UTCFormatter (logging.Formatter)
		 (def formatTime (self record &key (datefmt None))
		   (setf dt (dot
			     datetime
			     datetime
			     (fromtimestamp record.created
					    :tz datetime.timezone.utc)))
		   (return (dt.isoformat))))
	  (comments "Create formatter with UTC timestamps")
	  (setf formatter (UTCFormatter :fmt (string "%(asctime)s - %(name)s - %(levelname)s - %(message)s")))
	  (comments "Configure root logger")
	  (setf logger (logging.getLogger)
		)
	  (logger.setLevel logging.INFO)
	  (comments "Clear any existing handlers")
	  (logger.handlers.clear)
	  (comments "Console handler")
	  (setf console_handler (logging.StreamHandler))
	  (console_handler.setFormatter formatter)
	  (logger.addHandler console_handler)
	  (comments "File handler")
	  (setf file_handler (logging.FileHandler (string "transcript_summarizer.log")))
	  (file_handler.setFormatter formatter)
	  (logger.addHandler file_handler)
	  (comments "Get logger for this module")
	  (setf logger (logging.getLogger __name__)))

	 
	 #+auth
	 (do0
	  (setf client
		(GoogleAppClient
		 (os.getenv (string "AUTH_CLIENT_ID"))
		 (os.getenv (string "AUTH_CLIENT_SECRET"))))
	  (class Auth (OAuth)
		 (def get_auth (self info ident session state)
		   (when info.email_verified
		     (return (RedirectResponse
			      (string "/")
			      :status_code 303)))))
	  )
	 
	 (do0
	  (comments "Read the demonstration transcript and corresponding summary from disk")
	  (try
	   (do0
	    ,@(loop for e in `((:var g_example_input :data ,*example-input* :fn example_input.txt)
			       (:var g_example_output :data ,*example-output* :fn example_output.txt)
			       (:var g_example_output_abstract :data ,*example-output-abstract* :fn example_output_abstract.txt)
			       )
		    collect
		    (destructuring-bind (&key var data fn) e
		      (with-open-file (str (format nil "~a/source04/tsum/~a" *path* fn)
					   :direction :output
					   :if-exists :supersede
					   :if-does-not-exist :create)
			(format str "~a" data))
		      `(with (as (open (string ,fn))
				 f)
			     (setf ,var (dot f (read)))))))
	   ((as FileNotFoundError e)
			      (logger.error (fstring "Required example file not found: {e}"))
			      raise)))

	 (do0
	  " "
	  (comments "Use environment variable for API key")
	  (setf api_key (os.getenv (string "GEMINI_API_KEY")))
	  )

	 (genai.configure :api_key api_key)
	 
	 " "
	 (setf MODEL_OPTIONS
	       (list
		,@(loop for e in *models*
				collect
				(destructuring-bind (&key name input-price output-price context-length harm-civic) e 
				  `(string 
				   ,(format nil "~a| input-price: ~a output-price: ~a max-context-length: ~a" name input-price output-price context-length))))))


	 " "
	 (def validate_transcript_length (transcript &key (max_words 280_000))
	   (declare (values bool)
		    (type str transcript)
		    (type int max_words))
	   (string3 "Validate transcript length to prevent processing overly large inputs.")
	   (when (or (not transcript)
		     (not (transcript.strip)))
	     (raise (ValueError (string "Transcript cannot be empty"))))
	   (setf words (transcript.split))
	   (when (< max_words (len words))
	     (raise (ValueError (fstring "Transcript too long: {len(words)} words (max: {max_words})"))))
	   (return True))

	 " "
	 (def validate_youtube_id (youtube_id)
	   (declare (values bool)
		    (type str youtube_id))
	   (when (or (not youtube_id)
		      (!= (len youtube_id)
		       11))
	     (return False))
	   
	   (comments "YouTube IDs are alphanumeric with _ and -")
	   (return (all (for-generator (c youtube_id)
				       (or (c.isalnum)
					   (in c (string "_-")))))))
	 
	 " "
	 (def render (summary)
	   (declare (type Summary summary))
	   (setf identifier summary.identifier)
	   (setf sid (fstring "gen-{identifier}"))
	   (cond
	     (summary.timestamps_done
	      (return (generation_preview identifier)
		      #+nil(Div (Pre summary.timestamped_summary_in_youtube_format)
				:id sid
				:data_hx_post (fstring "/generations/{identifier}")
				:data_hx_trigger (string "")
				:data_hx_swap (string "outerHTML"))))
	     (summary.summary_done
	      (return (Div		;(Pre summary.summary)
		       (NotStr (markdown.markdown summary.summary))
		       :id sid
		       :data_hx_post (fstring "/generations/{identifier}")
		       :data_hx_trigger (? summary.timestamps_done
				      (string "")	
				      (string #+emulate "" #-emulate "every 1s"))
		       :data_hx_swap (string "outerHTML"))))
	     (t
	      (return (Div		;(Pre summary.summary)
		       (NotStr (markdown.markdown summary.summary))
		       :id sid
		       :data_hx_post (fstring "/generations/{identifier}")
		       :data_hx_trigger (string #+emulate "" #-emulate "every 1s")
		       :data_hx_swap (string "outerHTML")))))
	   )
	 
	 " "
	 (comments "open website")
	 (comments "summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
	 (setf (ntuple app rt summaries Summary)
	       (fast_app :db_file (string "data/summaries.db")
			 :live False	;True
			 :render render
			 ,@(loop for e in db-cols
				 appending
				 (destructuring-bind (&key name type no-show) e
				   `(,(make-keyword (string-upcase (format nil "~a" name)))
				     ,type)))

			 
			 :pk (string "identifier")
			 ))

	 #+auth
	 (do0
	  (setf oauth (Auth app client))
	  " "
	  (@rt (string "/login"))
	  (def get (req)
	    (return (ntuple
		     (Div (P (string "Not logged in")))
		     (A (string "Log in")
			:href (oauth.login_link req))))))
	 

	 (setf documentation 
	       
	       (+
		(string3 "**Get Your Summary:**

1.  For **YouTube videos**, paste the link into the input field for automatic transcript download.
2.  For **any other text** (like articles, meeting notes, or non-YouTube transcripts), paste the content directly into the text area below.
3.  Click 'Summarize' to get your summary.

**Important Notes:**

*   **For YouTube Links:** Automatic download requires **English subtitles** on the video. If they are missing, please use the manual method below.
*   **For Any Text Content:** You can summarize any text by pasting it into the text area. This is the best method for articles, your own notes, or transcripts from other sources.
    1.  **Copy** the entire text you wish to summarize.
    2.  **Paste** it into the '(Optional) Paste YouTube transcript here' field.
    3.  **Please note:** The summarizer is optimized for content that includes timestamps (e.g., `00:15:23 Key point is made.`). While it works well for any text, providing timestamped transcripts will produce the most detailed and well-structured summaries.


**For Very Long Content (e.g., over 2 hours):**

*   Select the **Pro model** for summarizing long-form content. It is equipped with advanced reasoning capabilities that produce more concise and higher-quality summaries.
*   **Performance Tip:** For the fastest results, you may experience better performance when using the Pro model on weekends or outside of standard US business hours.
")
		#+copy-prompt
		(string3 "*   If the Pro limit is reached (or if you prefer using your own tool), use the **Copy Prompt** button, paste the prompt into your AI tool, and run it there.
")
		))
	 

	 	 
	 #+dl
	 (def get_transcript (url identifier)
	   (comments "Call yt-dlp to download the subtitles. Modifies the timestamp to have second granularity. Returns a single string")
	   (try
	    (do0
	     (setf youtube_id (validate_youtube_url url))
	     (unless youtube_id
	       (logger.warning (fstring "Invalid YouTube URL: {url}"))
	       (return (string "URL couldn't be validated")))
	     (unless (validate_youtube_id youtube_id)
	       (logger.warning (fstring "Invalid YouTube ID format: {youtube_id}"))
	       (return (string "Invalid YouTube ID format")))
	   
	     (setf sub_file (fstring "/dev/shm/o_{identifier}"))
	     (setf sub_file_en (fstring "/dev/shm/o_{identifier}.en-orig.vtt"))
	     (comments "First, try to get English subtitles")
	     (setf cmds_en (list (string "uvx")
				 (string "yt-dlp")
				 (string "--skip-download")
				 (string "--write-auto-subs")
				 (string "--write-subs")
				 (string "--cookies-from-browser")
				 (string "firefox")
				 (string "--sub-langs")
				 (string "en-orig")
				 (string "-o")
				 sub_file
				 (string "--")
				 youtube_id))
	     (logger.info (fstring "Downloading English subtitles: {' '.join(cmds_en)}"))
	     (setf result
		   (subprocess.run cmds_en
				   :capture_output True
				   :text True
				   :timeout 60))
	     (when (!= result.returncode 0)
	       (logger.warning (fstring "yt-dlp failed with return code {result.returncode}: {result.stderr}")))

	     (setf sub_file_to_parse None)
	     (if (os.path.exists sub_file_en)
		 (setf sub_file_to_parse sub_file_en)
		 (do0
		  (comments "If English subtitles are not found, try to download any available subtitle")
		  (logger.info (string "English subtitles not found. Trying to download subtitles in other languages."))

		  (for (lang (list ,@(loop for e in `(en de zh iw lv fr pl ja)
					   collect
					   `(string ,e))))
		   (do0
		    (setf cmds_any (list (string "uvx")
					 (string "yt-dlp")
					 (string "--skip-download")
					 (string "--write-auto-subs")
					 (string "--write-subs")
					 (string "--cookies-from-browser")
					 (string "firefox")
					 (string "--sub-langs")
					 lang
					 (string "-o")
					 sub_file
					 (string "--")
					 youtube_id))
		    (logger.info (fstring "Downloading any subtitles: {' '.join(cmds_any)}"))
		    (setf result (subprocess.run cmds_any
						 :capture_output True
						 :text True
						 :timeout 60))
		    (comments "Find the downloaded subtitle file")
		    (setf subtitle_files (glob.glob (fstring "{sub_file}.*.vtt")))
		    (when subtitle_files
		      (setf sub_file_to_parse (aref subtitle_files 0))
		      (logger.info (fstring "Parse transcript from {sub_file_to_parse} out of the subtitle files: {subtitle_files}"))
		      break)))))
	     
	     (setf ostr (string "Problem getting subscript."))

	     (if (and sub_file_to_parse
		      (os.path.exists sub_file_to_parse))
		 (try
		  (do0
		   (setf ostr 
			 (parse_vtt_file sub_file_to_parse))
		   (logger.info (fstring "Successfully parsed subtitle file: {sub_file_to_parse}"))
		   (os.remove sub_file_to_parse))
		  
		  (FileNotFoundError
		   (logger.error (fstring "Subtitle file not found: {sub_file_to_parse}"))
		   (setf ostr (string "Error: Subtitle file disappeared")))
		  (PermissionError
		   (logger.error (fstring "Permission denied removing file: {sub_file_to_parse}"))
		   (setf ostr (string "Error: Permission denied cleaning up subtitle file")))
		  ("Exception as e"
		   (logger.error (fstring "Error processing subtitle file: {e}")
				 )
		   (setf ostr (fstring "Error: problem when processing subtitle file {e}"))))
		 (do0
		  (logger.error (string "No subtitle file found"))
		  (setf ostr (string "Error: No subtitles found for this video. Please provide the transcript manually."))))
	     (do0
	      (comments "Cleanup any other subtitle files that might have been downloaded")
	      (setf other_subs (glob.glob (fstring "{sub_file}.*.vtt")))
	      (for (sub other_subs)
		   (try (os.remove sub)
			((as OSError e)
			 (logger.warning (fstring "Error removing file {sub}: {e}"))))))
	     
	     (return ostr))
	    (subprocess.TimeoutExpired
	     (logger.error (fstring "yt-dlp timeout for identifier {identifier}"))
	     (return (string "Error: Download timeout")))
	    ("Exception as e"
	     (logger.error (fstring "Unexpected error in get_transcript: {e}"))
	     (return (fstring "Error712: {str(e)}")))))
	 " "
	 (setf documentation_html
	       (markdown.markdown documentation))
	 (@rt (string "/"))
	 (def get (request)
	    (declare (type Request request))
	    ;; how to format markdown: https://isaac-flath.github.io/website/posts/boots/FasthtmlTutorial.html
	    
	    (logger.info (fstring "Request from host: {request.client.host}"))
	    (setf nav (Nav
		       (Ul (Li (H1 (string "RocketRecap Content Summarizer"))))
		       (Ul (Li (A (string "Map")
				  :href (string "https://rocketrecap.com/exports/index.html")))
			   (Li (A (string "Extension")
				  :href (string "https://rocketrecap.com/exports/extension.html")))
			   (Li (A (string "Privacy Policy")
				  :href (string "https://rocketrecap.com/exports/privacy.html")))
			   (Li (A (string "Demo Video")
				  :href (string "https://www.youtube.com/watch?v=ttuDW1YrkpU")))
			   (Li (A (string "Documentation")
				  :href (string "https://github.com/plops/gemini-competition/blob/main/README.md")))
			   #+auth
			   (Li (A (string "Log out")
				  :href (string "/logout"))))))

	   (setf transcript (Textarea :placeholder (string "(Optional) Paste YouTube transcript here")
				      :style (string "height: 300px; width: 60%;")
				      :name (string "transcript")
				      :id (string "transcript-paste")))
	   
	    (setf
	     selector (list (for-generator (opt MODEL_OPTIONS)
					   (Option opt :value opt
						   :label (dot opt (aref (split (string "|")) 0)))))
	     model (Div
		    (Label (string "Select Model")
			   :_for (string "model-select")
			   :cls (string "visually-hidden"))
		    (Select
		     *selector
		     :id (string "model-select")
		     :style (string "width: 100%;")
		     :name (string "model"))
			:style (string "display: flex; align-items: center; width: 100%;")))
	    #+nil (setf model (aref (model.split (string "\\n"))
				    0))
	    #+nil ,(lprint :vars `((type model)))
	    
	    (setf form
		  (Form
                   (Fieldset 
			     (Legend (string "Submit Text for Summarization"))
			     (Div 
			      (Label (string "Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)")
				     :_for (string "youtube-link"))
			      (Textarea :placeholder (string "Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)")
					:id (string "youtube-link")
					:name (string "original_source_link"))
			      (Label (string "(Optional) Paste YouTube transcript here")
				     :_for (string "transcript-paste"))
			      transcript
			      model
			      #+nil
			      (Div (Label (string "Output Language") :_for (string "output_language"))
				   (Select
				    ,@(loop for e in *languages*
					    collect
					    `(Option (string ,e)))
				    :style (string "width: 100%;")
				    :name (string "output_language")
				    :id (string "output_language"))
				   :style (string #+simple "display: none; align-items: center; width: 100%;"
						  #-simple "display: flex; align-items: center; width: 100%;"))
		     
			      #+nil
			      ,@(loop for (e f default) in `((include_comments "Include User Comments" False)
							     (include_timestamps "Include Timestamps" True)
							     (include_glossary "Include Glossary" False)
							     #+optional-abstract (generate_abstract "Generate Abstract" True)
							     )
				      collect
				      `(Div
			       
					(Input :type (string "checkbox")
					       :id (string ,e)
					       :name (string ,e)
					       :checked ,default)
					(Label (string ,f) :_for (string ,e))
					:style #+simple (string "display: none; align-items: center; width: 100%;")
					#-simple (string "display: flex; align-items: center; width: 100%;")))
		     
			      (Button (string "Summarize Transcript"))
			      :style (string "display: flex; flex-direction:column;"))
		    )
		   :data_hx_post (string "/process_transcript")
		   :data_hx_swap (string "afterbegin")
		   :data_hx_target (string "#summary-list") ))


	    (setf summaries_to_show (summaries :order_by (string "identifier DESC")
					       :limit 3))
	   
	    (setf summary_list_container (Div *summaries_to_show
					      :id (string "summary-list")))
	   
	    (return (ntuple (Title (string "Video Transcript Summarizer"))
			    (Meta :name (string "description")
			      :content (string "Get AI-powered summaries of YouTube videos and websites. Paste a link or transcript to receive a concise summary with timestamps.")
			      )
			   
			    (Main nav
				  (NotStr documentation_html)
				  #+nil chrome_ext_promo
				  form
				  summary_list_container 
				  (Script (string3 "function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}"))
				  :cls (string "container"))
			    (Style
			     (string3 ".visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}")))))

	 
	 
	 " "
	 (comments "A pending preview keeps polling this route until we return the summary")
	 (def generation_preview (identifier)
	   (setf sid (fstring "gen-{identifier}"))
	   (setf text (string "Generating ...")
		 trigger (string #+emulate "" #-emulate "every 1s"))
	   (do0
	    (setf price_input (dict
			       ,@(loop for e in *models*
				     collect
				     (destructuring-bind (&key name input-price output-price context-length harm-civic) e 
				       `((string ,name)
					 ,input-price)))))
	  (setf price_output (dict
			      ,@(loop for e in *models*
				      collect
				      (destructuring-bind (&key name input-price output-price context-length harm-civic) e 
					`((string ,name)
					  ,output-price))))))
	   (try
	    (do0
	     (setf s (aref summaries identifier))
	     
	     (cond
	       (s.timestamps_done
		(comments "this is for <= 128k tokens")
		(setf real_model (dot s model (aref (split (string "|")) 0)))
		(setf price_input_token_usd_per_mio -1)
		(setf price_output_token_usd_per_mio -1)
		(try
		 (do0 (setf price_input_token_usd_per_mio (aref price_input real_model))
		      (setf price_output_token_usd_per_mio (aref price_output real_model)))
		 ("Exception as e"
		  pass))
		#+nil
		(if (or (dot s model (startswith (string "gemini-1.5-pro")))
			(dot s model (startswith (string "gemini-2.0-pro")))
			(dot s model (startswith (string "gemini-2.5-pro")))) 
		    (setf price_input_token_usd_per_mio 1.25
			  price_output_token_usd_per_mio 5.0)
		    (if (in (string "flash") (dot s model )) 
			(setf price_input_token_usd_per_mio 0.1
			      price_output_token_usd_per_mio 0.4)
			(if (dot s model (startswith (string "gemini-1.0-pro")))
			    (setf price_input_token_usd_per_mio 0.5
				  price_output_token_usd_per_mio 1.5)
			    (setf price_input_token_usd_per_mio -1
				  price_output_token_usd_per_mio -1)))
		    )
		(setf input_tokens (+ s.summary_input_tokens
				      s.timestamps_input_tokens)
		      output_tokens (+ s.summary_output_tokens
				       s.timestamps_output_tokens)
		      )
		(setf cost (+ (* (/ input_tokens 1_000_000)
				 price_input_token_usd_per_mio)
			      (* (/ output_tokens 1_000_000)
				 price_output_token_usd_per_mio)))
		(summaries.update :pk_values identifier
				  :cost cost)
		(if (< cost .02)
		    (setf cost_str (fstring "${cost:.4f}"))
		    (setf cost_str (fstring "${cost:.2f}")))
		(setf text (fstring3 "*AI Summary*

{s.timestamped_summary_in_youtube_format}

AI-generated summary created with {s.model.split('|')[0]} for free via RocketRecap-dot-com. (Input: {input_tokens:,} tokens, Output: {output_tokens:,} tokens, Est. cost: {cost_str}).")

		      
		      trigger (string ""))


		)
	       (s.summary_done
		(setf text s.summary))
	       ((and "s.summary is not None" (< 0 (len s.summary)))
		(setf text s.summary))

	       ((< (len s.transcript))
		(setf text (fstring "Generating from transcript: {s.transcript[0:min(100,len(s.transcript))]}"))))

	     ;; title of row
	     (setf summary_details
		   (Div ,@(remove-if
			   #'null
			   (loop for e in db-cols
				 collect
				 (destructuring-bind (&key name type no-show) e
				   (unless no-show
				     (cond ((eq name 'original_source_link)
					    `(A
					      (fstring "{s.original_source_link}")
					      :target (string "_blank")
					      :href (fstring "{s.original_source_link}")
					      :id (fstring "source-link-{identifier}")))
					   ((member name `(embedding
							   full_embedding
							   ))
					    nil)
					   (t `(P
						(B (string ,(format nil "~a:" name)))
						(Span (fstring ,(format nil "{s.~a}"
									(string-downcase name)))))))
				     #+nil
				     (format nil "~a: {s.~a}" name (string-downcase name))))))

			:cls (string "summary-details")
			))
	     (setf summary_container (Div summary_details
					  :cls (string "summary-container")))
	     (setf title summary_container)
	     (setf html0 (markdown.markdown s.summary ; :extensions (list (string "nl2br"))
					    ))
	     (if (== (string "") html0)
		 (do0
		  (setf real_model (dot s model (aref (split (string "|")) 0)))
		  (setf html (fstring "Waiting for {real_model} to respond to request...")))
		 (setf html (replace_timestamps_in_html html0 s.original_source_link)))
	     (setf hidden_pre_for_copy
		   (Div (Pre text :id (fstring "pre-{identifier}"))
			:id (fstring "hidden-markdown-{identifier}")
			:style (string "display: none;")))
	     (setf card_content
		   (list
		    (Header
		     (H4 (A (fstring "{s.original_source_link}")
			    :target (string "_blank")
			    :href (fstring "{s.original_source_link}")))
		     (P (fstring "ID: {s.identifier} | Model: {s.model.split('|')[0]}")
			:style (string "font-size: 0.9em; color: var(--pico-secondary-foreground); margin-bottom: 0;")))
		    (Div (NotStr html))
		    (Footer
		     hidden_pre_for_copy
		     (Button (string "Copy Summary")
			     :onclick (fstring "copyPreContent('pre-{identifier}')")
			     :cls (string "outline")))))
	     ;; Conditionally add HTMX attributes for polling
	     (if (== trigger (string ""))
		 (return (Article *card_content :id sid))
		 (do0
		  ;; Polling  state. Create attributes and add the spinner only if still loading.
		  (setf attrs (dictionary
				   :id sid
				   :data_hx_post (fstring "/generations/{identifier}")
				   :data_hx_trigger trigger
				   :data_hx_swap (string "outerHTML")))
		  ;; The process is still loading if the end timestamp has not yet been written.
		  ;; This correctly handles success, all errors, and in-progress states.3
		  (unless s.summary_timestamp_end
		    (setf (aref attrs (string "aria-busy"))
			  (string "true"))
		    (setf (aref attrs (string "aria-live"))
			  (string "polite")))
		  (return (Article *card_content
				   **attrs))))
	     
	     )

	    ("Exception as e"		; NotFoundError ()
	     (return (Article
		      (Header (H4 (fstring "Error processing Summary ID: {identifier}")))
		      (Div
		       (P (string "An error occurred while trying to render the summary. The page will continue to refresh automatically."))
		       (P (B (string "Details:")) (Code (fstring "{e}")))
		       (Pre text)) ; Shows the last known state, e.g., "Generating..."
		      
		      :id sid
		      :data_hx_post (fstring "/generations/{identifier}")
		      :data_hx_trigger trigger
		      :data_hx_swap (string "outerHTML")
		      :style (string "border-color: var(--pico-del-color);"))))
	 
	 ))
	 
	 " "
	 (@app.post (string "/generations/{identifier}"))
	 (def get (identifier)
	   (declare (type int identifier))
	   (return (generation_preview identifier)))

	 " "
	 (do0
	  (@rt (string "/process_transcript"))
	  (def post (summary request)
	    (declare (type Summary summary)
		     (type Request request))
	    (setf summary.host request.client.host
		  summary.summary_timestamp_start (dot datetime
						       datetime
						       (now)
						       (isoformat))
		  summary.summary (string ""))
	    (when (== 0 (len summary.transcript))
	      (setf summary.summary (string "Downloading transcript...")))
	    (setf s2 (summaries.insert summary))
	    (download_and_generate s2.identifier)
	    (return (generation_preview s2.identifier))))

	 (do0
	  " "
	  @threaded
	  (def download_and_generate (identifier)
	   (declare (type int identifier))

	    (try
	     (do0
	     
	      (setf s (wait_until_row_exists identifier))
	      (when (== s -1)
		(logger.error (fstring "Row {identifier} never appeared in database"))
		(return))

	      (when (or (is s.transcript None)
			(== 0 (len s.transcript)))
		(comments "No transcript given, try to download from URL")
		(setf transcript (get_transcript s.original_source_link identifier))
		(summaries.update :pk_values identifier
				  :transcript transcript))

	      (comments "re-fetch summary with transcript")
	      (setf s (aref summaries identifier))

	      (do0
	       (comments "Validate transcript length")
	       (try
		(validate_transcript_length s.transcript)
		((as ValueError e)
		 (logger.error (fstring "Transcript validation failed for {identifier}: {e}"))
		 (summaries.update :pk_values identifier
				   :summary (fstring "Error1031: {str(e)}")
				   :summary_done True)
		 (return))))
	      
	      (setf words (s.transcript.split))
	      (when (< (len words) 30)
		(summaries.update :pk_values identifier
				  :summary (string "Error: Transcript is too short. Probably I couldn't download it. You can provide it manually.")
				  :summary_done True)
		(return))
	      (when (< 280_000 (len words))
		(when (in (string "-pro") s.model)
		  (summaries.update :pk_values identifier
				    :summary (string "Error: Transcript exceeds 280,000 words. Please shorten it or don't use the pro model.")
				    :summary_done True)
		  (return)))
	      (logger.info (fstring "Processing link: {s.original_source_link}"))
	      (summaries.update :pk_values identifier
				:summary (string ""))
	      (generate_and_save identifier))
	     ("Exception as e"
	      (logger.error (fstring "Error in download_and_generate for {identifier}: {e}"))
	      (try
	       (summaries.update :pk_values identifier
				 :summary (fstring "Error1055: {str(e)}")
				 :summary_done True)
	       ("Exception as update_error"
		(logger.error (fstring "Failed to update database with error for {identifier}: {update_error}"))))))))
	 #+nil
	 (do0
	  (@rt (string "/process_transcript"))
	  (def post (summary request)
	    (declare (type Summary summary)
		     (type Request request))
	    
	    #+dl
	    (when (== 0 (len summary.transcript))
	      (comments "No transcript given, try to download from URL")
	      (setf summary.transcript (get_transcript summary.original_source_link))
	      )
	    (setf words (summary.transcript.split))
	    (when (< (len words) 30)
	      (return (Div (string "Error: Transcript is too short. No summary necessary")
			   :id (string "summary"))))
	    (when (< 280_000 (len words))
	      (when (in (string "-pro") (dot summary model))
		(return (Div (string "Error: Transcript exceeds 280,000 words. Please shorten it or don't use the pro model.")
			     :id (string "summary")))))
	    (setf summary.host request.client.host)
	    (setf summary.summary_timestamp_start (dot datetime
						       datetime
						       (now)
						       (isoformat)))
	    (print (fstring "link: {summary.original_source_link}") )
	    (setf summary.summary (string ""))
	    
	    (setf s2 (summaries.insert summary))
	    (comments "first identifier is 1")
	    (generate_and_save s2.identifier)

	    (return (generation_preview s2.identifier))))

	 " "
	 (def wait_until_row_exists (identifier)
	   (for (i (range 400))
		(try
		 (do0 (setf s (aref summaries identifier))
		      (return s))
		 (sqlite_minutils.db.NotFoundError
		  (logger.debug (fstring "Entry {identifier} not found, attempt {i + 1}")))
		 ("Exception as e"
		  (logger.error (fstring "Unknown exception waiting for row {identifier}: {e}"))))
		(time.sleep .1))
	   (logger.error (fstring "Row {identifier} did not appear after 400 attempts"))
	   (return -1))
	 
	 
	 
	 " "
	 (def get_prompt (summary)
	   (declare (type Summary summary)
		    (values str))
	   (rstring3 "Generate prompt from a given Summary object. It will use the contained transcript.")
	   (setf prompt (fstring3 "Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding abstract and summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please give an abstract of the transcript and then summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
{g_example_input}
Example Output:
{g_example_output_abstract}
{g_example_output}
Here is the real transcript. Please summarize it: 
{(summary.transcript)}"))

	   (return prompt)
	   )
	 " "
	 (def generate_and_save (identifier)
	  (declare (type int identifier))
	   (string3 "
    Generates a summary for the given identifier, stores it in the database, and computes embeddings for both
    the transcript and the summary. Handles errors and updates the database accordingly.

    Args:
        identifier (int): The unique identifier for the summary entry in the database.")
	   (logger.info (fstring "generate_and_save id={identifier}"))
	   (try
	    (do0
	    
	     
	     
	     (setf s (wait_until_row_exists identifier))
	     (when (== s -1)
	       (logger.error (fstring "Could not find summary with id {identifier}"))
	       (return))
	     (logger.info (fstring "generate_and_save model={s.model}"))
	     #-emulate
	     (do0
	      (setf m (genai.GenerativeModel (dot s model (aref (split (string "|")) 0))))
	      (setf safety (dict (HarmCategory.HARM_CATEGORY_HATE_SPEECH HarmBlockThreshold.BLOCK_NONE)
				 (HarmCategory.HARM_CATEGORY_HARASSMENT HarmBlockThreshold.BLOCK_NONE)
				 (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT HarmBlockThreshold.BLOCK_NONE)
				 (HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT HarmBlockThreshold.BLOCK_NONE)
					;(HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY HarmBlockThreshold.BLOCK_NONE)
				 )))
	     (do0 ;try
	      (do0
	       (do0 
		(setf prompt (get_prompt s))
		(do0
		 (setf response (m.generate_content
				 prompt
				 ;:tools (string "google_search_retrieval")
				 :safety_settings safety
				 :stream True))
		 (for (chunk response)
		      (try
		       (do0
			(logger.debug (fstring #+nil
					       "Adding text chunk to id={identifier}: {chunk.text}"
					       "Adding text chunk to id={identifier}"))
		      
			(summaries.update :pk_values identifier
					  :summary (+ (dot (aref summaries identifier)
							   summary)
						      chunk.text)))
		       ((as ValueError e)
				   (logger.warning (fstring "ValueError processing chunk for {identifier}: {e}"))
			(summaries.update :pk_values identifier
					  :summary (+ (dot (aref summaries identifier)
							   summary)
						      (fstring "\\nError: value error {str(e)}")))
				   )
		       ("Exception as e"
			(logger.error (fstring "Error processing chunk for {identifier}: {e}"))
			(summaries.update :pk_values identifier
					  :summary (+ (dot (aref summaries identifier)
							   summary)
						      (fstring "\\n[Error1189: {str(e)}]"))))
		       )
		      )))
	       (setf prompt_token_count response.usage_metadata.prompt_token_count
		     candidates_token_count response.usage_metadata.candidates_token_count)
	       (try
		(do0
		 (logger.info (fstring "Usage metadata: {response.usage_metadata}"))
		 (setf thinking_token_count response.usage_metadata.thinking_token_count)
		 )
		(AttributeError
		 (logger.info (string "No thinking token count available"))
		 (setf thinking_token_count 0)))
	       
	       (summaries.update :pk_values identifier
				 :summary_done True
				 :summary_input_tokens #+emulate 0 #-emulate prompt_token_count
				 :summary_output_tokens #+emulate 0 #-emulate (+ candidates_token_count
										 thinking_token_count)
				 :summary_timestamp_end (dot datetime
							     datetime
							     (now)
							     (isoformat))

				 :timestamps (string "") 
				 :timestamps_timestamp_start (dot datetime
								  datetime
								  (now)
								  (isoformat))))
	       
	       )

	   
	     
	   
	     )

	    ((as google.api_core.exceptions.ResourceExhausted e)
	       (logger.error (fstring "Resource exhausted for {identifier}: {e}"))
	       (summaries.update :pk_values identifier
				 :summary_done False
			       
				 :summary (+ (dot (aref summaries identifier)
						  summary)
					     (string "\\nError1234: resource exhausted"))
				 :summary_timestamp_end (dot datetime
							     datetime
							     (now)
							     (isoformat))

				 :timestamps (string "") 
				 :timestamps_timestamp_start (dot datetime
								  datetime
								  (now)
								  (isoformat)))
	       (return))

	    ("Exception as e"
	     (logger.error (fstring "Unexpected error in generate_and_save for {identifier}: {e}") )
	     (try
	      (summaries.update :pk_values identifier
				:summary_done False
				:summary (+ (dot (aref summaries identifier)
						 summary)
					    (fstring "Error1254: {str(e)}"))
				:summary_timestamp_end (dot datetime
							    datetime
							    (now)
							    (isoformat)))
	      ((as Exception update_error)
	       (logger.error (fstring "Failed to update database with error for {identifier}: {update_error}"))
	       ))
	     (return))
	    
	    #+nil ("Exception as e"
		   (logger.error (fstring "Error during embedding or final update for identifier {identifier}: {e}") )))
	   (try
	    (do0
	     (comments "Generate and store the embedding of the transcript")
	     (setf transcript_text (dot (aref summaries identifier)
					transcript))
	     (when transcript_text
	       (logger.info (fstring "Generating embedded transcript: {identifier}..."))
	       (setf embedding_result (genai.embed_content
				       :model (string "models/embedding-001")
				       :content transcript_text
				       :task_type (string "clustering")))
	       (setf vector_blob
		     (dot (np.array (aref embedding_result (string "embedding"))
				    :dtype np.float32)
			  (tobytes))
		     )
	       (summaries.update :pk_values identifier
				 :full_embedding vector_blob)
	       (logger.info (fstring "Embedding stored for identifier {identifier}."))))
	    (google.api_core.exceptions.ResourceExhausted
	     (logger.warning (string "Resource exhausted when embedding full transcript")))
	    ((as Exception e)
	     (logger.error (fstring "Error during full embedding for identifier {identifier}: {e}"))))

	   

	   (try
	    (do0

	     (setf text (dot (aref summaries identifier)
			     summary))
	     (setf text (convert_markdown_to_youtube_format text))
	     (summaries.update :pk_values identifier
			       :timestamps_done True
			       :timestamped_summary_in_youtube_format text
			       :timestamps_input_tokens 0 ; response2.usage_metadata.prompt_token_count
			       :timestamps_output_tokens 0 ; response2.usage_metadata.candidates_token_count
			       :timestamps_timestamp_end (dot datetime
							      datetime
							      (now)
							      (isoformat))))

	    (google.api_core.exceptions.ResourceExhausted
	     (logger.warning (string "Resource exhausted during summary update"))
	     (summaries.update :pk_values identifier
			       :timestamps_done False
			       
			       :timestamped_summary_in_youtube_format (fstring "resource exhausted")
			       :timestamps_timestamp_end (dot datetime
							      datetime
							      (now)
							      (isoformat)))
	     )
	    ((as Exception e)
	     (logger.error (fstring "Error during summary update for identifier {identifier}: {e}"))))


	   (try
	    (do0
	     (comments "Generate and store the embedding of the summary")
	     (setf summary_text (dot (aref summaries identifier)
				     summary))
	     (when summary_text
	       (logger.info (fstring "Generating summary embedding for identifier {identifier}..."))
	       (setf embedding_result (genai.embed_content
				       :model (string "models/embedding-001")
				       :content summary_text
				       :task_type (string "clustering")))
	       (setf vector_blob (dot (np.array
				       (aref embedding_result (string "embedding"))
				       :dtype np.float32)
				      (tobytes)))
	       (summaries.update :pk_values identifier
				 :embedding vector_blob)
	       (logger.info (fstring "Embedding stored for identifier {identifier}."))))
	    

	    (google.api_core.exceptions.ResourceExhausted
	     (logger.warning (string "Resource exhausted during embedding of summary"))
	     )
	    ((as Exception e)
	     (logger.error (fstring "Error during embedding for identifier {identifier}: {e}")))))
	 " "
	 (comments "in production run this script with: GEMINI_API_KEY=`cat api_key.txt` uvicorn p04_host:app --port 5001")
	 #+nil (serve :host (string "127.0.0.1") :port 5001)
	 #+nil
	 (when (== __name__ (string "main"))
	   (uvicorn.run :app (string "p04_host:app")
			:host (string "127.0.0.1")
			:port 5001
			:reload False
					;:ssl_keyfile (string "privkey.pem")
					;:ssl_certfile (string "fullchain.pem")
			))
	 ))))
  )
