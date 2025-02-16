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

;; Tests of individual steps
;; [X] Validate YouTube URL
;; [X] Bring Timestamps into HH:MM:SS format
;; [X] Deduplicate Transcript
;; [ ] Convert formatting of the summary from markdown to YouTube
;; [ ] Price estimation

(setf *features* (union *features* '(:example ;; store long example text in python (enable for production, disable for debugging)
				     :emulate ;; don't actually make the calls to gemini api (enable for debugging)
				     :dl ;; download transcript from link
				     :simple ;; use very few gui elements 
				     )))
(setf *features* (set-difference *features* '(;:example
					      :emulate
					      ;:simple
					      ;:dl
					      )))

(progn
  (defparameter *project* "143_helium_gemini")

  (load "gen04_data.lisp")
  ;; name input-price output-price context-length harm-civic
  
  (let ((iflash .1)
	(oflash .4)
	(ipro 1.25)
	(opro 5))
    (defparameter *models* `(
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
			     (:name gemini-1.5-flash :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic nil)
			     (:name gemini-1.5-pro :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic nil)
			     (:name gemini-1.0-pro :input-price .5 :output-price 1.5 :context-length 128_000 :harm-civic nil)
			     )))
  (defparameter *languages* `(en de fr ch nl pt cz it jp ar))
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
		    (:name host :type str)
		    (:name original_source_link :type str)
		    (:name include_comments :type bool)
		    (:name include_timestamps :type bool)
		    (:name include_glossary :type bool)
		    (:name output_language :type str)
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
		    (:name cost :type float)
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
	 (comments "pip install -U google-generativeai python-fasthtml markdown")
	 #+dl (comments "micromamba install python-fasthtml markdown yt-dlp; pip install  webvtt-py")
	 (imports ((genai google.generativeai)
					;google.generativeai.types.answer_types
		   os
		   google.api_core.exceptions
		   re
		   markdown
					; uvicorn
		   sqlite_minutils.db
		   datetime
		   difflib
		   #+dl subprocess
		   #+dl webvtt
		   time))

	 #+nil
	 (imports-from (threading Thread)
		       (functools wraps))

	 #-emulate (imports-from (google.generativeai.types HarmCategory HarmBlockThreshold))

	 (imports-from (fasthtml.common *)
		       (s01_validate_youtube_url *)
		       (s02_parse_vtt_file *))

	 (do0
	  (comments "Read the demonstration transcript and corresponding summary from disk")
	  ,@(loop for e in `((:var g_example_input :data ,*example-input* :fn example_input.txt)
			     (:var g_example_output :data ,*example-output* :fn example_output.txt)
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
	 
	 " "
	 (comments "Read the gemini api key from disk")
	 (with (as (open (string "api_key.txt"))
                   f)
	       (setf api_key (dot f (read) (strip))))

	 (genai.configure :api_key api_key)

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
				:hx_post (fstring "/generations/{identifier}")
				:hx_trigger (string "")
				:hx_swap (string "outerHTML"))))
	     (summary.summary_done
	      (return (Div		;(Pre summary.summary)
		       (NotStr (markdown.markdown summary.summary))
		       :id sid
		       :hx_post (fstring "/generations/{identifier}")
		       :hx_trigger (? summary.timestamps_done
				      (string "")	
				      (string #+emulate "" #-emulate "every 1s"))
		       :hx_swap (string "outerHTML"))))
	     (t
	      (return (Div		;(Pre summary.summary)
		       (NotStr (markdown.markdown summary.summary))
		       :id sid
		       :hx_post (fstring "/generations/{identifier}")
		       :hx_trigger (string #+emulate "" #-emulate "every 1s")
		       :hx_swap (string "outerHTML")))))
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


	 " "
	 #+nil(def render (summary)
		(declare (type Summary summary))
		(return (Li 
			 (A summary.summary_timestamp_start
			    :href (fstring "/summaries/{summary.identifier}")))))



	 (setf documentation #+nil (string3 "###### To use the YouTube summarizer:

1. **Copy the YouTube video link.**
2. **Paste the link into the provided input field.**
3. **Alternatively (for desktop browsers):** If you're on the YouTube video page, you can copy the video's title, description, transcript, and any visible comments, then paste them into the input field.
4. **Click the 'Summarize' button.** The summary with timestamps will be generated.

")
			     #+simple (string3 "###### To use the YouTube summarizer:

1. **Copy the YouTube video link.**
2. **Paste the link into the provided input field.**
3. **Click the 'Summarize' button.** The summary with timestamps will be generated.

")
			     
			     #-dl (string3 "###### **Prepare the Input Text from YouTube:**
 * **Scroll down a bit** on the video page to ensure some of the top comments have loaded.
 * Click on the \"Show Transcript\" button below the video.
 * **Scroll to the bottom** in the transcript sub-window.
 * **Start selecting the text from the bottom of the transcript sub-window and drag your cursor upwards, including the video title at the top.** This will select the title, description, comments (that have loaded), and the entire transcript.
 * **Tip:** Summaries are often better if you include the video title, the video description, and relevant comments along with the transcript.

###### **Paste the Text into the Web Interface:**
 * Paste the copied text (title, description, transcript, and optional comments) into the text area provided below.
 * Select your desired model from the dropdown menu (Gemini Pro is recommended for accurate timestamps).
 * Click the \"Summarize Transcript\" button.

###### **View the Summary:**
 * The application will process your input and display a continuously updating preview of the summary. 
 * Once complete, the final summary with timestamps will be displayed, along with an option to copy the text.
 * You can then paste this summarized text into a YouTube comment.
"))

	 
	 
	 
	 #+dl
	 (def get_transcript (url)
	   (comments "Call yt-dlp to download the subtitles. Modifies the timestamp to have second granularity. Returns a single string")

	   (setf youtube_id (validate_youtube_url url))
	   (unless youtube_id
	     (return (string "URL couldn't be validated")))
	   
	   (setf sub_file (string "/dev/shm/o"))
	   (setf sub_file_ (string "/dev/shm/o.en.vtt"))
	   (setf cmds (list (string "yt-dlp")
			    (string "--skip-download")
			    (string "--write-auto-subs")
			    (string "--write-subs")
					;(string "--cookies")
					;(string "yt_cookies.txt")
			    (string "--cookies-from-browser")
			    (string "firefox")
			    (string "--sub-lang")
			    (string "en")
			    (string "-o")
			    sub_file
			    youtube_id))
	   (print (dot (string " ")
		       (join cmds)))
	   (subprocess.run cmds)
	   
	   (try
	    (do0
	     (setf ostr (parse_vtt_file sub_file_))
	     
	     (os.remove sub_file_))
	    
	    (FileNotFoundError
	     (print (string "Error: Subtitle file not found")))
	    ("Exception as e"
	     (print (string "line 1639 Error: problem when processing subtitle file"))))
	   (return ostr)
	   )
	 " "
	 (setf documentation_html
	       (markdown.markdown documentation))
	 (@rt (string "/"))
	 (def get (request)
	   (declare (type Request request))
	   ;; how to format markdown: https://isaac-flath.github.io/website/posts/boots/FasthtmlTutorial.html
	   
	   (print request.client.host)
	   (setf nav (Nav
		      (Ul (Li (Strong (string "Transcript Summarizer"))))
		      (Ul #+nil (Li (A (string "About")
				       :href (string "#")))
			  (Li (A (string "Demo Video")
				 :href (string "https://www.youtube.com/watch?v=ttuDW1YrkpU")))
			  (Li (A (string "Documentation")
				 :href (string "https://github.com/plops/gemini-competition/blob/main/README.md")))
			  )))
	   
	   (setf transcript (Textarea :placeholder (string "(Optional) Paste YouTube transcript here")
				      :style (string ;#+simple "height: 300px; width=60%; display: none;"
						     "height: 300px; width=60%;")
				      :name (string "transcript")))
	   (setf  
	    model (Div (Select
			,@(loop for e in *models*
				collect
				(destructuring-bind (&key name input-price output-price context-length harm-civic) e 
				  `(Option (string ,name))))
			:style (string "width: 100%;")
			:name (string "model"))
		       :style (string "display: flex; align-items: center; width: 100%;")))

	   
	   (setf form
		 (Form
                  (Group
		   (Div 
		     
		    (Textarea :placeholder (string "Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)")
			      
			      :name (string "original_source_link"))
		    transcript
		    model
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
		    
		    ,@(loop for (e f default) in `((include_comments "Include User Comments" False)
						   (include_timestamps "Include Timestamps" True)
						   (include_glossary "Include Glossary" False)
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
		  :hx_post (string "/process_transcript")
		  :hx_swap (string "afterbegin")
		  :target_id (string "gen-list")))

	   (setf gen_list (Div :id (string "gen-list")))

	   (setf summaries_to_show (summaries :order_by (string "identifier DESC"))
		 )
	   (setf summaries_to_show (aref summaries_to_show (slice 0 (min 3 (len summaries_to_show)))))
	   (setf summary_list (Ul *summaries_to_show
				  :id (string "summaries")))
	   (return (ntuple (Title (string "Video Transcript Summarizer"))
			   (Main nav
					;(H1 (string "Summarizer Demo"))
				 (NotStr documentation_html)
				 form
				 gen_list
				 summary_list 
				 (Script (string3 "function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}"))
				 :cls (string "container")))))
	 " "
	 (comments "A pending preview keeps polling this route until we return the summary")
	 (def generation_preview (identifier)
	   (setf sid (fstring "gen-{identifier}"))
	   (setf text (string "Generating ...")
		 trigger (string #+emulate "" #-emulate "every 1s"))
	   (try
	    (do0
	     (setf s (aref summaries identifier))
	     (cond
	       
	       (s.timestamps_done
		(comments "this is for <= 128k tokens")
		(if (or (dot s model (startswith (string "gemini-1.5-pro")))
			(dot s model (startswith (string "gemini-2.0-pro")))) 
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
		(if (in (string "flash") (dot s model)) ;; flash price doesn't depend on number of tokens
		    (setf cost (+ price_input_token_usd_per_mio
				  price_output_token_usd_per_mio))
		    (setf cost (+ (* (/ input_tokens 1_000_000)
				     price_input_token_usd_per_mio)
				  (* (/ output_tokens 1_000_000)
				     price_output_token_usd_per_mio))))
		(summaries.update :pk_values identifier
				  :cost cost)
		(if (< cost .02)
		    (setf cost_str (fstring "${cost:.4f}"))
		    (setf cost_str (fstring "${cost:.2f}")))
		(setf text (fstring3 "{s.timestamped_summary_in_youtube_format}

I used {s.model} on rocketrecap dot com to summarize the transcript.
Cost (if I didn't use the free tier): {cost_str}
Input tokens: {input_tokens}
Output tokens: {output_tokens}")

		      
		      trigger (string ""))


		)
	       (s.summary_done
		(setf text s.summary))
	       ((< 0 (len s.summary))
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
				     (if (eq name 'original_source_link)
					 `(A
					   (fstring "{s.original_source_link}")
					   :target (string "_blank")
					   :href (fstring "{s.original_source_link}")
					   :id (string "source-link"))
					 `(P
					   (B (string ,(format nil "~a:" name)))
					   (Span (fstring ,(format nil "{s.~a}"
								   (string-downcase name))))))
				     #+nil
				     (format nil "~a: {s.~a}" name (string-downcase name))))))

			:cls (string "summary-details")
			
			#+nil (P
			       (B (string "Identifier:"))
			       (Span 558
				     :id (string "identifier")
				     ))
			#+nil (P (B (string "Model:"))
				 (Span (string "gemini2.0")
				       :id (string "model")
				       ))))
	     (setf summary_container
		   (Div summary_details
			:cls (string "summary-container")
			
			))
	     (setf title summary_container
		   #+nil (fstring ;"{s.summary_timestamp_start} id: {identifier} summary: {s.summary_done} timestamps: {s.timestamps_done}"
				  ,(format nil "~{~a~^ ~}"
					   (remove-if #'null
						      (loop for e in db-cols
							    collect
							    (destructuring-bind (&key name type no-show) e
							      (unless no-show
								(format nil "~a: {s.~a}" name (string-downcase name)))))))
				  ))

	     (setf html (markdown.markdown s.summary))
					;(print (fstring "md: {html}"))
	     (setf pre (Div (Div (Pre text
				      :id (fstring "pre-{identifier}"))
				 :id (string "hidden-markdown")
				 :style( string "display: none;"))
			    (Div
			     (NotStr html)
					;:id (fstring "pre-{identifier}-html")
			     )))
	     (setf button (Button (string "Copy Summary")
				  :onclick (fstring "copyPreContent('pre-{identifier}')")))

	     (do0
	      (setf prompt_text (get_prompt s)
		    prompt_pre (Pre prompt_text :id (fstring "prompt-pre-{identifier}")
						:style (string "display: none;"))
		    prompt_button (Button (string "Copy Prompt")
					  :onclick (fstring "copyPreContent('prompt-pre-{identifier}')"))))
	     
	     (if (== trigger (string ""))
		 (return (Div
			  title
			  pre
			  prompt_pre
			  button
			  prompt_button
			  :id sid
			  ))
		 (return (Div
			  title
			  pre
			  prompt_pre
			  button
			  prompt_button
			  :id sid
			  :hx_post (fstring "/generations/{identifier}")
			  :hx_trigger trigger
			  :hx_swap (string "outerHTML")))))
	    ("Exception as e"		; NotFoundError ()
	     (return (Div
		      (fstring "line 1897 id: {identifier} e: {e}")
		      (Pre text)
		      :id sid
		      :hx_post (fstring "/generations/{identifier}")
		      :hx_trigger trigger
		      :hx_swap (string "outerHTML"))))))

	 " "
	 (def parse_vtt_time (time_str)
	   (rstring3 "Parses a VTT timestamp string (HH:MM:SS) into a datetime object.")
	   (return (datetime.datetime.strptime time_str (string "%H:%M:%S"))))
	 " "
	 (def calculate_similarity (text1 text2)
	   (rstring3 "Calculates the similarity ratio between two strings using SequenceMatcher.")
	   (return (dot difflib
			(SequenceMatcher None text1 text2)
			(ratio))))

	 " "
	 (def deduplicate_transcript (vtt_content &key (time_window_seconds 5)
						  (similarity_threshold .7))
	   (rstring3 "
    Deduplicates a VTT transcript string.

    Args:
        vtt_content: The VTT transcript as a single string.
        time_window_seconds: The maximum time difference (in seconds) between lines
                             to be considered for deduplication.
        similarity_threshold:  The minimum similarity ratio (0.0 to 1.0) for two
                              lines to be considered near-duplicates.

    Returns:
        A deduplicated VTT transcript string.
")

	   (setf lines (dot vtt_content (strip) (split (string "\\n"))))
	   (do0
	    (setf 
	     deduplicated_lines (list)
	     )
	    (setf pattern (rstring3 "(\\d{2}:\\d{2}:\\d{2})\\s+(.*)"))
	    ,(lprint :vars `((len lines)))
	    (for (line lines)
		 ,(lprint :vars `(line))
		 (setf match (re.match pattern
				       line))
		 (unless match
		   ,(lprint :msg "no match" :vars `(match))
		   (comments "Keep non-timestamped lines")
		   (deduplicated_lines.append line)
		   continue))
	    (setf (ntuple current_time_str current_text)
		  (match.groups)
		  current_time (parse_vtt_time current_time_str)
		  is_duplicate False)
	    ,(lprint :msg "1950" :vars `(current_time current_text ))
	    (setf ll ("list" (range (- (len deduplicated_lines) 1)
				    -1 -1)))
	    ,(lprint :msg "1951" :vars `((- (len deduplicated_lines) 1)
					 ll))
	    (for (i (range (- (len deduplicated_lines) 1)
			   -1 -1))
		 ,(lprint :vars `(i prev_line prev_match))
		 (comments "Iterate backwards to efficiently check recent lines")
		 (setf prev_line (aref deduplicated_lines i)
		       prev_match (re.match pattern prev_line))
		 
		 (unless prev_match
		   ,(lprint :msg "ERROR" :vars `(prev_match))
		   continue)
		 (setf (ntuple prev_time_str prev_text)
		       (prev_match.groups)
		       prev_time (parse_vtt_time prev_time_str)
		       time_diff (dot (- current_time
					 prev_time)
				      (total_seconds)))
		 ,(lprint :vars `(time_diff prev_text))
		 (when (< time_window_seconds
			  time_diff)
		   (comments "Past the time window, stop checking")
		   break)
		 (when (<= 0 time_diff)
		   (comments "Ensure time is progressing")
		   (setf similarity (calculate_similarity prev_text current_text))
		   ,(lprint :vars `(similarity prev_text current_text))
		   (when (<= similarity_threshold
			     similarity)
		     (setf is_duplicate True)
		     break))
		 )
	    (unless is_duplicate
	      (deduplicated_lines.append line)
	      #+nil (setf (aref last_times current_text)
			  current_time)))
	   (setf dedup (dot (string "\\n")
			    (join lines #+nil
					deduplicated_lines)))
	   ,(lprint :vars `((len vtt_content)
			    (len dedup)))
	   (return dedup)
	   )


	 #+nil (do0
		(setf example (rstring3 *duplication-example-input*))
		(print (deduplicate_transcript example)))
	 " "
	 (@app.post (string "/generations/{identifier}"))
	 (def get (identifier)
	   (declare (type int identifier))
	   (return (generation_preview identifier)))

	 " "
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
	   
	   
	   (return (generation_preview s2.identifier)))

	 " "
	 (def wait_until_row_exists (identifier)
	   (for (i (range 10))
		(try
		 (do0 (setf s (aref summaries identifier))
		      (return s))
		 (sqlite_minutils.db.NotFoundError
		  (print (string "entry not found")))
		 ("Exception as e"
		  (print (fstring "line 1953 unknown exception {e}"))))
		(time.sleep .1))
	   (print (string "row did not appear"))
	   (return -1))
	 
	 
	 
	 " "
	 (def get_prompt (summary)
	   (declare (type Summary summary)
		    (values str))
	   (rstring3 "Generate prompt from a given Summary object. It will use the contained transcript.")
	   (setf prompt (fstring3 "Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
{g_example_input}
Example Output:
{g_example_output}
Here is the real transcript. Please summarize it: 
{(summary.transcript)}"))

	   (return prompt)
	   )
	 " "
	 "@threaded"
	 (def generate_and_save (identifier)
	   (declare (type int identifier))
	   (print (fstring "generate_and_save id={identifier}"))
	   (setf s (wait_until_row_exists identifier))
	   (print (fstring "generate_and_save model={s.model}"))
	   #-emulate
	   (do0
	    (setf m (genai.GenerativeModel s.model))
	    (setf safety (dict (HarmCategory.HARM_CATEGORY_HATE_SPEECH HarmBlockThreshold.BLOCK_NONE)
			       (HarmCategory.HARM_CATEGORY_HARASSMENT HarmBlockThreshold.BLOCK_NONE)
			       (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT HarmBlockThreshold.BLOCK_NONE)
			       (HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT HarmBlockThreshold.BLOCK_NONE)
					;(HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY HarmBlockThreshold.BLOCK_NONE)
			       )))
	   (try
	    (do0
	     (do0 
	      (setf prompt (get_prompt s))
	      #+emulate
	      (do0
	       (with (as (open (string "/dev/shm/prompt.txt")
			       (string "w"))
			 fprompt)
		     (fprompt.write prompt))
	       (summaries.update :pk_values identifier
				 :summary (string "emulate")))
	      #-emulate
	      (do0
	       (setf response (m.generate_content
			       #+comments (fstring3 ,(format nil "Below, I will provide input for an example video (comprising of title, description, optional viewer comments, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. Also, incorporate information from the viewer comments **if they clarify points made in the video, answer questions raised, or correct factual errors**. When including information sourced from the viewer comments, please indicate this by adding \"[From <user>'s Comments]\" at the end of the bullet point. Note that while viewer comments appear earlier in the text than the transcript they are in fact recorded at a later time. Therefore, if viewer comments repeat information from the transcript, they should not appear in the summary.

Example Input: 
~a
Example Output:
~a
Here is the real transcript. Please summarize it: 
{deduplicate_transcript(s.transcript)}"
							     #-example "input" #-example "output"
							     #+example example-input #+example example-output-nocomments
							     ))
			       prompt
			       :safety_settings safety
			       :stream True))

	       
	       
	       (for (chunk response)
		    (try
		     (do0
		      (print (fstring "add text to id={identifier}: {chunk.text}"))
		      
		      (summaries.update :pk_values identifier
					:summary (+ (dot (aref summaries identifier)
							 summary)
						    chunk.text)))
		     (ValueError ()
				 (summaries.update :pk_values identifier
						   :summary (+ (dot (aref summaries identifier)
								    summary)
							       (string "\\nError: value error")))
				 (print (string "Value Error ")))
		     ("Exception as e"
		      (summaries.update :pk_values identifier
					:summary (+ (dot (aref summaries identifier)
							 summary)
						    (fstring "\\nError: {str(e)}")))
		      (print (string "line 2049 Error")))
		     )
		    )))

	     (summaries.update :pk_values identifier
			       :summary_done True
			       :summary_input_tokens #+emulate 0 #-emulate response.usage_metadata.prompt_token_count
			       :summary_output_tokens #+emulate 0 #-emulate response.usage_metadata.candidates_token_count
			       :summary_timestamp_end (dot datetime
							   datetime
							   (now)
							   (isoformat))

			       :timestamps (string "") 
			       :timestamps_timestamp_start (dot datetime
								datetime
								(now)
								(isoformat))))
	    (google.api_core.exceptions.ResourceExhausted
	     (summaries.update :pk_values identifier
			       :summary_done False
			       
			       :summary (+ (dot (aref summaries identifier)
						summary)
					   (string "\\nError: resource exhausted"))
			       :summary_timestamp_end (dot datetime
							   datetime
							   (now)
							   (isoformat))

			       :timestamps (string "") 
			       :timestamps_timestamp_start (dot datetime
								datetime
								(now)
								(isoformat)))
	     return))

	   
	   (try
	    (do0
	     #+nil
	     (do0
	      (print (string "generate timestamps"))
	      (setf s (dot (aref summaries identifier)))
	      (setf response2 (m.generate_content
			       (fstring "Add a title to the summary and add a starting (not stopping) timestamp to each bullet point in the following summary: {s.summary}\nThe full transcript is: {s.transcript}")
			       :safety_settings safety
			       :stream True))

	      
	      
	      (for (chunk response2)
		   (try
		    (do0
		     (print (fstring "add timestamped text to id={identifier}: {chunk.text}"))
		     
		     (summaries.update :pk_values identifier
				       :timestamps (+ (dot (aref summaries identifier)
							   timestamps)
						      chunk.text)))
		    (ValueError ()
				(print (string "Value Error"))))
		   )
	      (setf text (dot (aref summaries identifier)
			      timestamps)))

	     (setf text (dot (aref summaries identifier)
			     summary))
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
	     (summaries.update :pk_values identifier
			       :timestamps_done False
			       
			       :timestamped_summary_in_youtube_format (fstring "resource exhausted")
			       :timestamps_timestamp_end (dot datetime
							      datetime
							      (now)
							      (isoformat)))
	     return))
	   
	   )
	 " "


					;(serve :host (string "localhost") :port 5001)
	 (serve :host (string "0.0.0.0") :port 5001)
	 #+nil (when (== __name__ (string "main"))
		 (uvicorn.run :app (string "p01_host:app")
			      :host (string "0.0.0.0")
			      :port 5001
			      :reload True
					;:ssl_keyfile (string "privkey.pem")
					;:ssl_certfile (string "fullchain.pem")
			      ))
	 )))))
