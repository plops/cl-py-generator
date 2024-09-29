(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

;; TODO
;; [ ] list mp4 files
;; [ ] store files in database
;; [ ] populate database with information from the video files (size, duration, resolution, framerate, bitrate)
;; [ ] allow adding more files, database migration
;; [ ] allow viewing in browser
;; [ ] gui for rating (good actor, good story, tags)
;; [ ] pre-load (if necessary)
;; [ ] whisper (convert audio to searchable transcript)
;; [ ] mediapipe (count and track people, faces, hands)
;; [ ] OCR for text in the video
;; [ ] summarization (using transcript, OCR results and possibly scene descriptions using multi-modal model like llama 3.2) 

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(progn
  (defparameter *project* "149_host_videos")
  (defparameter *idx* "01") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format (- (time.time) start_time)
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
	 
	 (db-cols `((:name identifier :type int)
		    (:name path :type str)
		    ,@(loop for e in `(actors story)
			    collect
			    `(:name ,(format nil "rating_~a" e)
			      :type int))
		    ))
	 )
    (write-source
     (format nil "~a/source01/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U python-fasthtml")

       (imports (
					;os
		 pathlib
		 ; re
		 ;markdown
		 ; uvicorn
		 sqlite_minutils.db
		 datetime
		 time))

       (imports-from (fasthtml.common *))

       " "
       (def render (video)
	 (declare (type Video video))
	 (setf identifier video.identifier)
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
       (comments "video is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
       (setf (ntuple app rt video Video)
	     (fast_app :db_file (string "data/video.db")
		       :live False	;True
		       :render render
		       ,@(loop for e in db-cols
			       appending
			       (destructuring-bind (&key name type no-show) e
				 `(,(make-keyword (string-upcase (format nil "~a" name)))
				   ,type)))

		       
		       :pk (string "identifier")
		       ))


       
       (@rt (string "/"))
       (def get (request)
	 (declare (type Request request))
	 	 
	 
	 (setf nav (Nav
		    (Ul (Li (Strong (string "Video Viewer"))))
		    (Ul	(Li (A (string "Overview")
			       :href (string "https://www.youtube.com/watch?v=ttuDW1YrkpU")))
			(Li (A (string "Details")
			       :href (string "https://github.com/plops/gemini-competition/blob/main/README.md")))
			)))
	 
	 (setf transcript (Textarea :placeholder (string "Paste YouTube transcript here")
				    :style (string "height: 300px; width=60%;")
				    :name (string "transcript"))
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
		  transcript
		  (Textarea :placeholder (string "(Optional) Add Link to original source")
				    
				    :name (string "original_source_link"))
		  model
		  (Div (Label (string "Output Language") :_for (string "output_language"))
		       (Select
			,@(loop for e in *languages*
				collect
				`(Option (string ,e)))
			:style (string "width: 100%;")
			:name (string "output_language")
			:id (string "output_language"))
			  :style (string "display: flex; align-items: center; width: 100%;"))
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
			    :style (string "display: flex; align-items: center; width: 100%;")))
		  
		  (Button (string "Summarize Transcript"))
		  :style (string "display: flex; flex-direction:column;"))
		 )
		:hx_post (string "/process_transcript")
		:hx_swap (string "afterbegin")
		:target_id (string "gen-list")))

	 (setf gen_list (Div :id (string "gen-list")))

	 (setf summaries_to_show (summaries :order_by (string "identifier DESC"))
	       )
	 (setf summaries_to_show (aref summaries_to_show (slice 0 (min 13 (len summaries_to_show)))))
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
	      (if (dot s model (startswith (string "gemini-1.5-pro"))) 
		  (setf price_input_token_usd_per_mio 1.25
			price_output_token_usd_per_mio 5.0)
		  (if (dot s model (startswith (string "gemini-1.5-flash"))) 
		      (setf price_input_token_usd_per_mio 0.075
			 price_output_token_usd_per_mio 0.3)
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
		    cost (+ (* (/ input_tokens 1_000_000)
			       price_input_token_usd_per_mio)
			    (* (/ output_tokens 1_000_000)
			       price_output_token_usd_per_mio)))
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
	   (setf title  (fstring ;"{s.summary_timestamp_start} id: {identifier} summary: {s.summary_done} timestamps: {s.timestamps_done}"
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
				    :id (fstring "pre-{identifier}")
				    )
			       :id (string "hidden-markdown")
			       
			       :style (string "display: none;"))
			  (Div
			   (NotStr html)
					;:id (fstring "pre-{identifier}-html")
			   )))
	   (setf button (Button (string "Copy")
				:onclick (fstring "copyPreContent('pre-{identifier}')")))
	   (if (== trigger (string ""))
	       
	       (return (Div
			title
		      
			pre
			button
			:id sid
			))
	       (return (Div
			title
			pre
			button
			:id sid
			:hx_post (fstring "/generations/{identifier}")
			:hx_trigger trigger
			:hx_swap (string "outerHTML")))))
	  ("Exception as e"		; NotFoundError ()
	   (return (Div
		    (fstring "id: {identifier} e: {e}")
		    (Pre text)
		    :id sid
		    :hx_post (fstring "/generations/{identifier}")
		    :hx_trigger trigger
		    :hx_swap (string "outerHTML"))))))
 
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
	 (setf words (summary.transcript.split))
	 (when (< 100_000 (len words))
	   (when (dot summary model (startswith (string "gemini-1.5-pro")))
	     (return (Div (string "Error: Transcript exceeds 20,000 words. Please shorten it or don't use the pro model.")
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
		(print (fstring "unknown exception {e}"))))
	      (time.sleep .1))
	 (print (string "row did not appear"))
	 (return -1))
       
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
	   (setf prompt (fstring3 ,(format nil "Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
~a
Example Output:
~a
Here is the real transcript. Please summarize it: 
{s.transcript}"
					      #-example "input" #-example "output"
					      #+example example-input #+example example-output-nocomments
					      )))
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
{s.transcript}"
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
		   (print (string "Error")))
		  )
		 ))

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
			     :timestamps_input_tokens 0	; response2.usage_metadata.prompt_token_count
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
       ))))
