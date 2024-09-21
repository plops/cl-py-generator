(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

;; TODO:
;; [ ] add choice for output language
;; [ ] show spinner
;; [ ] allow to store the original youtube link
;; [ ] optional timestamps
;; [ ] optional user comments
;; [ ] optional glossary
;; [ ] find better examples for comments, or glossary
;; [ ] communicate gemini's evaluation of the content (e.g. harassment) to the user
;; [ ] generate transcript from audio channel
;; [ ] get transcript from link


(setf *features* (union *features* '(:example)))
(setf *features* (set-difference *features* '(;:example
					      )))

(progn
  (defparameter *project* "144_fasthtml")
  (defparameter *idx* "02") 
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
	 (example-output  "out")
	 (example-input "in")
	 (db-cols `((:name identifier :type int)
		    (:name model :type str)
		    (:name transcript :type str)
		    (:name host :type str)
		    ,@(loop for e in `(summary timestamps)
			    appending
			    `((:name ,e :type str)
			      ,@(loop for (f f-type) in `((done bool)
							  (input_tokens int)
							  (output_tokens int)
							  (timestamp_start str)
							  (timestamp_end str))
				      collect
				      `(:name ,(format nil "~a_~a" e f) :type ,f-type))))
		    (:name timestamped_summary_in_youtube_format :type str)
		    (:name cost :type float)
		    )))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U google-generativeai python-fasthtml markdown")

       (imports (;(genai google.generativeai)
					;google.generativeai.types.answer_types
					;os
		 ;google.api_core.exceptions
		 re
		 markdown
		 uvicorn
		 sqlite_minutils.db
		 datetime
		 time))

       (imports-from (google.generativeai.types HarmCategory HarmBlockThreshold))

       (imports-from (fasthtml.common *))

       #+nil
       (do0
	" "
	(comments "Read the gemini api key from disk")
	(with (as (open (string "api_key.txt"))
                  f)
              (setf api_key (dot f (read) (strip))))

	(genai.configure :api_key api_key))

       " "
       (def render (summary)
	 (declare (type Summary summary))
	 (setf identifier summary.identifier)
	 (setf sid (fstring "gen-{identifier}"))
	 (cond
	   (summary.timestamps_done
	    (return 
		    (generation_preview identifier)
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
				    (string "" ; "every 1s"
					    ))
		     :hx_swap (string "outerHTML"))))
	   (t
	    (return (Div		;(Pre summary.summary)
		     (NotStr (markdown.markdown summary.summary))
		     :id sid
		     :hx_post (fstring "/generations/{identifier}")
		     :hx_trigger (string "" ; "every 1s"
					 )
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
			       (destructuring-bind (&key name type) e
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



       (setf documentation (string3 "###### **Prepare the Input Text from YouTube:**
"))

       
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
	 
	 (setf transcript (Textarea :placeholder (string "Paste YouTube transcript here")
				    :name (string "transcript"))
	       model (Select (Option (string "gemini-1.5-flash-latest"))
			     (Option (string "gemini-1.5-pro-exp-0801"))
			     
			     :name (string "model")))
	 (setf form
	       (Form
                (Group
		 transcript
		 model
		 (Button (string "Summarize Transcript")))
		:hx_post (string "/process_transcript")
		:hx_swap (string "afterbegin")
		:target_id (string "gen-list")))

	 (setf gen_list (Div :id (string "gen-list")))

	 (setf summaries_to_show (summaries :order_by (string "identifier DESC"))
	       )
	 (setf summaries_to_show summaries_to_show #+nil (aref summaries_to_show (slice 0 (min 3 (len summaries_to_show)))))
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
	       trigger (string "every 1s"))
	 (try
	  (do0
	   (setf s (aref summaries identifier))
	   (cond
	     
	     (s.timestamps_done
	      (comments "this is for <= 128k tokens")
	      (if (== s.model (string "gemini-1.5-pro-exp-0801"))
		  (setf price_input_token_usd_per_mio 3.5
			price_output_token_usd_per_mio 10.5)
		  (setf price_input_token_usd_per_mio 0.075
			price_output_token_usd_per_mio 0.3)
		  )
	      (setf input_tokens (+ s.summary_input_tokens
				    s.timestamps_input_tokens)
		    output_tokens (+ s.summary_output_tokens
				     s.timestamps_output_tokens)
		    cost (+ (* (/ input_tokens 1_000_000)
			       price_input_token_usd_per_mio)
			    (* (/ output_tokens 1_000_000)
			       price_output_token_usd_per_mio)))

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

	   (setf title  (fstring "{s.summary_timestamp_start} id: {identifier} summary: {s.summary_done} timestamps: {s.timestamps_done}"))

	   ;(setf summary_text s.summary)
	   (setf summary_text (dot s summary (aref (split (string "\\n")) 0)))
	   
	   (setf html (markdown.markdown (fstring "{s.summary_timestamp_start[:16]} {summary_text}")
					 ))
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
			;title
		      
			pre
			;button
			:id sid
			))
	       (return (Div
			;title
			pre
			;button
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
	   (when (== summary.model (string "gemini-1.5-pro-exp-0801"))
	     (return (Div (string "Error: Transcript exceeds 20,000 words. Please shorten it or don't use the pro model.")
			  :id (string "summary")))))
	 (setf summary.host request.client.host)
	 (setf summary.summary_timestamp_start (dot datetime
						    datetime
						    (now)
						    (isoformat)))
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
		(print (fstring "entry not found"))))
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
	 (setf m (genai.GenerativeModel s.model))
	 (setf safety (dict (HarmCategory.HARM_CATEGORY_HATE_SPEECH HarmBlockThreshold.BLOCK_NONE)
			    (HarmCategory.HARM_CATEGORY_HARASSMENT HarmBlockThreshold.BLOCK_NONE)
			    (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT HarmBlockThreshold.BLOCK_NONE)
			    (HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT HarmBlockThreshold.BLOCK_NONE)))
	 (try
	  (do0
	   (setf response (m.generate_content
			   (fstring3 ,(format nil "Below, I will provide input for an example video (comprising of title, description, optional viewer comments, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. Also, incorporate information from the viewer comments **if they clarify points made in the video, answer questions raised, or correct factual errors**. When including information sourced from the viewer comments, please indicate this by adding \"[From <user>'s Comments]\" at the end of the bullet point. Note that while viewer comments appear earlier in the text than the transcript they are in fact recorded at a later time. Therefore, if viewer comments repeat information from the transcript, they should not appear in the summary.

Example Input: 
~a
Example Output:
~a
Here is the real transcript. Please summarize it: 
{s.transcript}"
					#-example "input" #-example "output"
					      #+example example-input #+example example-output
					      ))
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
		)

	   (summaries.update :pk_values identifier
			     :summary_done True
			    
			     :summary_input_tokens response.usage_metadata.prompt_token_count
			     :summary_output_tokens response.usage_metadata.candidates_token_count
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
	   (comments "adapt the markdown to youtube formatting")
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


	   (comments "find any text that looks like a url and replace the . with -dot-")

	  
	   ;; text = re.sub(r"((?:https?://)?(?:www\.)?[^\s]+)\.((?:com|org|de|us|gov|net|edu|info|io|co\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr))", r"\1-dot-\2", text)

	   (setf text (re.sub (rstring3 "((?:https?://)?(?:www\\.)?[^\\s]+)\\.((?:com|org|de|us|gov|net|edu|info|io|co\\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr))")
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
