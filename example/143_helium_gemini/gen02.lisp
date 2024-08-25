(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
 (defparameter *project* "143_helium_gemini")
 (defparameter *idx* "01")
 (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
 (defparameter *day-names*
               '("Monday" "Tuesday" "Wednesday"
                          "Thursday" "Friday" "Saturday"
                          "Sunday"))
 (defun lprint (&key msg vars)
   `(do0 ;when args.verbose
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
     (format nil "~a/source02/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U google-generativeai python-fasthtml")

       (imports ((genai google.generativeai)
		 ;google.generativeai.types.answer_types
					;os
		 re
		 sqlite_minutils.db
		 datetime
		 time))
       (imports-from (fasthtml.common *))

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
	    (return (Div (Pre summary.summary)
			 :id sid
			 :hx_post (fstring "/generations/{identifier}")
			 :hx_trigger (? summary.timestamps_done
					(string "")
					(string "every 1s"))
			 :hx_swap (string "outerHTML"))))
	   (t
	    (return (Div (Pre summary.summary)
			  :id sid
			  :hx_post (fstring "/generations/{identifier}")
			  :hx_trigger (string "every 1s")
			  :hx_swap (string "outerHTML")))))
	 )
       
       " "
       (comments "open website")
       (comments "summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
       (setf (ntuple app rt summaries Summary)
	     (fast_app :db_file (string "data/summaries.db")
		       :live True
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


       " "
       
       (@rt (string "/"))
       (def get (request)
	 (declare (type Request request))
	
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
		 (Button (string "Send Transcript")))
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
			       (H1 (string "Summarizer Demo"))
			       
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

I used {s.model} to summarize the transcript.
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

	   (return (Div
		    (fstring "{s.summary_timestamp_start} id: {identifier} summary: {s.summary_done} timestamps: {s.timestamps_done}")
		    (Pre text :id (fstring "pre-{identifier}"))
		    (Button (string "Copy")
			    :onclick (fstring "copyPreContent('pre-{identifier}')"))
		    :id sid
		    :hx_post (fstring "/generations/{identifier}")
		    :hx_trigger trigger
		    :hx_swap (string "outerHTML"))))
	  ("Exception as e"		; NotFoundError ()
	   (return (Div
		    (fstring "id: {identifier}")
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
	 (when (< 20_000 (len words))
	   (return (Div (string "Error: Transcript exceeds 20,000 words. Please shorten it.")
			:id (string "summary"))))
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
	 (do0
	  (setf response (m.generate_content
			  (fstring "I don't want to watch the video. Create a self-contained bullet list summary: {s.transcript}")
			  :stream True))

	  
	  
	  (for (chunk response)
	       (try
		(do0
		 (print (fstring "add text to id={identifier}: {chunk.text}"))
		 
		 (summaries.update :pk_values identifier
				   :summary (+ (dot (aref summaries identifier)
						    summary)
					       chunk.text)
					;new
					;:alter True
				   ))
		(ValueError ()
			    (print (string "Value Error "))))
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

	 (do0
	  (print (string "generate timestamps"))
	  (setf s (dot (aref summaries identifier)
		       ))
	  (setf response2 (m.generate_content
			   (fstring "Add a title to the summary and add a starting (not stopping) timestamp to each bullet point in the following summary: {s.summary}\nThe full transcript is: {s.transcript}")
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
			  timestamps))

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

				

	  
	  
	  (summaries.update :pk_values identifier
			    :timestamps_done True
			    :timestamped_summary_in_youtube_format text
			    :timestamps_input_tokens response2.usage_metadata.prompt_token_count
			    :timestamps_output_tokens response2.usage_metadata.candidates_token_count
			    :timestamps_timestamp_end (dot datetime
							   datetime
							   (now)
							   (isoformat))))
	 
	 )
       " "
       (serve :port 5002)))))
