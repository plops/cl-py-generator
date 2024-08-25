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
	(db-cols `((:name id :type int)
		   (:name model :type str)
		   (:name transcript :type str)
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
		   (:name cost :type float)
		   )))
   (write-source
     (format nil "~a/source02/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U google-generativeai python-fasthtml")

       (imports ((genai google.generativeai)
		 os
		 datetime))
       (imports-from (fasthtml.common *))

       " "
       (comments "Read the gemini api key from disk")
       (with (as (open (string "api_key.txt"))
                 f)
             (setf api_key (dot f (read) (strip))))

       (genai.configure :api_key api_key)

       " "
       (def render (summary)
	 (if summary.summary_done
	     (return (Div (Pre summary.timestamps)
			  :id sid
			  :hx_post (fstring "/generations/{id}")
			  :hx_trigger (? summary.timestamps_done
					 (string "")
					 (string "every 1s"))
			  :hx_swap (string "outerHTML")))
	     (return (Div (Pre summary.summary)
			  :id sid
			  :hx_post (fstring "/generations/{id}")
			  :hx_trigger (string "every 1s")
			  :hx_swap (string "outerHTML")))
	     ))
       
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

		       
		       :pk (string "id")
		       ))


       " "
       (def render (summary)
	 (return (Li (A summary.summary_timestamp_start
			:href (fstring "/summaries/{summary.id}")))))

       " "
       (@rt (string "/"))
       (def get ()
	 (setf nav (Nav
		    (Ul (Li (Strong (string "Transcript Summarizer"))))
		    (Ul (Li (A (string "About")
			       :href (string "#")))
			(Li (A (string "Documentation")
			       :href (string "#")))
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
  
	 (return (ntuple (Title (string "Video Transcript Summarizer"))
			 (Main nav
			       (H1 (string "Summarizer Demo"))
			       
			       form
			       gen_list
			       :cls (string "container")))))
       " "
       (comments "A pending preview keeps polling this route until we return the summary")
       (def generation_preview (id)
	 (setf sid (fstring "gen-{id}"))
	 (if (aref summaries id)
	     (do0
	      (setf summary (dot (aref summaries id) summary))
	      (return (Div (Pre summary)
			   :id sid)))
	     (return (Div (string "Generating ...")
			  
			  :id sid
			  :hx_post (fstring "/generations/{id}")
			  :hx_trigger (string "every 1s")
			  :hx_swap (string "outerHTML")))))
 
      " "
       (@app.post (string "/generations/{id}"))
       (def get (id)
	 (declare (type int id))
	 (return (generation_preview id)))

       " "
       (@rt (string "/process_transcript"))
       (def post (summary)
	 (declare (type Summary summary))
	 (setf words (summary.transcript.split))
	 (when (< 20_000 (len words))
	   (return (Div (string "Error: Transcript exceeds 20,000 words. Please shorten it.")
			:id (string "summary"))))
	 (setf summary.timestamp_summary_start (dot datetime
						     datetime
						     (now)
						     (isoformat)))
	 (setf summary.summary (string ""))
	 "global s2"
	 (setf s2 (summaries.insert summary))
	 
	 (generate_and_save s2.id)
	 
	 
	 (return (generation_preview s2.id)))

       " "
       "@threaded"
       (def generate_and_save (id)
	 (declare (type int id))
	 (print (fstring "generate_and_save id={id}"))
	 (setf s (aref summaries id))
	 (print (fstring "generate_and_save model={s.model}"))
	 (setf m (genai.GenerativeModel s.model))
	 (setf response (m.generate_content
			 (fstring "I don't want to watch the video. Create a self-contained bullet list summary: {s.transcript}")
			 :stream True))
	 
	 (for (chunk response)
	      (print chunk)
	      (setf new (dictionary :summary (+ s.summary chunk.text)))
	      (summaries.update id
				:updates new
					;:alter True
				)
	      )
	 (if response._done
	     
	     (do0
	      (setf new (dictionary :summary_done True
				    :summary (+ s.summary chunk.text)
				    :summary_input_tokens response.usage_metadata.prompt_token_count
				    :summary_output_tokens response.usage_metadata.candidates_token_count
				    :summary_timestamp_stop (dot datetime
								 datetime
								 (now)
								 (isoformat))))
	      (summaries.update
	       id
	       :updates new
					;:alter True
	       )
	      )
	     (f.write (string "Warning: Did not finish!")))
	 
	 
	 
	 )
       " "
       (serve :port 5002)))))
