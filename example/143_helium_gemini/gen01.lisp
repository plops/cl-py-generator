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

 (let* ((notebook-name "host"))
   (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U google-generativeai")
       (comments "https://docs.fastht.ml/tutorials/by_example.html#full-example-2---image-generation-app")
					; M-S-Enter in alive to execute top-level expression

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
       (setf (ntuple app rt summaries summary)
	     (fast_app (string "summaries.db")
		       :live True
		       :render render
		       :id int
		       :model str
		       :summary str
		       :summary_done bool
		       :summary_input_tokens int
		       :summary_output_tokens int
		       :timestamps str
		       :timestamps_done bool
		       :timestamps_input_tokens int
		       :timestamps_output_tokens int
		       :cost float
		       :pk (string "id")
		       ))

       (comments "create a folder with current datetime gens_<datetime>/")
       (setf generations (list)
	     dt_now (dot datetime
			 datetime
			 (now)
			 (strftime (string "%Y%m%d_%H%M%S")))
	     folder (fstring "gens_{dt_now}/")
	     )
       (os.makedirs folder :exist_ok True)

       
       " "
       (@rt (string "/"))
       (def get ()
	 (setf transcript (Textarea :placeholder (string "Paste YouTube transcript here")
				    :name (string "transcript"))
	       model (Select (Option (string "gemini-1.5-pro-exp-0801"))
			     (Option (string "gemini-1.5-flash-latest"))
			     :name (string "model")))
	 (setf form (Form
                     (Group
		      transcript
		      model
		      (Button (string "Send Transcript")))
		     :hx_post (string "/process_transcript")
		     :hx_swap (string "afterbegin")
		     :target_id (string "gen-list")))

	 (setf gen_list (Div :id (string "gen-list")))
  
	 (return (ntuple (Title (string "Video Transcript Summarizer"))
			 (Main (H1 (string "Summarizer Demo"))
			       form
			       gen_list
			       :cls (string "container")
			       #+nil (Card (Div :id (string "summary"))
					   :header frm)))))
       " "
       (comments "A pending preview keeps polling this route until we return the summary")
       (def generation_preview (id)
	 (setf sid (fstring "gen-{id}"))
	 (setf pre_filename (fstring "{folder}/{id}.pre"))
	 (setf filename (fstring "{folder}/{id}.md"))
	 (if (os.path.exists filename)
	     (do0
	      (comments "Load potentially partial response from the file")
	      (with (as (open filename)
			f)
		    (setf summary (f.read)))
	      (return (Div (Pre summary)
			   :id sid)))
	     (if (os.path.exists pre_filename)
		 (do0
		  (with (as (open pre_filename)
			f)
		    (setf summary_pre (f.read)))
		  (return (Div (Pre
				summary_pre
				)
			       :id sid
			       :hx_post (fstring "/generations/{id}")
			       :hx_trigger (string "every 1s")
			       :hx_swap (string "outerHTML"))))
		 (return (Div (string "Generating ...")
			      
			      :id sid
			      :hx_post (fstring "/generations/{id}")
			      :hx_trigger (string "every 1s")
			      :hx_swap (string "outerHTML"))))))

       " "
       (@app.post (string "/generations/{id}"))
       (def get (id)
	 (declare (type int id))
	 (return (generation_preview id)))

       " "
       (@rt (string "/process_transcript"))
       (def post (transcript model)
	 (declare (type str transcript)
		  (type str model))
	 (setf words (transcript.split))
	 (when (< 20_000 (len words))
	   (return (Div (string "Error: Transcript exceeds 20,000 words. Please shorten it.")
			:id (string "summary"))))
	 (setf id (len generations))
	 (generate_and_save transcript id model)
	 (generations.append transcript)
	 
	 (return (generation_preview id)))

       " "
       "@threaded"
       (def generate_and_save (prompt id model)
	 (setf m (genai.GenerativeModel model))
	 (setf response (m.generate_content
			 (fstring "I don't want to watch the video. Create a self-contained bullet list summary: {prompt}")
			 :stream True))
	 (comments "consecutively append output to {folder}/{id}.pre and finally move from .pre to .md file")
	 (setf pre_file (fstring "{folder}/{id}.pre"))
	 (setf md_file (fstring "{folder}/{id}.md"))
	 (with (as (open pre_file
			 (string "w"))
		   f)
	       (for (chunk response)
		    (print chunk)
		    (f.write chunk.text))
	       (if response._done
		 (do0
		  (f.write (fstring "\\nSummarized with {model}"))
		  (f.write (fstring "\\nInput tokens: {response.usage_metadata.prompt_token_count}"))
		  (f.write (fstring "\\nOutput tokens: {response.usage_metadata.candidates_token_count}"))
		  )
		 (f.write (string "Warning: Did not finish!")))
	       )
	 (os.rename pre_file md_file)
	 
	 )
       " "
       (serve :port 5002)))))
