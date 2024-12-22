(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)


(setf *features* (union *features* '(:example
				     )))
(setf *features* (set-difference *features* '(:example
					      )))

(progn
  (defparameter *project* "143_helium_gemini")
  ;; name input-price output-price context-length harm-civic
  
  (let ((iflash .075)
	(oflash .3)
	(ipro 1.25)
	(opro 5))
   (defparameter *models* `((:name gemini-1.5-pro-exp-0827 :input-price ,ipro :output-price ,opro :context-length 128_000 :harm-civic t)
			    (:name gemini-2.0-flash-exp :input-price ,iflash :output-price ,oflash :context-length 128_000 :harm-civic t)
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
  (defparameter *idx* "03") 
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

  (let* ((notebook-name "google_genai_sdk_host"))
    (write-source
     (format nil "~a/source03/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "micromamba install python-fasthtml markdown; pip install google-genai webvtt-py")

       (imports (
					; re
		 ;markdown
					; uvicorn
		 sqlite_minutils.db
		 datetime
		 time
		 subprocess
		 webvtt))

       (imports-from ;(fasthtml.common *)
		     (google genai)
		     (google.genai types) ;; pydantic model types for paramters (alternative to TypedDict)
		     )
       (setf start_time (time.time))

       (do0
	(comments "Call yt-dlp to download the subtitles")
	(setf url (string "https://www.youtube.com/watch?v=ttuDW1YrkpU"))
	(setf sub_file (string "/dev/shm/o"))
	(setf sub_file_ (string "/dev/shm/o.en.vtt"))
	(subprocess.run (list (string "yt-dlp")
			      (string "--skip-download")
			      (string "--write-auto-subs")
			      (string "--write-subs")
			      (string "--sub-lang")
			      (string "en")
			      (string "-o")
			      sub_file
			      url))
	(for (c (webvtt.read sub_file_))
	     (setf start (dot c start (aref (split (string ".")) 0)))
	     ,(lprint :vars `(		;c.identifier
			      start
					;c.end
			      c.text
					;c.voice
			      ))))

       #+Nil
       (do0
	(comments "genai manual: https://googleapis.github.io/python-genai/")
	" "
	(comments "Read the gemini api key from  disk")
	(with (as (open (string "api_key.txt"))
                  f)
              (setf api_key (dot f (read) (strip))))
	(setf client (genai.Client :api_key api_key))

	
	(setf prompt (string "Tell me a joke about rockets!"))
	(setf model (string "gemini-2.0-flash-exp"))
	(setf safeties (list))
	(for (harm (aref types.HarmCategory.__args__ (slice 1 "")))
	     (comments "skip unspecified")
	     (safeties.append (types.SafetySetting
			       :category harm
			       :threshold (string "BLOCK_NONE") ;; or OFF
			       )))
	(setf config (types.GenerateContentConfig :temperature 2s0
						  :safety_settings safeties))
	(for (chunk
	      (client.models.generate_content_stream
	       :model model
	       :contents prompt
	       :config config
	       ))
	     (print chunk.text)))

       ))))
