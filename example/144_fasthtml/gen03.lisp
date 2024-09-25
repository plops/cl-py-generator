(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)


(setf *features* (union *features* '(:example)))
(setf *features* (set-difference *features* '(;:example
					      )))

(progn
  (defparameter *project* "144_fasthtml")
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

  
  
  (let* ((notebook-name "boost")
	 (proc-files `(bootconfig
buddyinfo
cgroups
cmdline
;config.gz
consoles
cpuinfo
crypto
devices
diskstats
dma
execdomains
filesystems
interrupts
iomem
ioports
kallsyms
;kcore
key-users
keys
;kmsg
;kpagecgroup
;kpagecount
;kpageflags
latency_stats
loadavg
locks
meminfo
misc
modules
mtrr
pagetypeinfo
partitions
schedstat
slabinfo
softirqs
stat
swaps
sysrq-trigger
timer_list
uptime
version
;vmallocinfo
vmstat
zoneinfo
) )
	 )
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `
     (do0
      "#!/usr/bin/env python3"
      ;(comments "pip install -U python-fasthtml markdown")

      #+nil (imports (	       ;(genai google.generativeai)
					;google.generativeai.types.answer_types
					;os
					;google.api_core.exceptions
		;re
		markdown
		uvicorn
		;sqlite_minutils.db
		datetime
		time))

      

      (imports-from (fasthtml.common *))

      #+nil (def render (summary)
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
      
      ;(comments "summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
      (setf (ntuple app rt		; summaries Summary
		    )
	    (fast_app		;:db_file (string "data/summaries.db")
	     :live True
					;:render render
	     #+nil ,@(loop for e in db-cols
			   appending
			   (destructuring-bind (&key name type) e
			     `(,(make-keyword (string-upcase (format nil "~a" name)))
			       ,type)))

		       
					;:pk (string "identifier")
	     ))



      (setf proc_files (list ,@(loop for e in proc-files collect `(string ,e))))

      (setf proc_links (list
		   (for-generator (f proc_files)
				  (Li (A f
				       
					 :href (fstring "/{f}"))))))
      (setf nav (Nav 
		 (Ul (Li (Strong (string "Linux Discover"))))
		 (Ul
		  *proc_links
		  #+nil ,@(loop for e in proc-files
				collect
				`(Li (A (string ,e)
				       
					:href (string ,(format nil "/~a" e))
				      
					)))
		  :hx_boost True
		       
		  )))

      (def get_proc_contents (proc)
	(setf lines (string ""))
	(when (< 0 (len proc))
	  (with (as (open (fstring "/proc/{proc}")
			  )
		    f)
		(setf lines (f.readlines))))

	(setf target (fstring "/{proc}"))
	(return (Pre (dot (string "") 
			  (join lines))
		     :hx_get (fstring "/{proc}/pre")
		     :hx_trigger (string "every 1s")
		     :hx_swap (string "outerHTML")
		     )
		)
	)
      
      (do0
       (@rt (string "/{proc}"))
       (def get (proc request)
	 (declare (type Request request)
		  (type str proc))
	 
	; (print (fstring "proc={proc} client={request.client.host}"))

	#+NIl (do0
	  (setf lines (string ""))

	  (when (< 0 (len proc))
	    (with (as (open (fstring "/proc/{proc}")
			    )
		      f)
		  (setf lines (f.readlines)))))

	 (return (Titled (fstring "/{proc}")
			 (Main nav
			     
			     
					; form
					; gen_list
					; summary_list
			       (get_proc_contents proc)
			       #+nil (Pre (dot (string "") 
					       (join lines))
					  :hx_get (fstring "/{proc}/pre")
					  :hx_trigger (string "every 1s")
					  :hx_swap (string "outerHTML")
					  )
			       :cls (string "container"))))))

      (do0
       (@rt (string "/{proc}/pre"))
       (def get (proc request)
	 (declare (type Request request)
		  (type str proc))
	 (return (get_proc_contents proc))
	 (setf lines (string ""))
	 (when (< 0 (len proc))
	  (with (as (open (fstring "/proc/{proc}")
			  )
		    f)
		(setf lines (f.readlines))))

	 (setf target (fstring "/{proc}"))
	 (return (Pre (dot (string "") 
			   (join lines))
		      :hx_get (fstring "/{proc}/pre")
		      :hx_trigger (string "every 1s")
		      :hx_swap (string "outerHTML")
		      )
		 )))

      (do0
       (@rt (string "/"))
       (def get (request)
	 (declare (type Request request))
	 
	 (print request.client.host)
	 
	 #+nil 
	 (setf transcript (Textarea :placeholder (string "Paste YouTube transcript here")
				    :name (string "transcript"))
	       model (Select (Option (string "gemini-1.5-flash-latest"))
			     (Option (string "gemini-1.5-pro-exp-0801"))
			     
			     :name (string "model")))
	 #+nil
	 (setf form
	       (Form
		(Group
		 transcript
		 model
		 (Button (string "Summarize Transcript")))
		:hx_post (string "/process_transcript")
		:hx_swap (string "afterbegin")
		:target_id (string "gen-list")))

	 #+nil
	 (setf gen_list (Div :id (string "gen-list")))

	 
	 (return (ntuple (Title (string "Linux Discover Tool"))
			 (Main nav
			       
			       
					; form
					; gen_list
					; summary_list 
			       
			       :cls (string "container"))))))
       
      #+nil
      (do0
       (@app.post (string "/generations/{identifier}"))
       (def get (identifier)
	 (declare (type int identifier))
	 (return (generation_preview identifier))))

      #+nil
      (do0 
       (@rt (string "/process_transcript"))
       (def post (summary request)
	 (declare (type Summary summary)
		  (type Request request))
	 ))

      (serve :host (string "0.0.0.0") :port 5001)
      ))))
