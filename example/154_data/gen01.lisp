(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "154_data")
  (defparameter *idx* "01") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *languages* `(en de fr ch nl pt cz it jp ar))
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

  
  (let* ((notebook-name "show"))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (imports (
		 json
					;os
		 tqdm
		 ;subprocess
		 ;pathlib
		 ;concurrent.futures
					; re
					;markdown
					; uvicorn
					;sqlite_minutils.db
		 ;datetime
					;time
		 (pd pandas)
		 ))

       (imports-from (sqlite_minutils *))
       (setf db (Database (string "tide.db")))
       (setf users (Table db (string "Users")))
       #+nil (setf q ("list" (for-generator (row (tqdm.tqdm users.rows)) (json.loads (aref row (string "data"))))))
       (setf res (list))
       (for (row (tqdm.tqdm users.rows))
	    (setf q (json.loads (aref row (string "data"))))
	    (setf d (dictionary))
	    (do0
	     ,@(loop for e in `(name
				id
				birth_date
				bio
				schools
				jobs
				locations
				distance
				)
		     collect
		     `(try
		       (do0 (setf (aref d (string ,e))
				  (aref q (string ,e)))
			    )
		       ("Exception as e"
			pass)
		       )))
	    
	    (for (s (aref row (string "selected_descriptors")))
		 (try
		  (setf (aref d (aref s (string "name")))
			(aref (aref (aref s (string "choice_selections"))
				    0)
			      (string "name")))
		  ("Exception as e"
		   pass)))
	    
	    (res.append d))
       (setf df (pd.DataFrame res))
       )))
)
