(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "155_fileselect")
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

  
  (let* ((notebook-name "fileselect"))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "given a text file containing parts of filenames and a directory create a list of files of the files that match the parts")
       (imports (
		 time
		 ;json
					;os
		 tqdm
		 ;subprocess
		 pathlib
		 ;concurrent.futures
					; re
					;markdown
					; uvicorn
					;sqlite_minutils.db
		 ;datetime
					;time
		 ;(pd pandas)
		 ;(np numpy)
		 ;requests
		 ;random
		 argparse
		 ))

       ;(imports-from (sqlite_minutils *))
       ;(setf db (Database (string "tide.db")))
       ;(setf users (Table db (string "Users")))
       
       #+nil
       (do0 (setf res (list))
	    (for (row (tqdm.tqdm users.rows))
		 (setf q (json.loads (aref row (string "data"))))
		 (setf d (dictionary :id (aref row (string "id"))
				     ))
		 
		 (res.append d)))


       (do0
	(setf parser (argparse.ArgumentParser))
	 (parser.add_argument (string "input_paths")
			      :nargs (string "+")
			      :help (string "Path(s) to search for matching files."))
	 ,@(loop for e in `((:name "file-parts-from"
				   ;:default (string "")
				   :help "A text file with parts that shall occur in the filename."))
		 collect
		 (destructuring-bind (&key name default help) e
		   (let ((cmd `(parser.add_argument (string ,(format nil "--~a" name))
			)))
		     (when default
		       (setf cmd (append cmd `(:default ,default))))
		     (when help
		       (setf cmd (append cmd `(:help (string ,help)))))
		     cmd))))
       
       ))))
