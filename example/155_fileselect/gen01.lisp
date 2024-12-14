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
      (print (dot (string ,(format nil "~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format 
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
		 sys
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
		 (pd pandas)
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
			     :help "A text file with parts that shall occur in the filename.")
			    (:name "min-size"
			     :type int
			     :default 0
			     :help "Minimum size in bytes for a file to be selected.")
			    (:name "suffix"
			     :type str
			     :default (string "*")
			     :help "File suffix pattern that must match the filename (e.g. *.mp4). The default pattern accepts all."))
		 collect
		 (destructuring-bind (&key name default type help) e
		   (let ((cmd `(parser.add_argument (string ,(format nil "--~a" name)))))
		     (when type
		       (setf cmd (append cmd `(:type ,type))))
		     (when default
		       (setf cmd (append cmd `(:default ,default))))
		     (when help
		       (setf cmd (append cmd `(:help (string ,help)))))
		     cmd)))
	 (setf args (parser.parse_args)))

       (do0
	(comments "stop if input_paths is empty")
	(when (== (len args.input_paths)
		  0)
	  (sys.exit 0)))
       (do0
	(setf files (list))
	(for (input_path args.input_paths)
	     (setf path (pathlib.Path input_path))
	     (cond ((path.is_dir)
		    (files.extend (path.rglob args.suffix)))
		   ((and (path.is_file)
			 (== path.suffix args.suffix))
		    (files.append path))))
	(setf df (pd.DataFrame (dictionary :file files))))

       (do0
	(comments "load parts")
	(with (as (open args.file_parts_from)
		  f)
	      (setf parts (f.readlines))))

       (do0
	,(lprint :msg "collect file sizes")
	(setf res (list))
	(for ((ntuple idx row) (tqdm.tqdm (df.iterrows)))
	     (setf st_size 0)
	     (try
	      (do0
	       (comments "this throws for dangling symlinks")
	       (setf st_size (dot row.file (stat) st_size)))
		  ("Exception as e"
		   ;(print e)
		   pass))
	     (res.append (dictionary :file (str row.file)
				     :st_size st_size)))
	(setf df (pd.DataFrame res)))
       (comments "keep only rows that have st_size>=args.min_size")
       (setf df (aref df (<= args.min_size df.st_size)))
       )))
  )
