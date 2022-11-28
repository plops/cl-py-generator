(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  ;; the following code needs inverted readtable, otherwise symbols
  ;; and filenames may have the wrong case and everything breaks in
  ;; horrible ways
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/80_fulltext")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key msg vars)
    `(print (dot (string ,(format nil "{:7.6f} ~a ~{~a={}~^ ~}" msg vars))
                 (format
		  (- (time.time) start_time)
                  ,@vars))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create python file in a directory below source/"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}" path))
					    ))) code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (gen `(collect_pdf_files)
	   `((python
	      (cell
	       (comments "python3.10 -m pip install --user tqdm"
			 "sudo dnf install poppler-utils")
	       (imports (pathlib
			 time
			 tqdm
			 (pd pandas)
			 subprocess
			 (db sqlite3)
			 ))
	       (setf start_time (time.time))))
	     #+nil
	     (python
	      (cell
	       ,(lprint :msg "find all pdf files")
	       (setf fns ("list"
			  (dot pathlib (Path (string "/home/martin"))
			       (glob (string "**/*.pdf")))))
	       ,(lprint :msg "numer of pdf files" :vars `((len fns)))
	       ))
	     #+nil
	     (python
	      (cell
	       ,(lprint :msg "get sizes of pdf files")
	       (do0
		(setf res (list))
		(for (fn (tqdm.tqdm fns))
		     (setf stat (dot fn (stat)))
		     (res.append (dict
				  ((string "fn") fn)
				  ,@(loop for e in `((dev)
						     (ino inode_number)
						     (mode)
						     (nlink number_hard_links)
						     (uid)
						     (gid)
						     (rdev special_file_id)
						     (size bytes)
						     (blksize)
						     (blocks blocks512B)
						     (atime last_access_time)
						     (mtime last_modification_time)
						     (ctime last_change_time))
					  collect
					  (destructuring-bind (&optional short (long short)) e
					    `((string ,long)
					      (dot stat ,(format nil "st_~a" short))))))))
		(setf df (pd.DataFrame res))
		(setf sum_pdf_size_MB (/ (df.bytes.sum)
					 (* 1024.0 1024)))
		,(lprint :msg "size of all pdfs" :vars `(sum_pdf_size_MB)))))
	     ,@(let ((table-def `((:name fn :type text)
				  (:name pdf_bytes :type integer)
				  (:name txt :type text)
					;(:name pdf_data :type blob)
				  )))
		 `(
		   #+nil
		   (python
		    (cell
		     ,(lprint :msg "convert pdf files to text and add to database")
		     (setf con (db.connect (string "example.db"))
			   cur (con.cursor))
		     (comments "this table has a 64bit rowid as key")
		     (cur.execute (rstring3
				   ,(format nil "CREATE TABLE IF NOT EXISTS docs (~{~a~^,~})"
					    (loop for e in table-def
						  collect
						  (destructuring-bind (&key name type) e
						    (format nil "~a ~a" name (string-upcase type)))))))
		     (def fn_exists_p (fn)
		       (cur.execute (rstring3 "SELECT fn FROM docs WHERE fn = :fn ")
				    (dictionary :fn fn))
		       (setf row (cur.fetchall))
		       (return (< 0 (len row))))
		     (for ((ntuple idx row) (tqdm.tqdm ("list" (df.iterrows))))
			  (setf fn (str row.fn)
				pdf_bytes row.bytes)
			  (if (fn_exists_p fn)
			      (do0
			       ,(lprint :msg "exists already" :vars `(fn)))
			      (do0
					;,(lprint :msg "convert to text" :vars `(fn))
			       (setf proc (subprocess.run (list (string "/usr/bin/pdftotext")
								row.fn
								(string "-"))
							  :capture_output True))
					;(proc.wait)
			       (setf txt (proc.stdout.decode (string "utf-8"))
				     )
			       (cur.execute (rstring3 ,(format nil "INSERT OR REPLACE INTO docs VALUES (~{:~a~^,~})"
							       (loop for e in table-def
								     collect
								     (destructuring-bind (&key name type) e
								       name))
							       ))
					    (dict ,@(loop for e in table-def
							  collect
							  (destructuring-bind (&key name type) e
							    `((string ,name) ,name)))))
			       (con.commit))))
		     (cur.close)
		     (con.close)

		     ))
		   (python
		    (cell
		     ,(lprint :msg "convert pdf files to text and add to database")
		     (setf con (db.connect (string "example.db"))
			   cur (con.cursor))
		     (cur.execute (rstring3
				   ,(format nil "CREATE VIRTUAL TABLE fts2 USING FTS5(fn,txt,content=tbl)"
					    )))
		     ;; CONTENT=tbl,CONTENT_ROWID=rowid,TOKENIZE='porter unicode61 remove_diacritics 1')

		     (cur.close)
		     (con.close)

		     ))))))))
  (sb-ext:run-program "/usr/bin/sh"
		      `("/home/martin/stage/cl-py-generator/example/80_fulltext/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




