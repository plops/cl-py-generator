(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "train_llm")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/~a/" *project*))
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
                   (format  (- (time.time) start_time)
                            ,@vars)))))

  
  
  (let* ((notebook-name "collect_examples_as_csv")
	 (cli-args `(#+nil (:short "c" :long "chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     #+nil (:short "p" :long "prompt" :type str
		      :default (string "Summarize the following video transcript as a bullet list.")
		      :help "The prompt to be prepended to the output file(s).")))
	 )
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
      
       (imports (os
		 time
		 pathlib
		 re
		 ;sys
		 (pd pandas)
		 ))

       #-nil(do0
	(setf start_time (time.time)
	      debug True)
	(setf
	 _code_git_version
	 (string ,(let ((str (with-output-to-string (s)
			       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		    (subseq str 0 (1- (length str)))))
	 _code_repository (string ,(format nil
					   "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/"
					   *project*))
	 _code_generation_time
	 (string ,(multiple-value-bind
			(second minute hour date month year day-of-week dst-p tz)
		      (get-decoded-time)
		    (declare (ignorable dst-p))
		    (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			    hour
			    minute
			    second
			    (nth day-of-week *day-names*)
			    year
			    month
			    date
			    (- tz))))))

       (do0
	(setf directory (dot pathlib (Path (string "/home/martin/stage/cl-py-generator"))))


	(setf training_data (list))
	,(let ((l-ext `(".py")))
	   (flet ((add ()
		    `(do0
				  (setf lisp_content (f.read_text)
					text_input (string "Convert the following Python code into s-expressions: \\n"))
				  (for (output_file output_files)
				       (incf text_input (fstring "// {output_file}\\n{output_file.read_text()}\\n\\n")))
				  (training_data.append
				   (dictionary
				    :path  (fstring "{f.parent.stem}/{f.stem}")
				    :text_input text_input
				    :output lisp_content)))))
	    `(do0

	      (setf gen_files0 ("list" (dot (/ directory (string "example")) (rglob (string "gen*.lisp")))))

	      (do0
	       (setf gen_files1 (list))
	       (for (f gen_files0)
		    (do0
		     (comments "exclude C++ generating files")
		     (setf content (f.read_text))
		     (when (re.search (rstring3 "\\(ql:quickload \"cl-cpp-generator2\"\\)"
						)
				      content)
		       (print (fstring "Info 0: Skip C++ generator {f.parent.stem}/{f.stem}."))
		       continue))
		    (setf folder f.parent)
		    (do0
		     (comments "count the number of python files. and also the characters (in column len_py)")
		     (setf py_files ("list" (folder.rglob (string "*.py"))))
		     (setf text_py (string "Create s-expressions that corresponds to the following Python code: "))
		     (do0
			  (for (p py_files)
			       (incf text_py (p.read_text))
			       ))
		     (setf len_py (len text_py))
		     (setf n_py (len py_files)))
		    (do0
		     (comments "same stats for notebooks")
		     (setf ipynb_files ("list" (folder.rglob (string "*.ipynb"))))
		     (do0 (setf len_ipynb 0)
			  (for (p ipynb_files)
			       (incf len_ipynb (len (p.read_text)))))
		     
		     (setf n_ipynb (len ipynb_files)))
		    (do0
		     (comments "count characters in lisp file")
		     (setf len_lisp (len (f.read_text))))
		    (gen_files1.append (dictionary :file f
						   :text_lisp (f.read_text)
						   :len_lisp len_lisp
						   :folder folder
						   :n_py n_py
						   :len_py len_py
						   :text_py text_py
						   :n_ipynb n_ipynb
						   :len_ipynb len_ipynb
						   :py_files py_files
						   :ipynb_files ipynb_files
						   :short (fstring "{folder.stem}/{f.stem}")
						   )))
	       (setf g1 (pd.DataFrame gen_files1)))
	      

	      (do0
	       (comments "count number of python-generating lisp files in this directory")
	       (setf folder_counts (dot g1 (groupby (string "folder"))
					(size)))
	       (setf g1 (g1.merge (folder_counts.rename (string "n_lisp")) :left_on (string "folder")
									   :right_index True)))


	      (do0
	       (comments "find folder with one python-generating .lisp input and no .py file. that should be generated, then")
	       (setf g20 (dot (aref g1 (& (== g1.n_lisp 1)
					  (== g1.n_py 0)
					  (!= g1.n_ipynb 1)))
			      (sort_values :by (string "short"))))
	       ,(lprint :msg "the following folders need python file"
			:vars `(g20.short))
	       )

	      
	      (do0
	       (comments "find folder with one python-generating .lisp input and one .py file") ;; 51 rows
	       (setf g2all (aref g1 (& (== g1.n_lisp 1)
				    (== g1.n_py 1)
				    )))
	       (print (dot g2all (aref (sort_values :by (string "len_lisp"))
				       (list (string "short")
					     (string "len_lisp")
					     (string "len_py")))))
	       (setf g2 (aref g2all (&
				     (< g1.len_py 40000)
				     (< g1.len_lisp 5000))))
	       ;; character limit leaves 27 examples out of 51
	       (g2.to_csv (string "/dev/shm/python_to_sexpr.csv"))
	       )

	      #+nil
	      (for ((ntuple idx row) (g1.iterrows))
		   
		   (setf n (len (aref g1 (== g1.folder row.folder))))
		   (setf (dot (aref g1.iloc idx) n) n))

	      #+nil
	      (for ((ntuple idx row) (g1.iterrows))
		   (setf f (aref row (string "file")))
		   
		   (comments "genXX.lisp -> sourceXX")
		   (setf output_dir (/ f.parent (dot (string "source{}")
						     (format (aref f.stem (slice 3 5))))))
		   (if (output_dir.exists)
		       (do0
			(setf output_files (+ ,@(loop for e in l-ext
						      collect
						      `("list" (dot output_dir (rglob (string ,(format nil "*~a" e))))))))
			(if (< 0 (len output_files))
			    (do0
			     (print (fstring "Info 1: Found match {f} {len(output_files)}."))
			     ,(add) )
			    (do0
			     (print (fstring "Warning 1: No matches in output directory for {f}."))
			     continue)))
		       (do0
			(setf content (f.read_text))
			(setf match (re.search (rstring3 "\\(defparameter \\*source-dir\\* .*\\\"(.*)\\\"\\)")
					       content))
			(if match
			    (do0
			     (setf output_dir (/ directory
						 (match.group 1))
				   output_files (+ ,@(loop for e in l-ext
							   collect
							   `("list" (dot output_dir (rglob (string ,(format nil "*~a" e))))))))
			     (if (< 0 (len output_files))
				 (do0
				  (print (fstring "Info 2: Found match {f} {len(output_files)}."))
				  ,(add))
				 (do0
				  (print (fstring "Warning 2: Not enough files for {f} in {output_dir} gp1={match.group(1)}."))
				  (print (fstring "Warning 4: match={match} ls {output_dir}={output_files}."))
			      
				  continue)))
			    (do0
			     (print (fstring "Warning 3: Could not determine output directory for {f}."))
			     continue)))))

	      #+nil
	      (do0 (setf df (pd.DataFrame training_data))
		   ,@(loop for e in `(text_input output)
			   collect
			   `(setf (aref df (string ,(format nil "~a_len" e)))
				  (dot df ,e str (len))))
		   
		   (setf df1 (aref df (& (< df.text_input_len 40000)
					 (< df.output_len 5000))))
		   (setf df1 (df1.sort_values :by (string "path")))
		   
		   (df1.to_csv (string "training_data.csv")
			       :index False))
	      ))))
       
       
       )))

  )
