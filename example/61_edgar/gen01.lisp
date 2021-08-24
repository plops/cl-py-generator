(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)
;; https://pypi.org/project/python-edgar/
;; python3 -m pip install --user  git+https://github.com/edgarminers/python-edgar

;; limit to 10 requests per second 

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/61_edgar")
  (defparameter *code-file* "run_01_read_edgar")
  (defparameter *source* (format nil "~a/source/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
		   (format (- (time.time) start_time)
			   ,@rest)))))
  (let* (
	 (code
	   `(do0
	     (imports (			;os
					;sys
					;time
					;docopt
					pathlib
					;(np numpy)
					;serial
		       (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
		       scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
		       (np numpy)
					; scipy.sparse
					;scipy.sparse.linalg
					; jax
					;jax.random
					;jax.config
					;copy
		       subprocess
		       threading
					;datetime
					;time
		       ; mss
		       ;cv2
		       time
		       edgar
		       tqdm
		       requests
		       ))
	     #+nil (imports-from (selenium webdriver)
			   (selenium.webdriver.common.keys Keys)
			   )
	     "from bs4 import BeautifulSoup"
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"))
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
				 (- tz)))))
	     (setf start_time (time.time)
		   debug True)

	     (setf fns ("list"
			(dot 
			 (pathlib.Path (string "data/"))
			 (glob (string "*-QTR*.tsv")))))

	     ,(let ((out-csv `(string "mu-links.csv")))
		`(if (dot (pathlib.Path ,out-csv)
			  (exists))
		     (do0
		      (setf df (dot (pd.read_csv ,out-csv)
				    (sort_values :by (string "filing_date")
						 :ascending False))))
		     (do0 
		  (setf df (pd.DataFrame))
		  (for (fn (tqdm.tqdm fns))
		       (setf df0 (pd.read_csv fn
					      :sep (string "|")
					;:lineterminator (string "\\n")
					      :names (list ,@(loop for e in `(cik name filing_type
										  filing_date
										  filing_url_txt
										  filing_url)
								   collect
								   `(string ,e)))
					      :header None))
		   
		       (setf df_ (aref df0
				       (& (== df0.name (string "MICRON TECHNOLOGY INC"))
					  (== df0.filing_type (string "10-Q")))))
		       (setf df (df.append df_)))
		  (setf df (dot df (sort_values :by (string "filing_date"))
				))
		  (setf (aref df (string "url"))
			(dot df
			     filing_url
			     (apply (lambda (x)
				      (dot (string "https://www.sec.gov/Archives/{}")
					   (format x))))))
		  (df.to_csv ,out-csv))))
	     (for (url (tqdm.tqdm df.url))
	      (do0
	       (do0
		(setf user_agent (string "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36"))
					;(setf url (dot df url (aref iloc 0)))
		(setf response (dot requests 
				    (get url
					 :headers
					 (dict ((string "User-agent") user_agent)))))
		(setf tables
		      (pd.read_html
		       (dot 
			response text)
		       ))
		(setf tab1 (aref tables 1)) 
		#+nil  (setf doc_name  (dot  (aref tab1 (== tab1.Type (string "EX-101.INS")))
					     Document))
		(do0
		 (setf soup (BeautifulSoup response.text (string "html.parser")))
		 (setf btab1 (aref (soup.findAll (string "table"))
				   1))
		 (setf links (list))
		 (do0 
		  (for (tr (btab1.findAll (string "tr")))
		       (setf trs (tr.findAll (string "td")))
		       (for (each trs)
			    (try
			     (do0
			      (setf link (dot each (aref (find (string "a"))
							 (string "href")))
				    )
			      (links.append link))
			     ("Exception as e"
			      pass)
			     ))))
		 (setf (aref tab1 (string "Link"))
		       links)
		 (setf (aref tab1 (string "url"))
		       (dot tab1
			    Link
			    (apply (lambda (x)
				     (dot (string "https://www.sec.gov{}")
					  (format x))))))
		 (setf (aref tab1 (string "filename"))
		       (dot tab1
			    Link
			    (apply (lambda (x)
				     (dot pathlib
					  (Path x)
					  name)
				     )))) 

		 (for ((ntuple idx row) (tab1.iterrows))
		  (do0
		   
		   (setf url (dot row			
				  url))
		   (setf response
			 (dot requests 
			      (get url
				   :headers
				   (dict ((string "User-agent") user_agent)))))
		   (with (as (open (/ (pathlib.Path (string "xml_all"))
				      row.filename) 
						    (string "w"))
			     f)
			 (f.write (dot response
				       text)))))
		 #+nil 
		 (do0
		  (comments "only download 2 files (xml and xsd)")
		  ,@(loop for (e f suffix) in `((ins (EX-101.INS XML) xml)
						  (sch EX-101.SCH xsd))
			    collect
			    `(do0
			      (setf row (dot (aref tab1 ,(if (listp f)
							     `(logior ,@(loop for ff in f
									      collect
									      `(== tab1.Type (string ,ff)))
								      )
							     `(== tab1.Type (string ,f))))
					     (aref iloc 0)))
			      (setf url (dot row			
					     url))
			      (setf ,(format nil "response_~a" e)
			            (dot requests 
					 (get url
					      :headers
					      (dict ((string "User-agent") user_agent)))))
			      (with (as (open (/ (pathlib.Path (string ,suffix))
						 row.filename) #+nil(string ,(format nil "response_~a.~a" e suffix))
							       (string "w"))
					f)
				    (f.write (dot ,(format nil "response_~a" e)
						  text))))))))))
	     
	     )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    ))

