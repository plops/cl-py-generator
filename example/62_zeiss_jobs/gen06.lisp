(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria")
  ;(ql:quickload "cl-who")
  )
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/62_zeiss_jobs")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                           ,@rest)))))

  (write-notebook
   :nb-file (format nil "~a/source/06_create_postgres.ipynb" *path*)
   :nb-code
   `(
     (python (do0
	      (do0
	       "%matplotlib notebook"
	       #-nil(do0
		  
		     (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		     (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation) 
					;(xrp xarray.plot)
			       ))
                  
		     (plt.ion)
					;(plt.ioff)
		     ;;(setf font (dict ((string size) (string 6))))
		     ;; (matplotlib.rc (string "font") **font)
		     )
	       (imports (		;os
					;sys
			 time
					;docopt
			 ;pathlib
					;(np numpy)
					;serial
			 (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;   scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
			 ;re
			 ;json
			 ; csv
					;io.StringIO
					;bs4
					;requests
			 psycopg2
			   
					;(np jax.numpy)
					;(mpf mplfinance)
			 ;selenium.webdriver ;.FirefoxOptions
			   
			 ))
	       
		 
	       
	       (imports-from (matplotlib.pyplot
			      plot imshow tight_layout xlabel ylabel
			      title subplot subplot2grid grid
			      legend figure gcf xlim ylim)
			     (memory_profiler memory_usage)
			     (functools wraps)
			     (bs4 BeautifulSoup)
			     )
	       
		 
	       )
	      ))
     (python
      (do0
       (setf start_time (time.time)
	     debug True)
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
	     debug True)))

     (python
      (do0
       (comments "sudo systemctl start postgresql")
       (setf connection (psycopg2.connect :host (string "127.0.0.1")
					  :database (string "zeiss")
					  :user (string "martin")
					  :password None
					  )
	     connection.autocommit True)
       ))

    (python
      (do0
       (def profile (fn)
	 (@wraps fn)
	 (def inner (*args **kwargs)
	   (setf fn_kwargs_str (dot (string ", ")
				    (join (for-generator ((ntuple k v)
							  (kwargs.items))
							 (dot (string "{}={}")
							      (format k v))))))
	   (print (dot (string "{}({})")
		       (format fn.__name__
			       fn_kwargs_str)))
	   (setf start (time.perf_counter))
	   (setf retval (fn *args **kwargs))
	   (setf elapsed (- (time.perf_counter)
			    start))
	   (print (dot (string "Time {:0.4}")
		       (format elapsed)))

	   (setf (ntuple mem
			 retval)
		 (memory_usage (tuple fn args kwargs)
			       :retval True
			       :timeout 200
			       :interval 1e-7))
	   (print (dot (string "Memory {}")
		       (format (- (max mem)
				  (min mem)))))
	   (return retval))
	 (return inner))))
     
     (python
      ,(let ((tab "staging_jobs")
	     (tab-contents `((id INTEGER)
			     (job TEXT)
			     (location TEXT)
			     (link TEXT)
			     (description TEXT)
			     ,@(loop for e in `(text0 text1 text2 text3
						      jobtitles locations
						      recruiter_info)
				     collect
				     `(,e TEXT))
			     )))
	 `(do0
	   (def create_staging_table (cursor)
	     (cursor.execute
	      (string3 ,(format nil "DROP TABLE IF exists ~a;
CREATE UNLOGGED TABLE ~a (~{~a~^,~%~});"
				tab
				tab
				(loop for (e f) in tab-contents
				      collect
				      (format nil "~a ~a" e f))))))
	   "@profile"
	   (def insert_one_by_one (connection data)
	     (with (as (connection.cursor)
		       cursor)
		   (create_staging_table cursor)
		   (for (e data)
			(dot cursor
			     (execute
			      (string3
			       ,(format nil "INSERT INTO ~a VALUES (~{%(~a)s~^,~%~})" tab  (mapcar #'first tab-contents)))
			      e))))))))
     (python
      (do0
       (with (as (connection.cursor)
		 cursor)
	     (create_staging_table cursor)
	     )
       (setf df (pd.read_csv (string "contents3.csv")))

       (do0
	(comments "rename id column")
	(setf df1 (dot df (rename :columns (dict ((string "Unnamed: 0")
						  (string "id")))))))
       (do0
	(comments "load html files")
	(def load_html (row)
	  (setf fn (dot row.link
			(aref (split (string "https://"))
			      1)))
	  (with (as (open fn)
		    f)
		(return (f.read))))
	(setf (aref df1 (string "description"))
	      (dot df1 (apply load_html :axis 1))))
       (do0
	(comments "parse html sites")
	(do0
	 ,(let* ((l `((h1 jobtitle)
		     (h2 locations)
		     (div recruiter-info :code
			  (lambda (x)
			    (dot (string " ")
				 (join (dot (aref x 1)
					    text
					    (strip)
					    (split))))))
		     ))
		(vars `(,@(loop for i below 4
			     collect
			     (format nil "text~a" i)
			      )
		      
		      ,@(loop for e in l
			      collect
			      (destructuring-bind (element class
						   &key
						     (code `(lambda (x)
							      (dot (aref x 0)
								   text))))
				  e
				(let (
				      (var (substitute #\_ #\- (format nil "~a" class))))
				  var))))))
	    `(do0
	      (do0
	       (def parse_html (row)
		 (setf soup (BeautifulSoup (dot row description )
					   (string "html.parser")))
		 ,@(loop for e in l
			 collect
			 (destructuring-bind (element class
					      &key
						(code `(lambda (x)
							 (dot (aref x 0)
							      text))))
			     e
			   (let ((var0 (substitute #\_ #\- (format nil "~a0" class)))
				 (var (substitute #\_ #\- (format nil "~a" class))))
			     `(do0
			       (setf ,var0
				     (dot soup
					  (find_all (string ,element )
						    :class_ (string ,class))))
			       (setf ,var (,code ,var0))
			       ))))

		 (do0
		  (setf texts (dot soup
				   (find_all (string "div")
					     :class_ (string "text"))))
		  ,@(loop for i below 4
			  collect
			  `(setf ,(format nil "text~a" i)
				 (aref texts ,i)))

		  (return
		    (list
		     ,@vars
		     ))))

	       (setf (aref df1 (list ,@(mapcar #'(lambda (x) `(string ,x))
				       vars)))
	     (dot df1 (apply parse_html :axis (string "columns")
					      :result_type (string "expand"))))
	       ))
	    )))
       
       (insert_one_by_one connection (dot df1
					  (to_dict (string "records"))))
       ))

     ,(let ((l `((h1 jobtitle)
		(h2 locations)
		(div recruiter-info :code
		     (lambda (x)
		       (dot (string " ")
			    (join (dot (aref x 1)
				       text
				       (strip)
				       (split))))))
		 )))
      `(python
	(do0
	 (comments "prototype code to parse html")
	 (do0
	  (setf idx 1)
	  (setf soup (BeautifulSoup (dot df1 (aref iloc idx) description )
				    (string "html.parser")))
	  ,@(loop for e in l
		  collect
		  (destructuring-bind (element class
				       &key
					 (code `(lambda (x)
						  (dot (aref x 0)
						       text))))
		      e
		    (let ((var0 (substitute #\_ #\- (format nil "~a0" class)))
			  (var (substitute #\_ #\- (format nil "~a" class))))
		      `(do0
			(setf ,var0
			      (dot soup
				   (find_all (string ,element )
					     :class_ (string ,class))))
			(setf ,var (,code ,var0))
			))))

	  (do0
	   (setf texts (dot soup
			    (find_all (string "div")
				      :class_ (string "text"))))
	   ,@(loop for i below 4
		   collect
		   `(setf ,(format nil "text~a" i)
			  (aref texts ,i)))
	   ,@(loop for i below 4
		   collect
		   `(print (dot (string ,(format nil "text~a='{}'" i))
				(format ,(format nil "text~a" i))))))
	  ,@(loop for e in l
		  collect
		  (destructuring-bind (element class
				       &key
					 (code `(lambda (x)
						  (dot (aref x 0)
						       text))))
		      e
		    (let (
			  (var (substitute #\_ #\- (format nil "~a" class))))
		      `(print (dot (string ,(format nil "~a = '{}'" var))
				   (format ,var)) )))))
	  )))
     
   #+nil  (python
      (do0
       "@profile"
       (def work (n)
	 (for (i (range n)
		 )
	      (** 2 n)))
       (work 10)
       (work :n 10_000)))))
  )



