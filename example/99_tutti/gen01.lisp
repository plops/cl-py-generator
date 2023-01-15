(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "99_tutti")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    cmd
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    rest)))
                   (format  (- (time.time) start_time)
                            ,@rest)))))

  
  
  (let* ((notebook-name "get_links")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       (do0
	,(format nil "#|default_exp p~a_~a" *idx* notebook-name))
	 (do0
	  
	  (imports (	os
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
					datetime
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
			
					;(np jax.numpy)
					;(mpf mplfinance)

					;argparse
			)))
	 (do0
	  (imports-from (helium *)))
	 (imports-from (bs4 BeautifulSoup))
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
			     (- tz)))))

	 (do0
	  (setf url (string "https://www.tutti.ch/de/q/suche/?sorting=newest&page=1&query=m1+macbook"))

	  
	  )
	 (do0
	  (start_chrome ;:headless True
	   ))
	 (do0
	  (go_to url))
	 (wait_until (dot (Text (string "Ihre Privatsphäre ist uns wichtig"))
			  exists)
		     :timeout_secs 30)
	 (wait_until (dot (Button (string "Akzeptieren"))
			  exists)
		     :timeout_secs 30)
	 
	 (click (string "Akzeptieren"))
	 
	 
	 (for (i (range 10))
	      (press PAGE_DOWN))

	 (setf entries (find_all (S (string ".css-1yxgef8"))))
	 (setf res (list))
	 (for (e entries)
	      (setf html (e.web_element.get_attribute (string "innerHTML"))
		    soup (BeautifulSoup html (string "html.parser")))
	      #+nil
	      (print (soup.prettify))
	      #+nil ,(lprint "entry"
			     `((dot e
				    web_element
				    text)))
	      (setf spans (soup.find_all
			   (string "span")
			   (dict ((string class) (string "css-13qotfz")))))
	      (setf description
		    (dot (aref spans 0)
			 text))
	      (setf price (dot (aref spans 1)
			       text))
	      (setf title (dot (aref
				(soup.find_all
				 (string "div")
				 (dict ((string class) (string "css-1haxbqe")))
				 ) 0)
			       text))
	      (setf address_date (dot
				  (aref
				   (soup.find_all
				    (string "span")
				    (dict ((string class) (string "css-q2t89t"))))
				   0)
				  text))
	      (res.append (dictionary :title title
				      :description description
				      :price price
				      :address_date address_date)))
	 (setf df (pd.DataFrame res))
	 (df.to_csv (dot (string "tutti_{}.csv")
			 (format (dot (datetime.datetime.now)
				      (strftime (string "%Y%m%d_%H_%M_%S"))))))
	 (kill_browser)
	 #+ni (do0 
	       (setf prices (find_all (S (string ".css-13qotfz"))))
	       (for (p prices)
		    ,(lprint "price"
			     `((dot p
				    web_element
				    text)))))
	 ""))))

