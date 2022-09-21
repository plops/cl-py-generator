(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "98_yt_audio_to_text")
  (defparameter *idx* "00")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                            ,@rest)))))

  (let* ((notebook-name "get_links")
	 (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-notebook
     :nb-file (format nil "~a/source/~a_~a.ipynb" *path* *idx* notebook-name)
     :nb-code
     `((python
	(export
	 ,(format nil "#|default_exp p~a_~a" *idx* notebook-name)))
       (python (export
		(do0
		 (comments "use helium to get youtube video links")
		 (imports (	os
					;sys
				time
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

					;(np jax.numpy)
					;(mpf mplfinance)
					
				argparse
				)))))
       (python
	(export
	 (imports-from (helium *))))
       (python
	(export
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
			     (- tz)))))))

       (python
	(export
	 (start_firefox
	  ;:headless True
	  )
	 
	 
	 ))
       (python
	(export
	 (go_to (string "https://www.youtube.com/c/VincentRacaniello/videos"))
	 ))
       (python
	(export
	 (comments "deal with the cookie banner")
	 (wait_until (dot (Button (string "REJECT ALL"))
			  exists))
	 (click (string "REJECT ALL"))))
       (python
	(export
	 (comments "infinite scrolling to make all the links visible")
	 (for (i (range 120))
	      (press PAGE_DOWN))))
       (python
	(export
	 (comments "extract all the links")
	 (setf
	  links
	  (find_all (S (string "a.ytd-grid-video-renderer"))))))
       (python
	(export
	 (comments "store links in pandas  table")
	 (setf res (list))
	 (for (l links)
	      (setf text l.web_element.text
		    location l.web_element.location
		    href (l.web_element.get_attribute (string "href")))
	      (res.append
	       (dictionary :text text
			   :location location
			   :href href)))))
       (python
	(export
	 (comments "store pandas table in csv file")
	 (setf df (pd.DataFrame res))
	 (df.to_csv (string "links.csv"))))
       (python
	(export
	 (kill_browser)))))))

