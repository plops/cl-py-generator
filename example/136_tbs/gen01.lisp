(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "136_tbs")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
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
  (defun doc (def)
    `(do0
      ,@(loop for e in def
	      collect
	      (destructuring-bind (&key name val (unit "-") (help name)) e
		`(do0
		  (comments ,(format nil "~a (~a)" help unit))
		  (setf ,name ,val))))))
  
  (let* ((notebook-name "scrape")
	 (cli-args `((:short "s" :long "socks_port" :type int :default 9150
		      :help "SOCKS port.")
		     (:short "c" :long "control_port" :type int :default 9151
		      :help "Control port.")
		     (:short "b" :long "browser-path" :type str
		      :default (string "/")
		      :help "Path to browser.")
		     (:short "u" :long "url" :type str
		      :default (string "http://news.ycombinator.com/")
		      :help "URL to scrape."))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (do0
	#+nil
	(do0
	 
	 (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
	 (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation)
					;(xrp xarray.plot)
		   ))

					;(plt.ion)
	 (plt.ioff)
	 ;;(setf font (dict ((string size) (string 6))))
	 ;; (matplotlib.rc (string "font") **font)
	 )
	#+nil 
	(imports-from  (matplotlib.pyplot
			plot imshow tight_layout xlabel ylabel
			title subplot subplot2grid grid text
			legend figure gcf xlim ylim)
		       )
	(imports (			;	os
					;sys
		  time
					;docopt
					;pathlib
					; (np numpy)
					;serial
					;(pd pandas)
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
					;(np numpy)
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
					
					;math
			
					;(np jax.numpy)
					;(mpf mplfinance)
					;(fft scipy.fftpack)
		  argparse
					;torch
					;(mp mediapipe)
					;mss
		  ;(cv cv2)
		  ;tiktoken
		  )))

       (imports-from (tbselenium.tbdriver TorBrowserDriver))
       
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

       

       (do0 ;when (== __name__ (string "__main__"))
	 (do0 (setf parser (argparse.ArgumentParser :description (string "Scrape url.")))
	      
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help type default required action nb-init) e
			`(parser.add_argument
			  (string ,(format nil "-~a" short))
			  (string ,(format nil "--~a" long))
			  :help (string ,help)
					;:required
			  #+nil
			  (string ,(if required
				       "True"
				       "False"))
			  :default ,(if default
					default
					"None")
			  :type ,(if type
				     type
				     "None")
			  :action ,(if action
				       `(string ,action)
				       "None"))))

	      (setf args (parser.parse_args))
	      (with (as (TorBrowserDriver args.browser_path :socks_port args.socks_port
					  :control_port args.control_port)
			driver)
		    (driver.get args.url))))
       ))))

