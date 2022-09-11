(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "97_makemore")
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

  (let* ((notebook-name "makemore")
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
		 (comments "this file is based on https://github.com/fastai/course22/blob/master/05-linear-model-and-neural-net-from-scratch.ipynb")
					;"%matplotlib notebook"
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
		 (imports (	os
					;sys
				time
					;docopt
				pathlib
					;(np numpy)
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
					;selenium.webdriver ;.FirefoxOptions
				argparse
				torch
				))

		 (imports-from (torch tensor))



		 #-nil
		 (imports-from  (matplotlib.pyplot
				 plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid
				 legend figure gcf xlim ylim)
				)

		 )
		))
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
			     (- tz)))))

	 (setf start_time (time.time)
	       debug True)))

       (python
	(export
	 (do0 (setf parser (argparse.ArgumentParser))
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help required action) e
			`(parser.add_argument
			  (string ,short)
			  (string ,long)
			  :help (string ,help)
					;:required
			  #+nil
			  (string ,(if required
				       "True"
				       "False"))
			  :action ,(if action
				       `(string ,action)
				       "None"))))

	      (setf args (parser.parse_args)))))

       #+nil
       (python
	(export
	 (setf path (pathlib.Path (string "titanic")))
	 (unless (path.exists)
	   (do0
	    (imports (zipfile))
	    (dot zipfile
		 (ZipFile (fstring "/content/drive/MyDrive/{path}.zip"))
		 (extractall path))))
	 ))

       (python
	(export
	 (setf words (dot
		      (open (string "/home/martin/stage/cl-py-generator/example/97_makemore/source/names.txt")
			    (string "r"))
		      (read)
		      (splitlines)
		      ))
	 (aref words (slice "" 10)))
	)
       ,@(loop for e in `(min max)
	       collect
	       `(python
		 (export
		  (,e (for-generator (w words)
				     (len w)))
		  )))

       (python
	(do0
	 (comments "collect statistics for pairs of characters")
	 (setf b "{}")
	 (for (w words)
	      (setf chs (+ (list (string "<S>"))
			   ("list" w)
			   (list (string "<E>"))))
	      (for (bigram
		    (zip chs (aref chs (slice 1 ""))))
		   (setf (aref b bigram)
			 (+ (b.get bigram 0)
			    1))))))
       (python
	(export
	 (comments "show statistics sorted by frequency")
	 (sorted (b.items)
		 :key (lambda (kv)
			(* -1  (aref kv 1))))))
       (python
	(export
	 (setf character_set
	       (sorted ("list"
			(set
			 (dot (string "")
			      (join words))))))
	 (len character_set)))
       (python
	(export
	 (setf stoi (curly (for-generator ((ntuple i s)
					   (enumerate character_set))
					  "s:i")))
	 (setf (aref stoi (string "<S>")) 26
	       (aref stoi (string "<E>")) 27
	       )
	 stoi))
       (python
	(export
	 (comments "2d array is more convenient")
	 (setf number_tokens (len stoi))
	 (setf N (torch.zeros (tuple number_tokens
				     number_tokens)
			      :dtype torch.int32))
	 (for (w words)
	      (setf chs (+ (list (string "<S>"))
			   ("list" w)
			   (list (string "<E>"))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (incf (aref N ix1 ix2))))))
       (python
	(do0
	 (imshow N)))

       )))
  (sb-ext:run-program "/usr/bin/ssh"
		      `("c11"
			"cd ~/arch/home/martin/stage/cl-py-generator/example/97_makemore/source; nbdev_export")))



