(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "96_colab_fastai")
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

  (let* ((notebook-name "titanic_from_scratch")
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
					;"%matplotlib notebook"
		 #+nil(do0

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

			   ))


		 #+nil  (imports-from (selenium webdriver)
				      (selenium.webdriver.common.keys Keys)
				      (selenium.webdriver.support.ui WebDriverWait)
				      (selenium.webdriver.common.by By)
				      (selenium.webdriver.support expected_conditions)


				      )

		 
		 #+nil
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

       (python
	(export
	 (setf path (pathlib.Path (string "titanic")))
	 (unless (path.exists)
	   (imports (zipfile
		     kaggle))
	   (dot kaggle
		api
		(competition_download_cli (str path)))
	   (dot zipfile
		(ZipFile (fstring "{path}.zip"))
		(extractall path))
	   )))
       (python
	(export
	 (imports (torch
		   (np numpy)
		   (pd pandas)))
	 (setf line_char_width
	       140)
	 (np.set_print_options :linewidth line_char_width)
	 (torch.set_print_options :linewidth line_char_width
				  :sci_mode False
				  :edgeitems 7)
	 (pd.set_option (string "display_width")
			line_char_width)))
       (python
	(export
	 (setf df (pd.read_csv (/ path
				  (string "train.csv"))))
	 df)
	)
       (python
	(do0
	 (dot df
	      (isna)
	      (sum))))
       (python
	(export
	 (setf mode (dot df
			 (mode)
			 (aref iloc 0)))))
       (python
	(export
	 (dot df
	      (fillna modes
		      :inplace True))))
       (python
	(do0
	 (dot df
	      (isna)
	      (sum))))

       ))))



