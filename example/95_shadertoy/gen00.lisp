(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)

;; /usr/bin/pip3 install --user nbdev helium

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/95_shadertoy")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                            ,@rest)))))

  (let* ((cli-args `((:short "-p" :long "--password" :help "password" :required t)
		     (:short "-i" :long "--input" :help "input file" :required t)
		     (:short "-H" :long "--headless"  :help "enable headless modex" :action "store_true" :required nil)
		     (:short "-k" :long "--kill"  :help "kill browser at the end" :action "store_true" :required nil)
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-notebook
     :nb-file (format nil "~a/source/00_upload_shader.ipynb" *path*)
     :nb-code
     `(
       (python
	(export
	 "#|default_exp p00_upload_shader"))
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
		 (imports (		;os
					;sys
			   time
					;docopt
					;pathlib
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



		 (imports-from #+nil (matplotlib.pyplot
				      plot imshow tight_layout xlabel ylabel
				      title subplot subplot2grid grid
				      legend figure gcf xlim ylim)
			       (helium *))

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

	      (setf args (parser.parse_args)
		    ))))

       (python
	(export
	 ,(lprint "" `(args))
	 ,(lprint "start chrome" `(args.headless))
	 (setf url  (string "https://www.shadertoy.com/view/7t3cDs"))
	 ,(lprint "go to" `(url))
	 (start_chrome url
		       :headless args.headless)))
       ;;https://github.com/mherrmann/selenium-python-helium/blob/master/docs/cheatsheet.md
       ;; https://selenium-python-helium.readthedocs.io/_/downloads/en/latest/pdf/
       (python
	(export
	 ,(lprint "wait for cookie banner")
	 (wait_until (dot (Button (string "Accept"))
			  exists))
	 (click (string "Accept"))


	 ,(lprint "login with password")

	 (click (string "Sign In"))
	 (write (string "plops"))
	 (press TAB)
	 (write args.password ;:into (string "Password")
		)
	 (click (string "Sign In"))))

       (python
	(export
	 ,(lprint "clear text")
	 (setf cm (S (string "//div[contains(@class,'CodeMirror')]")))
	 (click cm)
	 ,(lprint "select all")
	 (press (+ CONTROL (string "a")))
	 ,(lprint "delete")
	 (press DELETE)
	 #+nil (do0
		("list"
		 (map (lambda (x)
			(press ARROW_UP))
		      (range 12)))
		("list"
		 (map (lambda (x)
			(press (+ SHIFT DELETE)))
		      (range 12))))


	 ,(lprint "load source from" `(args.input))
	 (with (as (open args.input)
		   f)
	       (setf s (f.read))
	       )
	 ,(lprint "update the text" `(s))
	 (write s)
	 #+nil
	 (write (rstring3 "void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.1+ 0.5*cos(iTime+uv.xyx+vec3(1,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}"))
	 ))

       (python
	(export
	 ,(lprint "compile code")
	 (click (S (string "#compileButton")))
	 ,(lprint "save")
	 (click (string "Save"))
	 ,(lprint "wait for save to finish")
	 (wait_until (dot (Button (string "Save"))
			  exists))))
       (python
	(export
	 (when args.kill
	   (do0
	    ,(lprint "close browser")
	    (kill_browser)))))


       )))
  )



