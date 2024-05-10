(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "train_llm")
  (defparameter *idx* "02")
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

  
  
  (let* ((notebook-name "create_short_examples")
	 (cli-args `(#+nil (:short "c" :long "chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     #+nil (:short "p" :long "prompt" :type str
		      :default (string "Summarize the following video transcript as a bullet list.")
		      :help "The prompt to be prepended to the output file(s).")))
	 )
    (write-source
     (format nil "~a/source02/p~a_~a" *path* *idx* notebook-name)
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

       ,(let ((codes
	       `((:name 01_plot :code (do0
		   (imports (sys
			     (plt matplotlib.pyplot)
			     (np numpy)
			     (pd pandas)
			     pathlib))
		   (plt.ion)
		   (setf x (np.linspace 0 2.0 30)
			 y (np.sin x))
		   (plt.plot x y)
		   (plt.grid)))
		 (:name 02_qt :code (do0
	 (imports (sys
		   os
		   random
		   matplotlib))
	 (matplotlib.use (string "Qt5Agg"))
	 (imports ((qw PySide2.QtWidgets)
		   (qc PySide2.QtCore)
		   (np numpy)
		   (pd pandas)
		   pathlib
		   (agg matplotlib.backends.backend_qt5agg)
		   (mf matplotlib.figure)))
	 (class PlotCanvas (agg.FigureCanvasQTAgg)
		(def __init__ (self &key
				    (parent None)
				    (width 5)
				    (height 4)
				    (dpi 100))
		  (setf fig (mf.Figure
			     :figsize (tuple width height)
			     :dpi dpi)
			self.axes (fig.add_subplot 111))
		  (self.compute_initial_figure)

		  (agg.FigureCanvasQTAgg.__init__ self fig)
		  (self.setParent parent)
		  (agg.FigureCanvasQTAgg.setSizePolicy
		   self qw.QSizePolicy.Expanding
		   qw.QSizePolicy.Expanding)
		  (agg.FigureCanvasQTAgg.updateGeometry self))
		(def compute_initial_figure (self)
		  pass))
	 (class StaticCanvas (PlotCanvas)
		(def compute_initial_figure (self)
		  (setf t (np.arange 0 3 .01)
			s (np.sin (* 2 np.pi t))
			)
		  (self.axes.plot t s)))
	 (class DynamicCanvas (PlotCanvas)
		(def __init__ (self *args **kwargs)
		  (PlotCanvas.__init__ self *args **kwargs)
		  (setf timer (qc.QTimer self))
		  (timer.timeout.connect self.update_figure)
		  (timer.start 1000))
		(def compute_initial_figure (self)
		  (self.axes.plot (list 0 1 2 3)
				  (list 1 2 0 4)
				  (string "r")))
		(def update_figure (self)
		  (setf l (list))
		  (for (i (range 4))
		       (l.append (random.randint 0 10)))
		  (self.axes.cla)
		  (self.axes.plot (list 0 1 2 3)
				  l
				  (string "r"))
		  (self.draw)))
	 (class ApplicationWindow (qw.QMainWindow)
		(def __init__ (self)
		  (qw.QMainWindow.__init__ self)
		  (self.setAttribute qc.Qt.WA_DeleteOnClose)
		  (setf self.main_widget (qw.QWidget self)
			l (qw.QVBoxLayout self.main_widget)
			sc (StaticCanvas self.main_widget
					 :width 5
					 :height 4
					 :dpi 100)
			dc (DynamicCanvas self.main_widget
					 :width 5
					 :height 4
					 :dpi 100))
		  (l.addWidget sc)
		  (l.addWidget dc)
		  (self.main_widget.setFocus)
		  (self.setCentralWidget self.main_widget)
		  ))
	 (setf qApp (qw.QApplication sys.argv)
	       aw (ApplicationWindow)
	       )
	 (aw.show)
	 (sys.exit (qApp.exec_))
	 ))
		 (:code (do0
       (do0
	)
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
	  (start_chrome; :headless True
	   ))
	 (do0
	  (go_to url))
	 (try
	  (do0
	   (wait_until (dot (Text (string "Ihre Privatsphäre ist uns wichtig"))
			    exists)
		       :timeout_secs 30)
	   (wait_until (dot (Button (string "Akzeptieren"))
			    exists)
		       :timeout_secs 30))
	  ("Exception as e"
	   (print e)
	   (try 
	    (do0
	     (wait_until (dot (Button (string "Verify you are human"))
			      exists))
	     (click (string "Verify you are human"))
	     
	     (wait_until (dot (Text (string "Ihre Privatsphäre ist uns wichtig"))
			      exists)
			 :timeout_secs 30)
	     (wait_until (dot (Button (string "Akzeptieren"))
			      exists)
			 :timeout_secs 30))
	    ("Exception as e"
	     (print e)
	     (try 
	    (do0
	     
	     (click (string "Verify you are human"))
	     
	     (wait_until (dot (Text (string "Ihre Privatsphäre ist uns wichtig"))
			      exists)
			 :timeout_secs 30)
	     (wait_until (dot (Button (string "Akzeptieren"))
			      exists)
			 :timeout_secs 30))
	    ("Exception as e"
	     (print e)
	     (try 
	    (do0
	     
	     (click (string "Verify you are human"))
	     
	     (wait_until (dot (Text (string "Ihre Privatsphäre ist uns wichtig"))
			      exists)
			 :timeout_secs 30)
	     (wait_until (dot (Button (string "Akzeptieren"))
			      exists)
			 :timeout_secs 30))
	    ("Exception as e"
	     (print e)
	     (do0
	      (setf png (dot (string "tutti_{}.png")
			     (format (dot (datetime.datetime.now)
					  (strftime (string "%Y%m%d_%H_%M_%S"))))))
	      (dot (get_driver)
		   (save_screenshot png))
	      ,(lprint :msg "store screenshot" :vars `(png))
	      (kill_browser)
	      (exit))))))))
	   ))
	 
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
	 (do0
	  (setf csv (dot (string "tutti_{}.csv")
			  (format (dot (datetime.datetime.now)
				       (strftime (string "%Y%m%d_%H_%M_%S"))))))
	  (df.to_csv csv)
	  ,(lprint :msg "store" :vars `(csv)))
	 (kill_browser)
	 #+ni (do0 
	       (setf prices (find_all (S (string ".css-13qotfz"))))
	       (for (p prices)
		    ,(lprint "price"
			     `((dot p
				    web_element
				    text)))))
	 ""))

		 (:code
		  (do0
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
			requests

					;(np jax.numpy)
					;(mpf mplfinance)

					;argparse
			)))
	 (do0
	  (imports-from (bs4 BeautifulSoup)))
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
	  (setf response (requests.get url)
		soup (BeautifulSoup response.text
				    (string "html.parser"))
		listings (soup.find_all (string "div")
					:class_ (string "1yxgef8")))
	  ))
		  )
		 (:code
		  ((python
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
	 (comments "For now I only want videos with names like this:")
	 (comments "TWiV 908: COVID-19 clinical update #118 with Dr. Daniel Griffin")
	 (setf df2 (aref df
			 (df.text.str.contains
			  (rstring3 "TWiV .*: COVID-19 clinical update .* with Dr. Daniel Griffin"))))
	 (df2.to_csv (string "links_covid_update.csv"))
	 ))
       (python
	(export
	 (kill_browser)))

       (python
	(do0
	 (setf df (pd.read_csv (string "links_covid_update.csv")
			       :index_col 0))
	 (setf (aref df (list (string "twiv_nr")
			      (string "covid_update_nr")
			      ))
	       ;; TWiV 901: COVID-19 clinical update #115 with Dr. Daniel Griffin
	       (df.text.str.extract
		(rstring3 "TWiV\\ (\\d+):\\ COVID-19\\ clinical\\ update\\ #(\\d+)\\ with\\ Dr\\.\\ Daniel\\ Griffin")))
	 (df.to_csv (string "links_covid_update_parsed.csv"))))

       )
		  ))))

	  (loop for e in codes
		   and e-i from 0
		   do
		      (destructuring-bind (&key name code) e
			(if (listp code)
			    (cond
			      ((eq (first code) 'do0)
			       (write-source (format nil "/dev/shm/~3,'0d" e-i)
					     code))
			      ((eq (caar code) 'python)
				;; this used to be a notebook, containing:
			       #+nil ((python (export "<string>"))
				      (python (export (do0 ...)))
				      (python (do0 ...))
				      ...)
			       (write-source (format nil "/dev/shm/~3,'0d" e-i)
					     `(do0
					       ,@(loop for python-sexpr in code
						       collect
						       (cond ((eq 'export (car (second python-sexpr)))
							      (second (second python-sexpr)))
							     ((eq 'do0 (car (second python-sexpr)))
							      (second python-sexpr))
							     (t
							      (break "unsupported notebooks structure. i want (python (export ...: ~a" python-sexpr))))))
			       
			       )
			      (t (break "strange"))
			      )
			    (break "code should be a list starting with (do0 ... or (python ...")
			    ))))       
       
       )))

  )
