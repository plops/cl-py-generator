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
		  )


		 (:code
		  ((python
	(export
	 ,(format nil "#|default_exp p~a_~a" *idx* notebook-name)))
       (python (export
		(do0
					;(comments "this file is based on ")
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
				tqdm
				argparse
				torch
				))
		 "import torch.nn.functional as F"
		 (imports-from (torch tensor))



		 #-nil
		 (imports-from  (matplotlib.pyplot
				 plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid text
				 legend figure gcf xlim ylim)
				)

		 (imports-from  (torch
				 linspace
				 randn
				 randint
				 tanh
				 )
				)

		 )
		))

       (python
	(do0
	 (class Args ()
		(def __init__ (self)
		  ,@(loop for e in cli-args
			  collect
			  (destructuring-bind (&key short long help required action nb-init) e
			    `(setf (dot self ,long) ,nb-init)))))
	 (setf args (Args))))
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
	       debug True)
	 ))




       (python
	(export
	 (do0 (setf parser (argparse.ArgumentParser))
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help required action nb-init) e
			`(parser.add_argument
			  (string ,(format nil "-~a" short))
			  (string ,(format nil "--~a" long))
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
	 (setf words (dot
		      (open (string "/home/martin/stage/cl-py-generator/example/97_makemore/source/names.txt")
			    (string "r"))
		      (read)
		      (splitlines)
		      ))
	 (aref words (slice "" 10)))
	)
       (python
	(export
	 (setf chars (sorted ("list" (set (dot (string "")
					       (join words))))))
	 (setf stoi (curly (for-generator ((ntuple i s)
					   (enumerate chars))
					  (slice s (+ i 1))))
	       (aref stoi (string ".")) 0
	       itos (curly (for-generator ((ntuple s i)
					   (stoi.items))
					  (slice i s))))
	 (print itos)))
       ,@(let* ((block-size 3)
		(n2 2)
		(n100 100) ;; hidden layer
		(n6 (* n2 block-size))
		(n27 27))
	   `((python
	      (export
	       (setf block_size ,block-size
		     X (list)
		     Y (list))
	       (for (w words ;(aref words (slice "" 5))
		       )
		    (print w)
		    (setf context (* (list 0)
				     block_size))
		    (for (ch (+ w (string ".")))
			 (setf ix (aref stoi ch))
			 (X.append context)
			 (Y.append ix)
			 #+nil (print (dot (string "")
					   (join (for-generator (i context)
								(aref itos i))))
				      (string "--->")
				      (aref itos ix))
			 (setf context (+ (aref context (slice 1 ""))
					  (list ix)))))
	       (setf X (tensor X)
		     Y (tensor Y))))
	     (python
	      (do0
	       (setf C (torch.randn
			(tuple ,n27 ,n2)))))
	     (python
	      (do0
	       (@
		(dot (F.one_hot (tensor 5)
				:num_classes ,n27)
		     (float))
		C)
	       ))
	     (python
	      (do0
	       (setf emb (aref C X))
	       (do0
		(setf W1 (torch.randn (tuple ,n6 ,n100))
		      b1 (torch.randn (tuple ,n100)))
		(setf W2 (torch.randn (tuple ,n100 ,n27))
		      b2 (torch.randn (tuple ,n27))))

	       ))
	     (python
	      (do0
	       (setf h
		     (torch.tanh
		      (+ (@ (dot emb
				 (view -1 ,n6)) W1)
			 b1)))))
	     (python
	      (do0
	       (setf logits (+ (@ h W2)
			       b2)
		     counts (logits.exp)
		     prob (/ counts
			     (counts.sum 1 :keepdims True)))

	       ))
	     (python
	      (do0
	       (comments "negative log-likelihood loss")
	       (setf loss
		     (* -1
			(dot (aref prob
				   (torch.arange 32)
				   Y)
			     (log)
			     (mean))))))
	     (python
	      (export
	       (do0
		(setf g (dot torch
			     (Generator)
			     (manual_seed 2147483647))
		      C (torch.randn (tuple ,n27 ,n2)
				     :generator g))
		(do0
		 (setf W1 (torch.randn (tuple ,n6 ,n100))
		       b1 (torch.randn (tuple ,n100)))
		 (setf W2 (torch.randn (tuple ,n100 ,n27))
		       b2 (torch.randn (tuple ,n27))))
		(setf parameters (list C W1 b1 W2 b2))
		(for (p parameters)
		     (setf p.requires_grad True))
		(do0 (setf n_parameters (sum (for-generator (p parameters)
							    (p.nelement))))
		     ,(lprint :vars `(n_parameters)))
		)))
	     (python
	      (do0

	       (do0
		(comments "forward")
		(setf emb (aref C X))
		(setf h
		      (torch.tanh
		       (+ (@ (dot emb
				  (view -1 ,n6)) W1)
			  b1)))
		(setf logits (+ (@ h W2)
				b2))
		(setf loss (F.cross_entropy logits Y)))
	       #+nil
	       (do0
		(comments "explicit cross_entropy is not efficient a lot of tensors: ")
		(comments "positive numbers can overflow logits, offset doesn't influence loss")
		(setf logits2 (- logits (torch.max logits))
		      counts (logits2.exp)
		      prob (/ counts
			      (counts.sum 1 :keepdims True)))
		(setf loss
		      (* -1
			 (dot (aref prob
				    (torch.arange 32)
				    Y)
			      (log)
			      (mean)))))
	       (do0
		(comments "backward pass")
		(comments "set gradients to zero")
		(for (p parameters)
		     (setf p.grad None))
		(loss.backward)
		(comments "update")
		(for (p parameters)
		     (incf p.data
			   (* -.1 p.grad)))
		)
	       ))

	     (python
	      (do0
	       (comments "learning loop")

	       (for (_ (range 10))
		    (do0
		     (do0
		      (comments "forward")
		      (do0 (setf emb (aref C X))
			   (setf h
				 (torch.tanh
				  (+ (@ (dot emb
					     (view -1 ,n6)) W1)
				     b1)))
			   (setf logits (+ (@ h W2)
					   b2))
			   (setf loss (F.cross_entropy logits Y))
			   ,(lprint :vars `((loss.item)))))

		     (do0
		      (comments "backward pass")
		      (for (p parameters)
			   (setf p.grad None))
		      (loss.backward)
		      (comments "update")
		      (for (p parameters)
			   (incf p.data
				 (* -.1 p.grad)))
		      )))
	       ))

	     (python
	      (export
	       (comments "learn with minibatch")
	       (for (_ (range 10_000))
		    (do0
		     (do0
		      (setf ix (torch.randint 0 (aref X.shape 0)
					      (tuple 32)))
		      (comments "forward")
		      (setf emb (aref C (aref X ix)))
		      (setf h
			    (torch.tanh
			     (+ (@ (dot emb
					(view -1 ,n6)) W1)
				b1)))
		      (setf logits (+ (@ h W2)
				      b2))
		      (setf loss (F.cross_entropy logits (aref Y ix)))
		      ,(lprint :vars `((loss.item))))

		     (do0
		      (comments "backward pass")
		      (for (p parameters)
			   (setf p.grad None))
		      (loss.backward)
		      (comments "update")
		      (for (p parameters)
			   (incf p.data
				 (* -.1 p.grad)))
		      )))
	       (do0 (comments "report loss on entire data set")
		    (setf emb (aref C X))
		    (setf h
			  (torch.tanh
			   (+ (@ (dot emb
				      (view -1 ,n6)) W1)
			      b1)))
		    (setf logits (+ (@ h W2)
				    b2))
		    (setf loss_full (F.cross_entropy logits Y))
		    ,(lprint :vars `((loss_full.item))))
	       ))

	     (python
	      (do0
	       (comments "find a good learning rate, start with a very small lr and increase exponentially")
	       (setf lre (linspace -3 0 1000)
		     lre (** 10 lre))
	       (setf lri (list)
		     lossi (list))
	       (for (lr (tqdm.tqdm lre))
		    (do0
		     (do0
		      (setf ix (torch.randint 0 (aref X.shape 0)
					      (tuple 32)))
		      (comments "forward")
		      (setf emb (aref C (aref X ix)))
		      (setf h
			    (torch.tanh
			     (+ (@ (dot emb
					(view -1 ,n6)) W1)
				b1)))
		      (setf logits (+ (@ h W2)
				      b2))
		      (setf loss (F.cross_entropy logits (aref Y ix)))
					; ,(lprint :vars `((loss.item)))
		      )

		     (do0
		      (comments "backward pass")
		      (for (p parameters)
			   (setf p.grad None))
		      (loss.backward)
		      (comments "update")
		      (for (p parameters)
			   (incf p.data
				 (* -1 lr p.grad)))

		      (comments "track stats")
		      (lri.append lr)
		      (lossi.append (loss.item))
		      )))

	       ))
	     (python
	      (do0
	       (plot lre lossi
		     :alpha .4)
	       (grid)))
	     ))

       )
		  
		  )
		 (:code

		  ((python
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
		 "import torch.nn.functional as F"
		 (imports-from (torch tensor))



		 #-nil
		 (imports-from  (matplotlib.pyplot
				 plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid text
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
					  "s:i+1")))
	 #+nil(setf (aref stoi (string "<S>")) 26
		    (aref stoi (string "<E>")) 27
		    )
	 (setf (aref stoi (string ".")) 0

	       )
	 stoi))
       (python
	(export
	 (comments "invert lookup")
	 (setf itos (curly
		     (for-generator ((ntuple s i)
				     (stoi.items))
				    "i:s")
		     ))))
       (python
	(export
	 (comments "2d array is more convenient")
	 (setf number_tokens (len stoi))
	 (setf N (torch.zeros (tuple number_tokens
				     number_tokens)
			      :dtype torch.int32))
	 (for (w words)
	      (setf chs (+ (list (string "."))
			   ("list" w)
			   (list (string "."))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (incf (aref N ix1 ix2))))))
       (python
	(do0
	 (imshow N)))

       (python
	(do0
	 (figure :figsize (tuple 16 16))
	 (imshow N :cmap (string "Blues"))
	 (for (i (range number_tokens))
	      (for (j (range number_tokens))
		   (setf chstr (+ (aref itos i)
				  (aref itos j)))
		   (text j i chstr :ha (string "center")
			 :va (string "bottom")
			 :color (string "gray"))
		   (text j i (dot (aref N i j)
				  (item))
			 :ha (string "center")
			 :va (string "top")
			 :color (string "gray"))))
	 (plt.axis (string "off"))
	 ))

       (python
	(do0
	 (setf p (dot (aref N 0)
		      (float))
	       p (/ p (p.sum)))
	 p))

       (python
	(do0
	 (setf g (dot torch
		      (Generator)
		      (manual_seed 2147483647))
	       p (dot torch (rand 3 :generator g))
	       p (/ p (p.sum)))))
       (python
	(do0
	 (torch.multinomial p
			    :num_samples 20
			    :replacement True
			    :generator g)))

       (python
	(do0
	 (comments "https://pytorch.org/docs/stable/notes/broadcasting.html")
	 (comments "adding one for model smoothing (we don't want zeros in the matrix)")
	 (setf P (dot (+ N 1)  (float))
	       P (/ P
		    (P.sum 1 :keepdim True)))
	 ))

       (python
	(do0
	 (setf log_likelihood 0.0
	       n 0)
	 (for (w (list (string "andrej")))
	      (setf chs (+ (list (string "."))
			   ("list" w)
			   (list (string "."))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (setf prob (aref P ix1 ix2)
			 logprob (torch.log prob))
		   (incf log_likelihood logprob)
		   (incf n)
		   (comments "everything with probability higher than 4% is better than random")
		   (print (fstring "{ch1}{ch2}: {prob:.4f} {logprob:.4f}"))))
	 (print (fstring "{log_likelihood=}"))
	 (comments "we are intersted in the product of all probabilities. this would be a small number so we look at the log")
	 (comments "look at negative log_likelihood. the lowest we can get is 0")
	 (setf nll (* -1 log_likelihood))
	 (print (fstring "{nll=}"))
	 (comments "normalized log likelihood is what we use")
	 (comments "normalized log likelihood of the training model is 2.454")
	 (print (fstring "{nll/n:.3f}"))
	 ))

       (python
	(do0
	 (setf xs (list)
	       ys (list))
	 (for (w (aref words (slice "" 1)))
	      (setf chs (+ (list (string "."))
			   ("list" w)
			   (list (string "."))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (print ch1 ch2)
		   (dot xs (append ix1))
		   (dot ys (append ix2))))
	 (setf xs (tensor xs)
	       ys (tensor ys))

	 (comments "encode integers with one-hot encoding")
	 (setf xenc (dot (F.one_hot xs :num_classes number_tokens)
			 (float)))
	 (imshow xenc)
	 ))

       (python
	(do0
	 (do0
	  (setf g (dot
		   torch
		   (Generator)
		   (manual_seed 2147483647)))
	  (setf W (torch.randn (tuple 27 27)
			       :generator g
			       :requires_grad True)))))
       (python
	(do0




	 (comments "output is 5x27 @ 27x27 = 5x27"
		   "27 neurons on 5 inputs"
		   "what is the firing rate of the 27 neurons on everyone of the 5 inputs"
		   "xenc @ W [3,13] indicates the firing rate of the 13 neuron for input 3. it is a dot-product of the 13th column of W with the input xenc"
		   "we exponentiate the numbers. negative numbers will be 0..1, positive numbers will be >1"
		   "we will interpret them as something equivalent to count (positive numbers). this is called logits. equivalent to the counts in the N matrix"
		   "converting logits to probabilities is called softmax")

	 (comments "the closer values in W the closer the probabilities to equal"
		   "you can regularize by forcing W to be closer to zero ... W**2 term in loss")

	 (setf logits (@ xenc W)
	       counts (dot logits (exp))
	       probs (/ counts
			(counts.sum 1 :keepdims True)))
	 probs
	 (setf loss
	       (* -1 (dot
		      (aref probs
			    (torch.arange 5)
			    ys)
		      (log)
		      (mean))))
	 (print (loss.item))
	 (comments "this is the forward pass")
	 )
	)
       (python
	(do0
	 (comments "backward pass"
		   "clear gradient")
	 (setf W.grad None)
	 (loss.backward)
	 (incf W.data (* -.1 W.grad))
	 (comments "gradient descent gives exactly the same model. sampling will be the same as the frequency counter")
	 ))

       )
		  
		  )

		 (:code
		  ((python
	(export
	 ,(format nil "#|default_exp p~a_~a" *idx* notebook-name)))
       (python (export
		(do0
		 (comments "this file is based on https://github.com/fastai/course22/blob/master/05-linear-model-and-neural-net-from-scratch.ipynb")
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
				torch
				))

		 (imports-from (torch tensor))
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
	 (comments "i want to run this on google colab. annoyingly i can't seem to access the titanic.zip file. it seems to be necessary to supply some kaggle login information in a json file. rather than doing this i downloaded the titanic.zip file into my google drive")
	 (imports (google.colab.drive))
	 (google.colab.drive.mount (string "/content/drive"))

	 ))

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
	 (np.set_printoptions :linewidth line_char_width)
	 (torch.set_printoptions :linewidth line_char_width
				 :sci_mode False
				 :edgeitems 7)
	 (pd.set_option (string "display.width")
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
	 (setf modes (dot df
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

       (python
	(do0
	 (dot df
	      (describe :include (tuple np.number)))))
       (python
	(do0
	 (df.Fare.hist)))
       (python
	(export
	 (setf (aref df (string "LogFare"))
	       (np.log
		(+ 1 df.Fare)))))
       (python
	(do0
	 (comments "histogram of logarithm of prices no longer shows the 'long' tail")
	 (df.LogFare.hist)))
       (python
	(do0
	 (comments "look at the three values that are in passenger class. more details about the dataset are here: https://www.kaggle.com/competitions/titanic/data")
	 (setf pclasses (sorted (df.Pclass.unique)))
	 pclasses))
       (python
	(do0
	 (comments "look at columns with non-numeric values")
	 (df.describe :include (list object))))
       (python
	(export
	 (comments "replace non-numeric values with numbers by introducing new columns (dummies). The dummy columns will be added to the dataframe df and the 3 original columns are dropped."
		   "Cabin, Name and Ticket contain too many unique values for this approach to be useful")
	 (setf df (pd.get_dummies
		   df
		   :columns (list (string "Sex")
				  (string "Pclass")
				  (string "Embarked"))))
	 df.columns))
       (python
	(do0
	 (comments "look at the new dummy columns")
	 (setf added_columns (list ,@(loop for e in
					   `(Sex_male Sex_female
						      Pclass_1 Pclass_2 Pclass_3
						      Embarked_C Embarked_Q Embarked_S
						      )
					   collect
					   `(string ,e))))
	 (dot (aref df added_columns)
	      (head))))
       (python
	(export
	 (comments "create dependent variable as tensor")
	 (setf t_dep (tensor df.Survived))))
       (python
	(export
	 (comment "independent variables are all continuous variables of interest and the newly created columns")
	 (setf indep_columns
	       (+ (list ,@(loop for e in `(Age SibSp Parch LogFare)
				collect
				`(string ,e)))
		  added_columns))
	 (setf t_indep (tensor (dot (aref df indep_columns)
				    values)
			       :dtype torch.float)
	       )
	 t_indep))
       (python
	(do0
	 t_indep.shape))
       (python
	(do0
	 (comments "set up linear model. first we calculate manually a single step for the loss of every row in the dataset. we start with a random coefficient in (-.5,.5) for each column of t_indep")
	 (torch.manual_seed 442)
	 (setf n_coeffs (dot t_indep
			     (aref shape 1)))
	 (setf coeffs (- (dot torch
			      (rand n_coeffs))
			 .5))
	 coeffs))
       (python
	(do0
	 (comments "our predictions are formed by multiplying a row with coefficients and summing them up. we don't need to introduce a bias (or intercept) term by introducing a column containing only ones. Such a 'one' is already present in each row in either the dummy column Sex_male or Sex_female.")
	 (* t_indep coeffs)
	 ))
       (python
	(do0
	 (comments "we have a potential problem with the first column Age. Its values are bigger in average than the values in other columns"
		   "In the lecture Jeremy mentions two options to normalize Age I can think of two more methods: 1) divide by maximum or 2) subtract mean and divide by std 3) subtract median and divide by MAD 4) find lower 2 perscentile and upper 2 percentile increase the value gap by +/- 10% and use this interval to normalize the input values. In the book jeremy uses 1). 1) and 3) differ by how they handle outliers. The maximum will be influenced a lot by outliers. I would like to know if 3) is better than 1) for typical problems. I think that boils down to how big the training dataset is. Once it is big enough there may be always enough outliers to ensure even the maximum is stable.")
	 (when True
	   (do0
	    (comments "method 1)")
	    (setf (ntuple vals indices)
		  (t_indep.max :dim 0))
	    (setf t_indep (/ t_indep
			     vals))))
	 (when False
	   (do0
	    (comments "method 2)")
	    (setf (ntuple means indices1)
		  (t_indep.mean :dim 0))
	    (setf (ntuple stdts indices2)
		  (t_indep.std :dim 0))
	    (setf t_indep (/ (- t_indep
				means)
			     stds))))))

       (python
	(do0
	 (comments "create predictions by adding up the rows of the product")
	 (setf preds  (dot (* t_indep
			      coeffs)
			   (sum :axis 1)))))
       (python
	(do0
	 (comments "look at first few")
	 (aref preds (slice "" 10))
	 (comments "as the coefficents were random these predictions are no good")))
       (python
	(do0
	 (comments "in order to improve the predictions modify the coefficients with gradient descent"
		   "define the loss as the average error between predictions and the dependent")
	 (setf loss (dot torch
			 (abs (- preds
				 t_dep))
			 (mean)))
	 loss))
       (python
	(export
	 (comments "using what we learned in the previous cells create functions to compute predictions and loss")
	 (def calc_preds (&key coeffs indeps)
	   (return
	     (dot (* indeps
		     coeffs)
		  (sum :axis 1))))
	 (def calc_loss (&key coeffs indeps deps)
	   (setf preds (calc_preds :coeffs coeffs
				   :indeps indeps))
	   (setf loss (dot torch
			   (abs (- preds
				   deps))
			   (mean)))
	   (return loss))
	 ))
       (python
	(do0
	 (comments "perform a single 'epoch' of gradient descent manually"
		   "tell pytorch that we want to calculate the gradients for the coeffs object. the underscore indicates that the coeffs object will be modified in place")
	 (dot coeffs
	      (requires_grad_))

	 ))
       (python
	(do0
	 (comments "compute the loss, pytorch will perform book keeping to compute gradients later")

	 (setf loss (calc_loss :coeffs coeffs
			       :indeps t_indep
			       :deps t_dep))))
       (python
	(do0
	 (comments "compute gradient")
	 (loss.backward)

	 coeffs.grad
	 (comments "note that every call of backward() adds the gradients to grad")))
       (python
	(do0
	 (comments "calling the steps a second time will double the values in .grad")
	 (do0 (setf loss (calc_loss :coeffs coeffs
				    :indeps t_indep
				    :deps t_dep))
	      (loss.backward)
	      coeffs.grad)))
       (python
	(do0
	 (comments "we can now perform a single gradient step. the loss should reduce")
	 (do0 (setf loss (calc_loss :coeffs coeffs
				    :indeps t_indep
				    :deps t_dep))
	      (loss.backward)
	      (with (torch.no_grad)
		    (do0
		     (dot coeffs
			  (sub_
			   (* coeffs.grad
			      .1)))
		     (dot coeffs
			  grad
			  (zero_)))
		    (print (calc_loss :coeffs coeffs
				      :indeps t_indep
				      :deps t_dep)))
	      )
	 (comments "a.sub_(b) subtracts the gradient from coeffs in place (a = a - b) and zero_ clears the gradients")))

       (python
	(export
	 (comments "before we can perform training, we have to create a validation dataset"
		   "we do that in the same way as the fastai library does")
	 (imports (fastai.data.transforms))
	 (comments "get training (trn) and validation indices (val)")
	 (setf (ntuple trn
		       val)
	       ((fastai.data.transforms.RandomSplitter
		 :seed 42
		 )
		df))
	 ))
       (python
	(export
	 ,@(loop for e in `(indep dep)
		 collect
		 `(do0
		   ,@(loop for f in `(trn val)
			   collect
			   `(setf ,(format nil "~a_~a" f e)
				  (aref ,(format nil "t_~a" e)
					,f)))))
	 (ntuple (len trn_indep)
		 (len val_indep))))
       (python
	(export
	 (comments "create 3 functions for the operations that were introduced in the previous cells")
	 (def update_coeffs (&key coeffs learning_rate)
	   (do0
	    (dot coeffs
		 (sub_
		  (* coeffs.grad
		     learning_rate)))
	    (dot coeffs
		 grad
		 (zero_))))
	 (def init_coeffs ()
	   (setf coeffs (- (dot torch
				(rand n_coeffs))
			   .5))
	   (coeffs.requires_grad_)
	   (return coeffs))
	 (def one_epoch (&key coeffs learning_rate)
	   (do0 (setf loss (calc_loss :coeffs coeffs
				      :indeps trn_indep
				      :deps trn_dep))
		(loss.backward)
		(with (torch.no_grad)
		      (update_coeffs :coeffs coeffs
				     :learning_rate learning_rate)
		      )

		(print (fstring "{loss:.3f}")
		       :end (string "; "))))))

       (python
	(export
	 (comments "now use these functions to train the model")
	 (def train_model (&key (epochs 30)
				(learning_rate .01))
	   (torch.manual_seed 442)
	   (setf coeffs (init_coeffs)
		 )
	   (for (i (range epochs)
		   )
		(one_epoch :coeffs coeffs
			   :learning_rate learning_rate))
	   (return coeffs))))
       (python
	(do0
	 (comments "try training. the loss should decrease")
	 (setf coeffs (train_model :epochs 18
				   :learning_rate .2))))
       (python
	(do0
	 (def show_coeffs ()
	   (return (dict
		    (zip indep_cols
			 (map
			  (lambda (x)
			    (x.item))
			  (coeffs.requires_grad_ False))))))
	 (show_coeffs)))

       (python
	(do0
	 (comments "the kaggle competition scores accuracy -- the proportion of rows where we correctly predict survival"
		   "determine accuracy using the validation set"
		   "first compute the predictions")
	 (setf preds (calc_preds :coeffs coeffs
				 :indeps val_indep))
	 (comments "for passenger with preds > 0.5 our model predicts survival. compare this with the dependent variable")
	 (setf results (== (val_dep.bool)
			   (> preds 0.5)))
	 (aref results (slice "" 16))
	 ))
       (python
	(do0
	 (comments "compute average accuracy")
	 (dot results
	      (float)
	      (mean))))
       (python
	(export
	 (comments "create a function to compute accuracy")
	 (def acc (coeffs)
	   (setf results (== (val_dep.bool)
			     (> preds 0.5)))
	   (return (dot results
			(float)
			(mean)
			(item))))
	 (print (dot (string "{:3.2f}")
		     (format (acc coeffs))))
	 ))
       (python
	(do0
	 (comments "some predictions are >1 and some are <0. We don't want that")
	 ))
       (python
	(do0
	 (imports (sympy))
	 (sympy.plot (string "1/(1+exp(-x))")
		     :xlim (tuple -7 7))
	 (comments "pytorch contains the sigmoid function")))
       (python
	(export
	 (def calc_preds (coeffs indeps)
	   (return (torch.sigmoid
		    (dot (* indeps
			    coeffs)
			 (sum :axis 1)))))))
       (python
	(export
	 (comments "train a new model now using the updated function to calculate predictions (that will always be in (0,1))")
	 (setf coeffs (train_model :learning_rate 100))

	 ))
       (python
	(do0
	 (acc coeffs)))

       (python
	(do0
	 (show_coeffs)
	 (comments "older people and males are less likely to survive. first class passengers are more likely to survive.")))


       (markdown "## Submitting to Kaggle")
       (python
	(do0
	 (comments "read the test set")
	 (setf tst_df (pd.read_csv (/ path
				      (string "test.csv"))))
	 (comments "clean up one missing Fare, set it to 0")
	 (setf (aref tst_df (string "Fare"))
	       (dot tst_df
		    Fare
		    (fillna 0)))))
       (python
	(do0
	 (comments "perform the same steps we did for the training set"
		   )))




       )
		  )

		 (:code
		  (
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
	 ,(lprint :msg "" :vars `(args))
	 ,(lprint :msg "start chrome" :vars `(args.headless))
	 (setf url  (string "https://www.shadertoy.com/view/7t3cDs"))
	 ,(lprint :msg "go to" :vars `(url))
	 (start_chrome url
		       :headless args.headless)))
       ;;https://github.com/mherrmann/selenium-python-helium/blob/master/docs/cheatsheet.md
       ;; https://selenium-python-helium.readthedocs.io/_/downloads/en/latest/pdf/
       (python
	(export
	 ,(lprint :msg "wait for cookie banner")
	 (wait_until (dot (Button (string "Accept"))
			  exists))
	 (click (string "Accept"))


	 ,(lprint :msg "login with password")

	 (click (string "Sign In"))
	 (write (string "plops"))
	 (press TAB)
	 (write args.password ;:into (string "Password")
		)
	 (click (string "Sign In"))))

       (python
	(export
	 ,(lprint :msg "clear text")
					;(setf cm (S (string "//div[contains(@class,'CodeMirror')]")))
	 (setf cm (S (string "//div[contains(@class,'CodeMirror-lines')]")))
	 (click cm)
	 ,(lprint :msg "select all")
	 (press ARROW_UP)
	 (press (+ CONTROL (string "a")))
	 ,(lprint :msg "delete")
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


	 ,(lprint :msg "load source from" :vars `(args.input))
	 (with (as (open args.input)
		   f)
	       (setf s (f.read))
	       )
	 ,(lprint :msg "update the text" :vars `(s))
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
	 (do0 ,(lprint :msg "save")
	      (click (string "Save"))
	      ,(lprint :msg "wait for save to finish")
	      (wait_until (dot (Button (string "Save"))
			       exists)))

	 (do0 ,(lprint :msg "compile code")
	      (click (S (string "#compileButton"))))
	 (time.sleep 3)
	 ))
       (python
	(export
	 (when args.kill
	   (do0
	    ,(lprint :msg "close browser")
	    (kill_browser))))))
		  
		  )
		 #+nil
		 (:code
		  
		  ))))

	  (loop for e in codes
		and e-i from 0
		do
		   (destructuring-bind (&key name code) e
		     (let* ((fn (format nil "/dev/shm/~3,'0d" e-i))
			    (fn-lisp (format nil "/dev/shm/~3,'0d.lisp" e-i))
			    (code-to-emit (if (listp code)
					      (cond
						((eq (first code) 'do0)
						 code)
						
						((eq (caar code) 'python)
						 ;; this used to be a notebook, containing:
						 #+nil ((python (export "<string>"))
							(python (export (do0 ...)))
							(python (do0 ...))
							(markdown ...) 
							...)
						 ;; ignore the markdown 
						 `(do0
						   ,@(loop for python-sexpr in code
							   collect
							   (cond ((eq (car python-sexpr) 'markdown)
								  `(do0 ,(second python-sexpr)))
								 ((eq 'export (car (second python-sexpr)))
								  (second (second python-sexpr)))
								 ((eq 'do0 (car (second python-sexpr)))
								  (second python-sexpr))
								 (t
								  (break "unsupported notebooks structure. i want (python (export ...: ~a" python-sexpr)))))
						 
						 )
						(t (break "strange"))
						)
					      (break "code should be a list starting with (do0 ... or (python ... or (markdown ...")
					      )))
		       (write-source fn code-to-emit)
		       
		       (with-output-to-file (s fn-lisp 
						       :if-exists :supersede
						       :if-does-not-exist :create
						       )
			 (format s "~s" code-to-emit)
			 )
		       #+nil 
		       (sb-ext:run-program "/usr/bin/black" (list fn))))))       
       
       )))

  )
