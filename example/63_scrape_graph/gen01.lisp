(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/63_scrape_graph")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  (write-notebook
   :nb-file (format nil "~a/source/01_scrape.ipynb" *path*)
   :nb-code
   `(
     (python (do0
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
					bs4
					requests
			 (nx networkx)
					;(np jax.numpy)
					;(mpf mplfinance)
			   
			 ))
	      
		#+nil 
	       (imports-from (selenium webdriver)
			     (selenium.webdriver.common.keys Keys)
			     (selenium.webdriver.support.ui WebDriverWait)
			     (selenium.webdriver.common.by By)
			     (selenium.webdriver.support expected_conditions)
			     
			     
			     )
		 
	       #+nil
	       (imports-from (matplotlib.pyplot
			      plot imshow tight_layout xlabel ylabel
			      title subplot subplot2grid grid
			      legend figure gcf xlim ylim))
		 
	       )
	      ))
     (python
      (do0
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

     (markdown "    1. Create an empty graph
    2. Visit Homepage
    3. Find all HTML a tags and select those that are within the same website
    4. Create edge between current link and next
    5. Visit next link
    6. Repeat…")
     (python
      (do0
       (setf domain (string "connectingfigures.com")
	     url (dot (string "https://{}/")
		      (format domain)))
       (setf processed (list)
	     queue (list url)
	     G (nx.DiGraph))
       (while queue
	      (setf l (queue.pop 0)
		    req (requests.get l)
		    soup (bs4.BeautifulSoup req.content (string "html.parser"))
		    links (soup.find_all (string "a"))
		    links (list (for-generator (ln "links if ln.get('href')")
					       (ln.get (string "href"))))
		    links (list (for-generator (ln links)
					       (aref 
						(ln.split (string "#")) 0)))
		    links (list (for-generator (ln "links if domain in ln")
					       ln))
		    links (list (for-generator (ln "links if ln != l")
					       ln))
		    links (set links))
	      (setf to_add (list (for-generator (ln "links if ln not in queue")
						ln))
		    to_add (list (for-generator (ln "to_add if ln not in processed")
						ln)))
	      (queue.extend to_add)
	      (for (link links)
		   (print (tuple l link))
		   (G.add_edge l link))
	      (processed.append l)
	)
       ))
     

     ))
  )



