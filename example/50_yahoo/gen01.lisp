(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/50_yahoo")
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
			 re
			 json
			 csv
			 ;io.StringIO
			 bs4
			 requests
			   
			 (np jax.numpy)
			   
			 ))
		 

		 
		 
	       ,(format nil "from matplotlib.pyplot import ~{~a~^, ~}"
			`(plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
	       
		 
		 
	       )
	      ))
     (python
      (do0 (setf stock (string "F"))))
     ,@ (loop for (name url) in `((stats "https://finance.yahoo.com/quote/{}/key-statistics?p={}")
				  ;(profile "https://finance.yahoo.com/quote/{}/profile?p={}")
				  ;(financials "https://finance.yahoo.com/quote/{}/financials?p={}")
				  )
	      collect
	      `(python
		(do0 ;def ,name (stock)
		  (setf response (requests.get (dot (string ,url)
						    (format stock stock))))
		  (setf soup (bs4.BeautifulSoup response.text
						(string "html.parser")))
		  (setf pattern (re.compile (rstring3 "\\s--\\sData\\s--\\s"))
			script_data (dot (soup.find (string "script")
						    :text pattern)
					 (aref contents 0)))
		  (setf start (- (script_data.find (string "context")) 2))
		  (setf json_data (json.loads (aref script_data (slice start -12))))
		  ,@(loop for e in `((is incomeStatementHistory)
				     (cf cashflowStatementHistory cashflowStatements)
				     (bs balanceSheetHistory balanceSheetStatements))
			  collect
			  (destructuring-bind (short-name s1 &optional (s2 s1)) e
			   (labels ((name (prefix)
				      (format nil "~a_~a" prefix short-name)))
			     `(do0 (setf ,(name "annual")
				     (aref
				      (aref
				       (aref
					(aref
					 (aref
					  (aref json_data (string "context"))
					  (string "dispatcher"))
					 (string "stores"))
					(string "QuoteSummaryStore"))
				       (string ,s1))
				      (string ,s2)))
				   (setf ,(name "quarterly")
				     (aref
				      (aref
				       (aref
					(aref
					 (aref
					  (aref json_data (string "context"))
					  (string "dispatcher"))
					 (string "stores"))
					(string "QuoteSummaryStore"))
				       (string ,(format nil "~aQuarterly" s1)))
				      (string ,s2))))))))))
     (python (do0
	      (with (as (open (string "/dev/shm/data.json")
			      (string "w"))
			outfile)
		    (json.dump json_data outfile))))
     ))
  )



