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
			 (mpf mplfinance)
			   
			 ))
		 

		 
		 
	       ,(format nil "from matplotlib.pyplot import ~{~a~^, ~}"
			`(plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
	       
		 
		 
	       )
	      ))
     (python
      (do0 (setf stock (string "F"))))
     ,@ (loop for (name url json-headers)
		in
	      `((stats "https://finance.yahoo.com/quote/{}/key-statistics?p={}"
		       ((is
			 (dot  (pd.DataFrame (aref x (string "financialData")))
			       (transpose)))
										 
					; (cf cashflowStatementHistory cashflowStatements)
					;(bs balanceSheetHistory balanceSheetStatements)
										  ))
		(profile "https://finance.yahoo.com/quote/{}/profile?p={}"
			 ((officer (pd.DataFrame (aref (aref x (string "assetProfile")) (string "companyOfficers"))))
			  (business_summary (aref (aref x (string "assetProfile")) (string "longBusinessSummary")))
			  (sec (pd.DataFrame (aref (aref x (string "secFilings")) (string "filings"))))
			  #+nil (events (pd.DataFrame (aref (aref x (string "calendarEvents"))
						      (string "earnings"))))
			  (summary_detail (dot (pd.DataFrame (aref x (string "summaryDetail")))
					       (transpose)))))

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
		  (setf x (aref
			   (aref
			    (aref
			     (aref json_data (string "context"))
			     (string "dispatcher"))
			    (string "stores"))
			   (string "QuoteSummaryStore")))
		  ,@(loop for e in json-headers
			  collect
			  (destructuring-bind (short-name code) e
			   (labels ((name (prefix)
				      (format nil "~a_~a" prefix short-name)))
			     `(do0
			       (try (do0
				 (setf ,(name "dat")
				       ,code
				       )
				 (display ,(name "dat")))
				    ("Exception as e"
				     (display e)
				     pass)))))))))


     (python
      (do0
       (setf stock_url (dot (string "https://query1.finance.yahoo.com/v7/finance/download/{}?")
			    (format stock)))
					; period1=1587535484&period2=1619071484&interval=1d&events=history&includeAdjustedClose=true
       (setf params (dictionary ;:period1 1587535484
				;:period2 1619071484
				:range (string "1y")
				:interval (string "1d")
				:events (string "history")
				:includeAdjustedClose True))
       (setf response (requests.get stock_url :params params))
       (setf df_stock (pd.read_csv (io.StringIO response.text)
				   :index_col 0
				   :parse_dates True)
	     )
       (display df_stock)
       #+nil (do0
	(figure)
	(dot df_stock (set_index (string "Date"))
	     High (plot))
	(dot df_stock (set_index (string "Date"))
	     Low (plot))
	(grid))
       (do0
	(mpf.plot df_stock :type (string "candle") :volume True)
	;(grid)
	)))
     
     (python (do0
	      (with (as (open (string "/dev/shm/data.json")
			      (string "w"))
			outfile)
		    (json.dump json_data outfile))))
     ))
  )



