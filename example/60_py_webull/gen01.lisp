(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/60_py_webull")
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
					;bs4
					;requests
			   
					;(np jax.numpy)
					;(mpf mplfinance)
			   
			 ))
		 
	       (imports-from (selenium webdriver)
			     (selenium.webdriver.common.keys Keys)
			     (selenium.webdriver.support.ui WebDriverWait)
			     (selenium.webdriver.common.by By)
			     (selenium.webdriver.support expected_conditions)
			     
			     
			     )
		 
	       
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

     (python (do0
	      (comments "open browser that can be automated with python")
	      (comments "zoom out, so that everything is rendered")
	      (setf fp (webdriver.FirefoxProfile))
	      (fp.set_preference (string "layout.css.devPixelsPerPx") (string ".5"))
	      (setf driver (webdriver.Firefox fp))
	      (driver.set_window_size (* 2 1900) (* 2 1040))
	      (driver.get (string "https://app.webull.com/market/region/6"))))
     (python
      (do0
       (comments "wait up to 3 seconds for elements to appear")
       (driver.implicitly_wait 3)))

     #+nil (python
      (do0 ;; doesnt work in firefox
       (comments "tables that are not visible in the window will return empty text. ensure that all information is rendered by zooming out")
       (driver.execute_script (string "document.body.style.zoom='50 %'"))))

     (python
      (do0
       (setf a (driver.find_element_by_xpath (string "/html/body/div/div/div[3]/div/div[1]/div/div[2]/div/div/div[3]/div[1]/span/div/div")))
       (a.click)))

     (python
      (do0
       (comments "get header title with sector name")
					;                                                                               row     col
					;"/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[1]/div[1]/div[1]/span[1]"
					;"/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[1]/div[2]/div[1]/span[1]"
					;"/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[1]/div[3]/div[1]/span[1]"
					;"/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[2]/div[1]/div[1]/span[1]"
					;"/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[3]/div[1]/div[1]/span[1]"
					;"/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[5]/div[3]/div[1]/span[1]"
       (setf header (driver.find_element_by_xpath
		     (string "/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[3]/div[1]/div[1]/span[1]"))
	     perc (driver.find_element_by_xpath
		     (string "/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[3]/div[1]/div[1]/span[2]")))
       (comments "get growth percentage for the entire sector")
       (display header.text)
       (display perc.text)))
     (python
      (do0
       (comments "get all sector names from the grid of tables")
       (setf res (list ))
       (for (col (range 1 (+ 1 3)))
	    (for (row (range 1 (+ 1 5)))
		 (setf header (driver.find_element_by_xpath
			       (dot (string "/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[{}]/div[{}]/div[1]/span[1]")
				    (format row col))))
		 (setf perc (driver.find_element_by_xpath
			       (dot (string "/html/body/div/div/div[3]/div/div[1]/div/div[2]/div[1]/div[2]/div/div/div[1]/div[{}]/div[{}]/div[1]/span[2]")
				    (format row col))))
		#+nil  (setf myelem (dot (WebDriverWait driver delay)
				   (until
				    expected_conditions.presence_of_element_located
				    (tuple By.ID
					   (string "")))))
		 (res.append (dictionary :row row
					 :col col
					 :sector header.text
					 :sector_growth perc.text))))
       (setf df (pd.DataFrame res))
       (dot df (sort_values :by (string "sector_growth")
			    :ascending False))))

     ))
  )



