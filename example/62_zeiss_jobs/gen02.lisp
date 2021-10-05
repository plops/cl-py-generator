(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/62_zeiss_jobs")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                           ,@rest)))))

  (write-notebook
   :nb-file (format nil "~a/source/02_helium_scrape.ipynb" *path*)
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
			 selenium.webdriver ;.FirefoxOptions
			   
			 ))
	       
		 
	      #+nil  (imports-from (selenium webdriver)
			     (selenium.webdriver.common.keys Keys)
			     (selenium.webdriver.support.ui WebDriverWait)
			     (selenium.webdriver.common.by By)
			     (selenium.webdriver.support expected_conditions)
			     
			     
			     )
		 
	       
	       (imports-from (matplotlib.pyplot
			      plot imshow tight_layout xlabel ylabel
			      title subplot subplot2grid grid
			      legend figure gcf xlim ylim)
			     (helium *))
		 
	       )
	      ))
     (python
      (do0
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
      (do0
       (do0
	(setf options (selenium.webdriver.FirefoxOptions))
	(options.set_preference
	 (string "layout.css.devPixelsPerPx") (string ".5")
	 )
	#+nil (do0 (options.set_preference
	      (string "browser.window.width") (string "0"))
	     (options.set_preference
	      (string "browser.window.height") (string "0")))
	#+nil (do0 (options.add_argument (string "--width=2560"))
	     (options.add_argument (string "--height=1440"))))
       (setf driver (start_firefox			;:head_less True
		       :options options
		       ))
       (driver.set_window_size (* 2 1900) (* 2 1040))))
     (python (do0
	      (go_to (string "https://www.zeiss.de/jobs"))
	      #+nil (when
		  (dot (Text (string "Datenschutzeinstellungen"))
		       (exists))
		(click (string "Alle akzeptieren")))
	      (click (string "Alle akzeptieren"))
	      (press PAGE_DOWN)
	      ;; Button().exists()
	      ;;(click (string "Nein, danke"))
	      ;; RadioButton
	      (click (string "alle ausklappen"))
	      (click (string "Semiconductor Manufacturing Technology"))
	      (click (string "Oberkochen"))
	      (click (string "Suchen"))
	      (time.sleep 3)
	      (press PAGE_DOWN)))
     (python
      (do0
       (setf next_href None
	     old_next_href None)
       (do0
	(setf res (list))
	(while True
	 (do0
	  (do0
	   (press PAGE_DOWN)
	   (time.sleep 1)
	   (press PAGE_DOWN)
	   
	   (for (count (range 3))
       	    (try
	     (do0 (setf l (find_all (S (string "ul.list-container > li"))))
		  (dot (aref l 0)
		       web_element
		       text)
		  break)
	     ("Exception as e"
	      ,(lprint "exception" `(count e))
	      pass)))
	   (for (e l)
		(setf (ntuple job location) (dot e web_element text (split (string "\\n"))))
		(res.append (dictionary :job job
					:location location)))
	   )
	  (do0
       	   (setf next_button (aref
			      (find_all (S (string "li.pagination-next > a.nolinkrewrite")))
			      0)
		 old_next_href next_href
		 next_href (next_button.web_element.get_property (string "href")))
	   ,(lprint "href" `(old_next_href next_href))
	   (if (== next_href old_next_href)
	       (do0
		;; the last link will point to the current page
		;; then we stop
		break)
	       (do0 
		(click next_button)
		(time.sleep 3)))))))

       (do0 (setf df (pd.DataFrame res))
	    (display df)
	    (df.to_csv (string "contents.csv")))
       ))

     (python
      (do0
       (kill_browser)))
     #+nil 
     (python (do0
	      (comments "open browser that can be automated with python")
	      (comments "zoom out, so that everything is rendered")
	      (setf fp (webdriver.FirefoxProfile))
	      (fp.set_preference (string "layout.css.devPixelsPerPx") (string ".5"))
	      (setf driver (webdriver.Firefox fp))
	      (driver.set_window_size (* 2 1900) (* 2 1040))
	      (driver.get (string "https://www.zeiss.de/corporate/karriere/stellenangebote-und-bewerbung.html?jr=1400&jl=&jsl=&jd=300000005&jb=&jp=200000001"))
	      (do0
	       (comments "wait up to 3 seconds for elements to appear")
	       (driver.implicitly_wait 3))))
     #+nil 
     (python
      (do0
       (def highlight (element)
	 (setf driver element._parent)
	 (def apply_style (s)
	   (driver.execute_script (string "arguments[0].setAttribute('style',arguments[1]);")
				  element
				  s))
	 (setf original_style (element.get_attribute (string "style")))
	 (apply_style (string "background: yellow; border: 2px solid red;"))
	 (time.sleep .3)
	 (apply_style original_style))))

     #+nil
     ,@(loop for e in
	     `("button[id=onetrust-accept-btn-handler]"
	       "span[class=show-all]")
	     collect
	     `(python
	       (do0
		(setf el
		      (driver.find_element_by_css_selector (string ,e)))
		(highlight el)
		(dot el (click)))))
     
     
     #+nil
     (python
      (do0
       (driver.quit)))
     #+nil (python
      (do0 ;; doesnt work in firefox
       (comments "tables that are not visible in the window will return empty text. ensure that all information is rendered by zooming out")
       (driver.execute_script (string "document.body.style.zoom='50 %'"))))


     ))
  )



