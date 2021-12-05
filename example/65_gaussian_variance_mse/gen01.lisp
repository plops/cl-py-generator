(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "alexandria")
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/65_gaussian_variance_mse")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (let ((nb-file "source/01_random_sim.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp random_sim01"))
      (python (do0
	       
	       "#export"
	       (do0
					;"%matplotlib notebook"
		(do0
		      
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
					(np numpy)
					;serial
					(pd pandas)
					(xr xarray)
					(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
			  scipy.optimize
			  scipy.stats
			  scipy.special
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
					;(nx networkx)
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
		(imports-from (flask Flask render_template
				     request redirect url_for))
		
		(imports-from (matplotlib.pyplot
			       plot imshow tight_layout xlabel ylabel
			       title subplot subplot2grid grid
			       legend figure gcf xlim ylim))
		 
		)
	       ))
      (python
       (do0
	"#export"
	(setf
	 _code_git_version
	 (string ,(let ((str (with-output-to-string (s)
			       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		    (subseq str 0 (1- (length str)))))
	 _code_repository (string ,(format nil "~a/~a/~a" *repo-dir-on-github* *example-subdir* nb-file))
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
	"#export"
	(def gen_noise (&key (n 100) (repeats 300) (loc 0) (scale 1))
	  (setf a
		(np.random.normal :loc loc :scale scale :size (list repeats n)))
	  (return (xr.DataArray :data a
				:dims (list (string "repeats")
					    (string "n"))
				:coords (dictionary
					 :repeats (np.arange repeats)
					 :n (np.arange n))
				:attrs (dictionary
					:description (string "normal noise")
					:loc loc
					:scale scale)
				)))
	(setf q (gen_noise))
	q))
      
      (python
       (do0
	(comments "look at numerical stability of gamma function calculation"
		  "without logarithm the gamma function stops working at around n=350")
	
	(setf n (np.arange 1 320))
	(setf tn (/ (scipy.special.gamma (/ n 2))
		    (scipy.special.gamma (/ (- n 1) 2)))
	      tn_ (np.exp (- (scipy.special.gammaln (/ n 2))
			     (scipy.special.gammaln (/ (- n 1) 2))))
	      )
	(do0
	 (plot n tn :label (string "gamma"))
	 (plot n tn_ :label (string "exp gammaln"))
	 (xlabel (string "n"))
	 (legend)
	 (grid))
	(do0
	 (figure)
	 (plot n (np.abs (- tn tn_)) :label (string "residual gamma-exp gammaln"))
	 (xlabel (string "n"))
	 (legend)
	 (grid))))

      (python
       (do0
	"#export"
	(setf n 10
	      repeats 300
	      xbar 0.0
	      C 1.0
	      sC (np.sqrt C))
	(setf q (gen_noise :n 10 :repeats repeats :loc xbar :scale sC))
	(setf 
	      tn (np.exp (- (scipy.special.gammaln (/ n 2))
			    (scipy.special.gammaln (/ (- n 1) 2)))))

	
	(do0
	 (comments "marginal distributions for mean")
	 (comments "confidence interval alpha x 100 percent")
	 (setf alpha .96)
	 (setf da (* (np.sqrt (/ C (- n 1)))
			    (scipy.stats.t.ppf (* .5 (+ 1 alpha))
					       (- n 1))))
	 (setf a (- xbar da))
	 (setf db da)
	 (setf b (+ xbar db))
	 (do0
	  (setf mu_lo (* -6 sC)
		mu_hi (* 6 sC)))
	 (setf mu (np.linspace (+ xbar mu_lo) 
			       (+ xbar mu_hi)
			       200))
	 
	 (setf f_mu_X (* (/ tn (np.sqrt (* np.pi C)))
			 (np.power (+ 1 (/ (** (- mu xbar) 2)
					   C))
				   (/ n -2))))
	 (do0
	  (plot mu f_mu_X :label (string "f_mu_X"))
	  (title (dot (string "n={}")
		      (format n)))
	  (plt.axvline a :color (string "r") :linestyle (string "dashed")
			 :label (dot (string "a={num:6.{precision}f}")
				     (format :num a :precision (int (* -1 (np.floor
									   (np.log10 da))))
					     )))
	  (plt.axvline b :color (string "orange") :linestyle (string "dashed")
		       :label (dot (string "b={num:6.{precision}f}")
				     (format :num b :precision (int (* -1 (np.floor
									   (np.log10 db))))
					     )))
	  (do0
	   (comments "fill the confidence interval")
	   (setf mu_ab (np.linspace a b 120))
	   (plt.fill_between
	    mu_ab
	   0
	    (* (/ tn (np.sqrt (* np.pi C)))
			 (np.power (+ 1 (/ (** (- mu_ab xbar) 2)
					   C))
				   (/ n -2)))
	    :color (string "red")
	    :alpha .5
	    ))
	  (grid)
	  (legend)))
	
	))))))



