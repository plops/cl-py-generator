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
	      repeats 100_000
	      xbar 0.0
	      C 1.0
	      sC (np.sqrt C))
	(setf q (gen_noise :n n :repeats repeats :loc xbar :scale sC))
	(setf 
	      tn (np.exp (- (scipy.special.gammaln (/ n 2))
			    (scipy.special.gammaln (/ (- n 1) 2)))))

	(setf log False)
	,@(loop for e in `((mu mean
			       :center xbar
			       :pdf f_mu_X
			       :code-confidence
			       (do0 (setf da (* (np.sqrt (/ C (- n 1)))
						(scipy.stats.t.ppf (* .5 (+ 1 alpha))
								   (- n 1))))
				    
				    (setf db da)
				    (setf a (- xbar da))
				    (setf b (+ xbar db))
				    (do0
				     (setf var_lo (* -3 sC)
					   var_hi (* 3 sC))
				     (setf var (np.linspace (+ xbar var_lo) 
							    (+ xbar var_hi)
							    200))))
			       :code-pdf
			       (do0
				(setf mu var)
				(setf f_mu_X (* (/ tn (np.sqrt (* np.pi C)))
						(np.power (+ 1 (/ (** (- mu xbar) 2)
								       C))
							  (/ n -2)))))
			       :code-pdf-shade
			       (* (/ tn (np.sqrt (* np.pi C)))
				  (np.power (+ 1 (/ (** (- var_ab xbar) 2)
						    C))
					    (/ n -2)))
			       :code-value
			       (dot q (mean :dim (string "n"))
				   values)
			       )
			   (sigma standard-deviation
				  :center sC
				  :pdf f_std_X
			       :code-confidence
			       (do0 
				    (setf arg0 (np.sqrt (/ (* n C)
							   2)))
				    (setf c_invcdf_gengamma -2)
				    (comments "median as center point")
				    (setf a (* arg0 (scipy.stats.gengamma.ppf (/ (- 1 alpha)
										 2)
									      (/ (- n 1)
										 2)
									      c_invcdf_gengamma)))
				    (setf b (* arg0 (scipy.stats.gengamma.ppf (/ (+ 1 alpha)
										 2)
									      (/ (- n 1)
										 2)
									      c_invcdf_gengamma)))
				    (do0
				     (setf var (np.linspace (np.maximum 0 (- a (* 3 (- b a))))
							    (+ b (* 3 (- b a)))
							    
							    200))))
			       :code-pdf
			       (do0
				(setf std var)
				(setf f_std_X
				      #+nil (* (np.sqrt (/ (** (* n C)
							 (- n 1))
						     (** 2 (- n 1))))
					 (* (/ 2
					       (** std n))
					    (np.exp (- (/ (* n C)
							  (*   -2 (** std 2)))
						       (scipy.special.gammaln (/ (- n 1)
										 2))))))
				      (* 
					 (* 2 
					    (np.exp (+ (/ (* n C)
							  (*   -2 (** std 2)))
						       (* -1 (scipy.special.gammaln (/ (- n 1)
										       2)))
						       (* -1 n (np.log std))
						       (- (* 
							   (/ (- n 1) 2)
							   (np.log  (* n C)))
							  (* (/ (- n 1)
								2)
							     (np.log 2)))))))
				      )
				)
				  :code-pdf-shade
				    (* 
					 (* 2 
					    (np.exp (+ (/ (* n C)
							  (*   -2 (** var_ab 2)))
						       (* -1 (scipy.special.gammaln (/ (- n 1)
										       2)))
						       (* -1 n (np.log var_ab))
						       (- (* 
							   (/ (- n 1) 2)
							   (np.log  (* n C)))
							  (* (/ (- n 1)
								2)
							     (np.log 2)))))))
				  #+nil
				  (* (np.sqrt (/ (** (* n C)
								 (- n 1))
							     (** 2 (- n 1))))
						 (* (/ 2
						       (** var_ab n))
						    (np.exp (- (/ (* n C)
								  (*   -2 (** var_ab 2)))
							       (scipy.special.gammaln (/ (- n 1)
											 2))))))
				  :code-value
			       #+nil (dot q (std :dim (string "n"))
					  values)
			       (dot (* (/ tn
				      (- n 1))
				   (np.sqrt
				    (* 2
				       (dot
					(** (- q
					       (dot q (mean :dim (string "n"))))
					    2)
					(sum :dim (string "n"))))))
				    values)
			       )
			   )
		collect
		(destructuring-bind (var var-string
				     &key
				       center
				       pdf
				       code-confidence
				       code-pdf
				       code-pdf-shade
				       code-value
				       ) e
		 `(do0
		   (comments ,(format nil "marginal distributions for ~a" var-string))
		   (comments "confidence interval alpha x 100 percent")
		   (setf alpha .96)
		   ,code-confidence
		   
		   

		   ,code-pdf
	     
		   (do0
		    (figure)
		    
		    (comments "plot histogram")
		    (plt.hist ,code-value
			      :bins (np.linspace (- a (* 3 (- b a)))
						 (+ b (* 3 (- b a)))
						 200)
			      :density True
			      :histtype (string "step")
			      :color (string "yellowgreen")
			      :linewidth 3
			      :label (string "histogram")
			      :log log
			      )
		    (xlabel (string ,(format nil "$\\~a$" var))))
		   (do0
		    (if log
			(plt.semilogy var ,pdf :color (string "k") :label (string ,pdf))
			(plot var ,pdf :color (string "k") :label (string ,pdf)))
		    
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
		     (setf var_ab (np.linspace a b 120))
		     (setf y_fill ,code-pdf-shade)
		     (plt.fill_between
		      var_ab
		      0
		      y_fill
		      :color (string "red")
		      :alpha .5
		      ))
		    (title (dot (string "n={}, red .. {:.1f}% confidence (numeric {:6.4f}%)")
				(format n (* 100 alpha)
					(* 100 (np.trapz y_fill var_ab)))))
		    (when log
		      (ylim 1e-6 10))
		    (grid)
		    
		    (legend)))))
	
	))))))



