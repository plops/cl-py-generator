(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-r-generator"))

(in-package :cl-r-generator)

(progn
  ;; the following code needs inverted readtable, otherwise symbols
  ;; and filenames may have the wrong case and everything breaks in
  ;; horrible ways
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/87_semiconductor")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*))
  (let ((show-counter 1))
    (defun show (name code &key width height)
      (prog1
	  `(do0
	    (png (string ,(format nil "~2,'0d_~a.png" show-counter name))
		 ,@(when width
		     `(:width ,width))
		 ,@(when height
		     `(:height ,height)))
	    ,code
	    (dev.off))
	(incf show-counter))))
  (let ((nb-counter 2))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((r (do0
				       (comments
					,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ()
	(gen `(run_fit)
	     `((r
		(cell
		 (comments "load dependencies and file")
		 (require gamlss)
		 (setf location
		       (read.csv
			(string "/home/martin/stage/cl-py-generator/example/87_semiconductor/source/dir87_gen01_location.csv")))))
	       (r
		(cell
		 (do0
		  (comments "look at statistics of localization with 10e3 photons")
		  (setf dx ($ (aref location
				    (== location$max_phot 10000)
				    "")
			      dx))
		  (setf fit (fitDist dx :k 2
				     :type (string "realAll")))
		  (comments "i tried 10 and 10e3 photons, the best distribution fit seems to be normal")


		  )
		 ))

	       (r
		(cell (comments "explicitly fit normal distribution, and generate some diagnostic plots")
		      (setf mNO (histDist dx (string "NO")
					  :xlab (string "dx (nm)")
					;:bins 30
					;:n.cyc 100
					  ))
		      ))

	       (r
		(cell
		 (comments "show worm plot"
			   "https://rdrr.io/cran/gamlss/man/wp.html"
			   "detrended QQ-plot"
			   "departure from normality is highlighted"
			   "residual is fitted with cubic polynomial (can be used to identify areas of model violation)")
		 (wp mNO)))

	       (r
		(cell


		 (do0

		  (comments "compare empirical cumulative distribution function with the cdf for the gaussian fit")
		  (plot (ecdf dx))
		  (setf xs (seq -4 4 .01))
		  (comments "pdf .. prefix=d, cdf .. prefix=p, inverse cdf .. prefix=q")
		  (lines
		   xs
		   ((lambda (y)
		      (pNO y
			   :mu mNO$mu
			   :sigma  mNO$sigma)) xs)

		   :col (string "red")
		   :lwd 3)
		  (do0
		   (comments "lower and upper specification limit")
		   (setf value_nominal 0
			 value_lsl -1
			 value_usl 1)

		   ,@(loop for e in `((:name nominal)
				      (:name lsl :extra t :fun (lambda (x) (* 100 x)))
				      (:name usl :extra t :fun (lambda (x) (* 100 (- 1 x)))))
			   collect
			   (destructuring-bind (&key name extra fun) e
			     (let ((val (format nil "value_~a" name))
				   (area (format nil "area_~a_perc" name)))
			       `(do0
				 (abline :v ,val
					 :col (string "red"))
				 ,(if extra
				      `(setf ,area (,fun (pNO ,val
							      :mu mNO$mu
							      :sigma  mNO$sigma)))
				      `(comments "no area to compute for nominal"))
				 (text ,val .1 ,(if extra
						    `(sprintf
						      (string ,(format nil "~a\\noutside: %.1f%%" name))
						      ,area)
						    `(string ,(format nil "~a\\nvalue" name))))))))))
		 ))

	       (r
		(cell


		 (do0

		  (comments "determine measurement uncertainty with fitted inverses cumulative distribution function")

		  (setf xs (seq 0 1 .01))

		  (plot
		   xs
		   ((lambda (y)
		      (qNO y
			   :mu mNO$mu
			   :sigma  mNO$sigma)) xs)

		   :col (string "red")
		   :lwd 3
		   :xlab (string "probability")
		   :ylab (string "dx (nm)"))
		  (setf u_hi (qNO (- 1 .025)
				  :mu mNO$mu
				  :sigma  mNO$sigma))
		  (setf u_lo (qNO .025
				  :mu mNO$mu
				  :sigma  mNO$sigma))
		  (abline :h u_hi)
		  (abline :h u_lo)
		  (title (sprintf (string "95%% uncertainty U=%.3f nm, production tolerance=%.3f nm")
				  (- u_hi u_lo)
				  (- (max 0 (- value_usl u_hi))
				     (max 0 (+ value_lsl u_lo)))))
		  )
		 ))


	       )))))
  )




