(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/33_tkinter")
  (defparameter *code-file* "run_00_start")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 "# %% imports"
		 
		 (imports (		;os
			   ;sys
			   ;time
					;docopt
			   pathlib
			   (np numpy)
			   ;serial
			   (pd pandas)
			   ;(xr xarray)
			   ;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			   ;scipy.ndimage
			   ;scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
			   ;datetime
			   ;dask.distributed
					;(da dask.array)
					;PIL
					;libtiff
			   ;visdom
			   ))
		 "from tkinter import *"


		 (setf
	       _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

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
		 
		 #+nil (do0
		  (setf root (Tk))

		  (def myclick ()
		    (setf mylab (Label root :text (string "look!"))
			  )
		   (mylab.pack))
		  
		  (setf but (Button root
				    :text (string "click")
				    :padx 23t 
				    :command myclick))
		  (but.pack)
		  ,@(loop for (e r c) in `(("hello" 0 0)
				      ("my name is" 1 1))
			 and i from 0
			 collect
			 `(do0
			   (setf ,(format nil "lab~a" i)
				 (dot (Label root :text (string ,e))
				      (grid :row ,r
					  :column ,c)))
			   #+nil (dot ,(format nil "lab~a" i)
				(grid :row ,r
					  :column ,c))))

		  
		  (root.mainloop)
		  )

		 (do0
		  (do0
		   (setf root (Tk ))
		   (root.title  (string "simple calculator")))
		  (do0
		   (setf entry (Entry root :width 35 :borderwidth 5)
			 )
		   (entry.grid :row 0
			       :column 0
			       :columnspan 3
			       :padx 10
			       :pady 10)
		   
		   
		   (def button_click (n)
		   
		     (setf cur (entry.get))
		     (entry.delete 0 END)
		     (entry.insert 0 (+ (str cur)
				    (str n)))
		     return)
		   ,@(loop for e in `((0 4 0)

					    (1 3 0)
					    (2 3 1)
					    (3 3 2)

					    (4 2 0)
					    (5 2 1)
					    (6 2 2)

					    (7 1 0)
					    (8 1 1)
				      (9 1 2)

				      (add 5 0 :padx 39)
				      (eq 5 1 :padx 91 :columnspan 2)
				      (clear 4 1 :padx 79 :columnspan 2)
					    )
			 
			   collect
			   (destructuring-bind (name r c &key (padx 40)
							   (pady 20)
							   (command `(lambda () (button_click ,name)))
							   (columnspan 1)) e
			     (let ((but (format nil "button_~a" name)))
			      `(do0
				(setf ,but
				      (dot (Button root :text (string ,name)
							:padx ,padx
							:pady ,pady
							:command ,command
						   
							)
					   )
				      )
				(dot ,but
				 (grid :row ,r
				       :column ,c
				       :columnspan ,columnspan
				       ))
				)))))
		 ; (root.mainloop)
		  )
		 ))
 	   ))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

