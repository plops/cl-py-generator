(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)

;; try to use Arch module


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/36_cadquery")
  (defparameter *code-file* "run_01_arch")
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
					;scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime

					;(cq cadquery)
			   Arch
			   Part
			   ))
					;"from Helpers import show"
		 (setf
		  _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/")
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
				     (- tz))
			     )))
		 )

	    (do0
	     (setf doc (dot App (newDocument (string "bed")))))
	    (do0
	     (comments "https://wiki.freecadweb.org/Python_scripting_tutorial/de")
	     (setf sph_ (Part.makeSphere 10))

	     (do0 
	      #+nil (do0 (setf sph (doc.addObject (string "Part::Feature")
					(string "sph"))
			 sph.Shape sph_)
			 (doc.recompute))
	      (setf sph (Part.show sph_)))
	     sph.Volume
	     sph.Area)
	    (do0
	     (comments "https://wiki.freecadweb.org/Topological_data_scripting/de")))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

