(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/36_cadquery")
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

			   (cq cadquery)))
		 "from Helpers import show"
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
				     (- tz))
			     )))
		 (setf length 80.0
		       height 60.0
		       thickness 10.0
		       center_hole_dia 22.0
		       cbore_hole_dia 2.4
		       cbore_dia 4.4
		       cbore_depth 2.1
		       r (dot (cq.Workplane (string "XY"))
			      (box length height thickness)
			      ;; use selector: maximize in z direction
			      (faces (string ">Z"))
			      (workplane)
			      (hole center_hole_dia)
			      (faces (string ">Z"))
			      (rect (- length 8)
				    (- height 8)
				    :forConstruction True)
			      ;; create 4 holes at the corners
			      (vertices)
			      (cboreHole
			       cbore_hole_dia
			       cbore_dia
			       cbore_depth)

			      ))

		 (show r (tuple 204 204 204 0.0))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

