(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)
;; python3 -m pip install  --user justpy



(progn
  
  (defparameter *path*
    (format  nil "~a/stage/cl-py-generator/example/89_justpy"
	     (user-homedir-pathname)
	     ))
  (defparameter *code-file* "run_00_justpy")
  (defparameter *source* (format nil "~a/source/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when debug
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
		   (format (- (time.time) start_time)
			   ,@rest)))))
  (let* (
	 (code
	   `(do0
	     (imports ((jp justpy)))	     
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (external-program:run "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
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
		   debug True)
	     (def hello_world ()
	       (setf wp (jp.WebPage)
		     d (jp.Div :text (string "hello world"))
		     )
	       (wp.add d)
	       (return wp))
	     (jp.justpy hello_world))))
    (write-source (format nil "~a/~a" *source* *code-file*) code)))
