(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "160_udp_holepunch")
  (defparameter *idx* "02") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format 
                          ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  
  (let* ((notebook-name "run_self_on_remote")
	 )
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (imports (subprocess sys os))

       (def run_self_on_remote (host)
	 (with (as (open __file__ (string "rb"))
		   f)
	       (setf script_content (f.read)))
	 (setf cmd (list (string "ssh")
			 host
			 (string "python3")
			 (string "-")
			 ;(str port)
			 ))
	 (setf result
	       (subprocess.run
		cmd
		:input script_content
		:capture_output True
		:check False ;; prevent raising error on non-zero exit
		:text False
		))
	 (when (== __name__ (string "main"))
	   (comments "remote running script name is '-'")
	   (if (== (aref sys.argv 0)
		   (string "-"))
	       (do0
		(print "remote script"))
	       (do0
		(print "local script")
		(run_self_on_remote (string "tux")
				    ;2224
				    )))
	   )
	 ))))
  )
