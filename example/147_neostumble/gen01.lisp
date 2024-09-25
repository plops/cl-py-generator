(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:example)))
(setf *features* (set-difference *features* '(;:example
					      )))

(progn
  (defparameter *project* "147_neostumble")
  (defparameter *idx* "01") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format (- (time.time) start_time)
                          ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  (let* ((notebook-name "host")
	 )
    (write-source
     (format nil "~a/source01/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"

       (imports ((pd pandas)
		 datetime
		 time))

      
       (setf b (pd.read_csv (string "../data/beacons.csv")))
       (setf c (pd.read_csv (string "../data/cells.csv")))
       (setf w (pd.read_csv (string "../data/wifis.csv")))

       (setf u (w.ssid.unique))

       (setf w1 (w.set_index (string "ssid")))
       (setf w2 (w1.sort_values :by (string "signalStrength")
				:ascending False))
       (setf w3 (dot w2
		     (reset_index)
		     (set_index (list (string "ssid")
				      (string "signalStrength")))))
       (print (dot (w.sort_values :by (string "signalStrength")
				  :ascending False)
		   (aref iloc (slice 0 300))))
       ))))
