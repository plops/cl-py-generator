(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "160_udp_holepunch")
  (defparameter *idx* "04") 
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

  
  (let* ((notebook-name "simultaneous_packets")
	 )
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "04" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (imports (subprocess sys os socket time))
       (imports-from (subprocess Popen PIPE DEVNULL))
       ;; curl ifconfig.me
       (setf port 60001
	     msg (string-b A)
	     client_ip (string "14.40.11.250")
	     server_ip (string "13.4.0.26")
	     any_ip (string "0.0.0.0"))
       
       ;; mosh-server new -p 60001
       ;; MOSH_KEY=... mosh-client 193.8.40.126 60001
       (def emit (src dst)
	 (try
	  (do0
	   (setf sock (socket.socket
		       socket.AF_INET
		       socket.SOCK_DGRAM))
	   (sock.bind (tuple src port))
	   (for (i (range 33))
		(sock.sendto msg (tuple dst port))
		(time.sleep .1)))
	  ("Exception as e"
	   (print (fstring "exception {e}")))
	  (finally (sock.close))))
       
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
	 (print (fstring "run {cmd}"))
	 (setf process (Popen cmd
			      :stdin PIPE
			      :stdout DEVNULL
			      :stderr DEVNULL))
	 (process.stdin.write script_content)
	 (process.stdin.close)
	 )
       (when (== __name__ (string "__main__"))
	   (print (fstring "start {sys.argv}"))
	   (if (== (aref sys.argv 0)
		   (string "-"))
	       (do0
		(comments "remote running script name is '-'")
		(print (string "remote script"))
		(emit any_ip client_ip))
	       (do0
		(print (string "local script"))
		(run_self_on_remote (string "tux"))
		(emit any_ip server_ip)))
	   ))))
  )
