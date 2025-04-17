(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "160_udp_holepunch")
  (defparameter *idx* "05") 
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
     (format nil "~a/source01/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (imports (subprocess sys os socket time
			    argparse))
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
	   (for (i (range 3))
		,(lprint :msg "sendto" :vars `(src dst port))
		(sock.sendto msg (tuple dst port))
		(time.sleep .1)))
	  ("Exception as e"
	   (print (fstring "exception {e}")))
	  (finally (sock.close))))

       (def run_cmd_and_wait (host cmds)
	 (with (as (open __file__ (string "rb"))
		   f)
	       (setf script_content (f.read)))
	 (setf cmd (+ (list (string "ssh")
				 host
				 )
			   cmds))
	 ;(print (fstring "run {cmd}"))
	 (setf result
	       (subprocess.run
		cmd
		;:input script_content
		:capture_output True
		:check False ;; prevent raising error on non-zero exit
		:text False
		))
	 ;(print (fstring "remote script {result}"))
	 (setf ip (dot result stdout (decode (string "utf-8"))))
	 ,(lprint :vars `(cmd ip))
	 (return ip)
	 )

       (def run_local_cmd_and_wait ( cmds)
	 (with (as (open __file__ (string "rb"))
		   f)
	       (setf script_content (f.read)))
	 (setf cmd cmds)
	 ;(print (fstring "run {cmd}"))
	 (setf result
	       (subprocess.run
		cmd
		;:input script_content
		:capture_output True
		:check False ;; prevent raising error on non-zero exit
		:text False
		))
	 ;(print (fstring "local script {result}"))
	 (setf ip (dot result stdout (decode (string "utf-8"))))
	 ,(lprint :vars `(cmd ip))
	 (return ip)
	 )
       
       (def run_self_on_remote (host cmds)
	 (with (as (open __file__ (string "rb"))
		   f)
	       (setf script_content (f.read)))
	 (setf cmd (+ (list (string "ssh")
			  host
			  (string "python3")
			  (string "-")
					;(str port)
			  )
		      cmds))
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
	   (do0
	    (setf parser (argparse.ArgumentParser))
	    #+nil (parser.add_argument (string "-r")
				 (string "--remote")
				 :action (string "store_true")
				 :help (string "script executes on remote host"))
	    (parser.add_argument (string "-j")
				 (string "--jump-host")
				 :type str
				 :default (string "tinyeu")
				 :help (string "ssh hostname of jump host"))
	    (parser.add_argument (string "-s")
				 (string "--server-host")
				 :type str
				 :default (string "tux")
				 :help (string "ssh hostname of server host"))
	    (parser.add_argument (string "-e")
				 (string "--extra-host")
				 :type str
				 :default (string "tinyus")
				 :help (string "ssh hostname of second server host"))
	    (parser.add_argument (string "-E")
				 (string "--extra-ip")
				 :type str
				 :default (string "1.1.3.4")
				 :help (string "ip of second server host"))
	    (parser.add_argument (string "-S")
				 (string "--server-ip")
				 :type str
				 :default (string "1.2.3.4")
				 :help (string "ip of server host"))
	    (parser.add_argument (string "-J")
				 (string "--jump-ip")
				 :type str
				 :default (string "1.3.3.4")
				 :help (string "ip of jump host"))
	    (parser.add_argument (string "-L")
				 (string "--local-ip")
				 :type str
				 :default (string "1.4.3.4")
				 :help (string "ip of local computer"))
	    (setf args (parser.parse_args)))
	   (if (== (aref sys.argv 0)
		   (string "-"))
	       (do0
		(comments "remote running script name is '-'")
		(print (string "remote script"))
		(emit any_ip extra_ip)
		(emit any_ip local_ip)
		(emit any_ip jump_ip)
		)
	       (do0
		(print (string "local script"))
		(setf local_ip
		      (run_local_cmd_and_wait 
					      (list (string "curl")
						    (string "ifconfig.me"))
					      )
		      )
		(setf server_ip
		      (run_cmd_and_wait args.server_host
					(list (string "curl")
					      (string "ifconfig.me"))
					)

		      )
		(setf jump_ip
		      (run_cmd_and_wait args.jump_host
					(list (string "curl")
					      (string "ifconfig.me"))
					)
		      )
		(setf extra_ip
		      (run_cmd_and_wait args.extra_host
					(list (string "curl")
					      (string "ifconfig.me"))
					)
		      )
		(run_self_on_remote (string "tux")
				    (list (string "-L")
					  local_ip
					  (string "-J")
					  jump_ip
					  (string "-S")
					  server_ip
					  (string "-E")
					  extra_ip))
		(emit any_ip extra_ip)
		(emit any_ip jump_ip)
		(emit any_ip server_ip)
		))
	   ))))
  )
