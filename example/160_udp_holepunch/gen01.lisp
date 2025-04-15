(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "160_udp_holepunch")
  (defparameter *idx* "01") 
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

  
  (let* ((notebook-name "holepunch")
	 )
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (string3
	"
UDP Hole Punching Tool

Establishes a bidirectional UDP communication path (pinhole) between two machines,
potentially behind NATs or firewalls, using a token exchange mechanism.

This script is intended to be run on a client machine. It uses SSH to execute
itself on a remote server, coordinating the UDP hole punching process.

Requires:
- Python 3.6+ (for f-strings, secrets)
- SSH access to the remote server with Python 3 available.
- The target UDP port should ideally be > 1024 if running as a non-root user.

Example Usage (Client):
  ./udp_hole_puncher.py user@remote.example.com 60001

How it works:
1. Client initiates, binds to the local UDP port, generates a token.
2. Client SSHes to the server and runs this script remotely.
3. Server instance binds to the same UDP port, generates its token.
4. Both continuously send their token to the other peer and listen for replies.
5. When a peer receives the other's token, it includes it in subsequent messages.
6. When a peer receives a message confirming both tokens, the hole is considered punched.
")
       (imports (sys os time socket secrets argparse
		     subprocess threading queue logging select))
       (imports-from (typing Optional Tuple List))
       (setf DEFAULT_TIEMOUT_SECONDS 1.0
	     MAX_ATTEMPTS 15
	     RETRY_DELAY_SECONDS .5
	     LOG_FORMAT (string "$(asctime)s - %(levelname)s - [%(threadName)s] %(message)s")
	     TOKEN_BYTE_LENGTH 16
	     MSG_SEPARATOR (string-b "|")
	     ACK_FLAG (string-b "ACK")
	     NULL_TOKEN_PLACEHOLDER (string-b "NULL")
	     BUFFER_SIZE 1024)

       ;; setup logging
       (logging.basicConfig :level logging.INFO
			    :format LOG_FORMAT)
       (setf logger (logging.getLogger __name__))

       ;; helper classes
       )))
  )
