(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

;; on arch linux inside termux on android phone:
;; sudo pacman -S jq jupyterlab
;; pip3 install --user nbdev ;; not working
(progn
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/94_tor")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key (msg "") vars)
    `(print (dot (string ,(format nil "{:7.6f} \\033[31m ~a \\033[0m ~{~a={}~^ ~}" msg vars))
                 (format
		  (- (time.time) start_time)
                  ,@vars))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let* ((dir (format nil "~a/~a/source/"
				*repo-dir-on-host*
				*example-subdir*
				))
		   (fn  (format nil "~a/~3,'0d_~{~a~^_~}.ipynb"
				dir
				nb-counter path)))
	       (ensure-directories-exist dir)
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ((cli-args `((:short "-H" :long "--host" :help "url to reach" :required True)
			 (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true"))))
	(gen `(tor)
	     `((python
		(cell
		 (imports (argparse))
		 (setf parser (argparse.ArgumentParser))
		 ,@(loop for e in cli-args
			 collect
			 (destructuring-bind (&key short long help required action) e
			   `(parser.add_argument
			     (string ,short)
			     (string ,long)
			     :help (string ,help)
			     :required (string ,(if required
					     "True"
					     "False"))
			     :action ,(if action
					  `(string ,action)
					  "None"))))
		 (setf args (parser.parse_args))))
	       (python
		(cell
					;(imports ((plt matplotlib.pyplot)))
					;(plt.ion)
		 (imports (;pathlib
			   time
					;(pd pandas)
					;(np numpy)
					;tqdm
			   hashlib ;; sha1
			   hmac
			   logging
			   operator
			   random
			   socket
			   ssl
			   struct
			   traceback
			   base64 ;; b64decode b16encode b16decode
			   os ;; urandom
			   sys ;; exit
			   time ;; time
			   cryptography.hazmat.backends ;; default_backend
			   cryptography.hazmat.primitives.ciphers ;; Cipher
			   cryptography.hazmat.primitives.ciphers.algorithms ;; AES
			   cryptography.hazmat.primitives.ciphers.modes ;; CTR
			   urllib.request ;; Request urlopen HTTPError
			   urllib.parse ;; urlparse
			   
			   
			   ))

		 		 
		 ))

	       (python
		(cell
		 (setf indexbytes operator.getitem
		       intlist2bytes bytes
		       int2byte (operator.methodcaller (string "to_bytes")
						       1
						       (string "big")))))
	       (python
		(cell
		 (setf start_time (time.time))))
	       (python
		(cell
		 (dot logging
		      (basicConfig :format (string "[%(levelname)s] %(filename)s - %(message)s")
				   :level logging.INFO))
		 (setf log (logging.getLogger __name__))))
	       (python
		(cell
		 (class DirectoryAuthority ()
			(def __init__ (self &key name ip dir_port tor_port)
			  ,@(loop for e in `(name ip dir_port tor_port)
				  collect
				  `(setf (dot self ,e)
					 ,e)))
			(def get_consensus_url (self)
			  (return (dot (string "http://{}:{}/tor/status-vote/current/consensus")
				       (format self.ip
					       self.dir_port))))
			)
		 
		 ))

	       (python
		(cell
		 ,(let ((l `(nickname ip dir_port tor_port identity)))
		  `(class OnionRouter ()
			 (def __init__ (self &key ,@l)
			   ,@(loop for e in l
				   collect
				   `(setf (dot self ,e)
					  ,e))
			   (setf self.flags None
				 self.key_ntor None
				 self._forward_digest None
				 self._backward_digest None
				 self.encryption_key None
				 self.decryption_key None))
			 (def get_descriptor_url (self)
			   (return (dot (string "http://{}:{}/tor/server/fp/{}")
					(format self.ip
						self.dir_port
						self.identity))))
			 (def parse_descriptor (self)
			   (setf headers
				 (dict
				  ((string "User-Agent")
				   (string "Mozilla/5.0 (Windows NT 6.1; rv:60.0) Gecko/20100101 Firefox/60.0"))))
			   (setf request (urllib.request.Request
					  :url (self.get_descriptor_url)
					  :headers headers)
				 response (urllib.request.urlopen
					   request
					   :timeout 8))
			   (for (line response)
				(setf line (line.decode))
				(when (line.startwith (string "ntor-onion-key "))
				  (self.key_ntor (dot line
						      (aref (split (string "ntor-onion-key"))
							    1)
						      (strip)))
				  (unless (== (aref self.key_ntor -1)
					      (string "="))
				    (incf self.key_ntor (string "=")))
				  break)))
			 ))
		 
		 ))
	       )))))
  #+nil (progn
   (sb-ext:run-program "/usr/bin/sh"
		       `("/home/martin/stage/cl-py-generator/example/86_playwright/source/setup01_nbdev.sh"))
   (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC )))




