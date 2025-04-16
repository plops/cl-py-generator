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
       (class PipeReader (threading.Thread)
	      (string3 "Reads liens from a file descriptor (pipe) into a queue.")
	      (def __init__ (self pipe_fd output_queue &key (prefix (string "")))
		(declare (type queue.Queue output_queue)
			 (type str prefix))
		(dot (super)
		     (__init__ :daemon True))
		(comments "Daemon thread exit when main thread exits")
		(setf self._pipe_fd pipe_fd
		      self.output_queue output_queue
		      self._prefix prefix
		      self._running True))
	      (def run (self)
		(string3 "Read lines until the pipe closes.")
		(try
		 (do0
		  (for (line (iter self._pipe_fd.readline
				   (string-b "")))
		       (unless self._running
			 break)
		       (sef.output_queue.put (fstring "{self._prefix} {line.decode(errors='relpace').strip()}")))
		  (comments "Signal EOF")
		  (self.output_queue.put None))
		 ("Exception as e"
		  (logger.error (fstring "Error reading pipe: {e}"))
		  (comments "Signal EOF on error")
		  (self.output_queue.put None))
		 (finally
		  (try
		   (self._pipe_fd.close)
		   ("OSError"
		    (comments "Ignore errors closing already closed pipe")
		    pass)))))
	      (def stop (self)
		(self._running False))
	      (def join (self &key (timeout None))
		(self.stop)
		(dot (super) (join timeout)))
	      )

       ;; core logic
       (def run_self_on_remote (remote_host_spec target_port)
	 (declare (type str remote_host_spec)
		  (type int target_port)
		  (values None))
	 (string3 "
    Executes this script on the remote host via SSH.

    Args:
        remote_host_spec: SSH destination (e.g., user@hostname).
        target_port: The UDP port to use for punching.
")
	 (setf script_path (os.path.abspath __file__))
	 (comments "Use '-T' for non-interactive session, disable pseudo-tty allocation"
		   "Pass script via stdin using '-'"
		   "Pass port and a flag '--remote-instance' to indicate server-side execution")
	 (setf ssh_command (list (string "ssh")
				 (string "-T")
				 remote_host_spec
				 (string "python3")
				 (string "-")
				 (str target_port)
				 (string "--remote-instance")))
	 (logger.info (fstring "Attempting to start remote puncher via SSH: {' '.join(ssh_command)}"))
	 (try
	  (do0
	   (with (as (open script_path (string "rb"))
		     script_file)
		 (setf process (subprocess.Popen
				ssh_command
				:stdin script_file
				:stdout subprocess.PIPE
				:stderr subprocess.PIPE)))
	   (setf "log_queue: queue.Queue[Optional[str]]"
		 (queue.Queue))
	   (setf stdout_reader (PipeReader process.stdout
					   log_queue
					   :prefix (string "[REMOTE STDOUT] ")))
	   (setf stderr_reader (PipeReader process.stdout
					   log_queue
					   :prefix (string "[REMOTE STDERR] ")))
	   (stdout_reader.start)
	   (stderr_reader.start)
	   (setf active_readers 2)
	   (while (< 0 active_readers)
		  (try
		   (do0
		    (comments "Wait for logs")
		    (setf log_line (log_queue.get :timeout 1.0))
		    (if (is log_line None)
			(decf active_readers)
			(logger.info log_line)))
		   (queue.Empty
		    (comments "Check if process terminated unexpectedly")
		    (when (and "process.poll() is not None"
			       (< 0 active_readers))
		      (logger.warning (string "Remote process terminated unexpectedly. Flushing logs."))
		      (comments "Flush remaining queue items quickly")
		      (while (< 0 active_readers)
			     (try
			      (do0
			       (setf log_line (log_queue.get_nowait))
			       (if (is log_line None)
				   (decf active_readers)
				   (logger.info log_line)))
			      (queue.Empty
			       (comments "Queue empty, likely means readers finished after poll check")
			       break)))
		      (comments "Assum readers finished if process is done")
		      (setf active_readers 0)
		      ))))
	   (comments "Ensure threads are joined")
	   (stdout_reader.join :timeout 1.0)
	   (stderr_reader.join :timeout 1.0)
	   (setf return_code (process.wait))
	   (logger.info (fstring "Remote process finished with exit code: {return_code}"))
	   (when (!= 0 return_code)
	     (logger.error (string "Remote script execution failed."))))

	  (FileNotFoundError
	   (logger.exception (fstring "Error: Could not find script '{script_path}' or ssh command.")))
	  ("Exception as e"
	   (logger.exception (fstring "Error running script on remote host: {e}"))))
	 )

       (def perform_udp_punch (remote_ip target_port)
	 (declare (values bool)
		  (type str remote_ip)
		  (type int target_port))
	 (string3 "
    Attempts to perform UDP hole punching with the specified remote peer.

    Args:
        remote_ip: The IP address of the remote peer.
        target_port: The UDP port to use for punching (must be the same on both ends).

    Returns:
        True if the hole punch was likely successful (ACK received), False otherwise.")
	 (setf local_token (dot secrets
				(token_hex TOKEN_BYTE_LENGTH)
				(encode))
	       "received_remote_token: Optional[bytes]" None
	       ack_received False
	       ack_sent False)
	 (logger.info
	  (fstring "Attempting UDP punch to {remote_ip}:{target_port}"))
	 (logger.info
	  (fstring "Binding local UDP socket to port {target_port}"))
	 (logger.debug
	  (fstring "Local Token: {local_token.decode()}"))

	 (try
	  (do0
	   (with (as (socket.socket socket.AF_INET
				    socket.SOCK_DGRAM)
		     sock)
		 (sock.bind (tuple (string "0.0.0.0")
				   target_port))
		 (sock.setblocking False)
		 (setf remote_addr (tuple remote_ip target_port))
		 (for (attempt (range MAX_ATTEMPTS))
		      (when ack_received
			(logger.info (string "Acknowledgement received. Hole punch appears successful."))
			break)
		      (logger.debug (string "====== Attempt #{attempt+1}/{MAX_ATTEMPTS}"))
		      (comments "Prepare message to send")
		      (setf message_parts (list local_token))
		      (if received_remote_token
			  (do0
			   (message.parts.append received_remote_token)
			   (when (or ack_received
				     ack_sent)
			     (message_parts.append ACK_FLAG)
			     (setf ack_sent True)))
			  (do0
			   (comments "Placeholder until remote token known")
			   (message.parts.append NULL_TOKEN_PLACEHOLDER)))
		      (setf message_to_send (dot
					     MSG_SEPARATOR
					     (join message_parts)))
		      (setf (ntuple readable writable exceptional
				    )
			    (select.select (list sock)
					   (list sock)
					   (list sock)
					   DEFAULT_TIMEOUT_SECONDS))
		      (when exceptional
			(logger.error (string "Socket exception detected!"))
			(return False))

		      (when writable
			(try
			 (do0
			  (sock.sendto message_to_send remote_addr)
			  (logger.debug (fstring "Sent: {message_to_send!r}"))
			  )
			 ("socket.error as e"
			  (logger.warning (fstring "Socket error sending data: {e}")))
			 ("Exception as e"
			  (logger.error (fstring "Unexpected error sending data: {e}"))
			  (return False))))
		      (when readable
			(try
			 (do0
			  (setf (ntuple data addr)
				(sock.recvfrom BUFFER_SIZE))
			  (logger.debug (fstring "Received: {data!r} from {addr}"))
			  (comments "Ignore packets not from target IP, port might change due to NAT so only check IP")
			  (unless (== remote_ip
				      (aref addr 0))
			    (logger.warning (fstring  "Received packet from unexpected IP {addr[0]}. Ignoring."))
			    continue)
			  (setf parts (data.split
				       MSG_SEPARATOR))
			  (unless parts
			    (logger.warning (string "Received empty packet."))
			    continue)
			  (comments "Process receieved remote token")
			  (setf current_remote_token (aref parts 0))
			  (unless (== NULL_TOKEN_PLACEHOLDER
				      current_remote_token)
			    (cond ((is received_remote_token None)
				   (setf received_remote_token
					 current_remote_token)
				   (logger.info (fstring "Received remote token: {received_remote_token.decode()}")))
				  ((!= received_remote_token
				       current_remote_token)
				   (logger.warning (fstring "Received conflicting remote token {current_remote_token.decode()}. Sticking with wirst."))
				   continue)))
			  (comments "Check  for acknowledgement")
			  (when (and (<= 3 (len parts))
				     (== local_token (aref parts 1))
				     (== ACK_FLAG (aref parts 2)))
			    (if (and received_remote_token
				     (== received_remote_token aref parts 0))
				(do0
				 (logger.info (string 'ACK received from remote peer with matching tokens.))
				 (setf ack_received True)
				 (unless ack_sent
				   (setf final_ack_msg
					 (dot MSG_SEPARATOR
					      (join (list local_token
							  received_remote_token
							  ACK_FLAG))))
				   (when writable
				     (sock.sendto final_ack_msg remote_addr)
				     (logger.debug (fstring "Sent final ACK: {final_ack_msg!r}"))
				     (setf ack_sent True))))
				(do0
				 (logger.warning
				  (fstring "Received ACK but remote token mismatch: Expected {received_remote_token.decode() if received_remote_token else 'None'}, Got {parts[0].decode()}")))))
			  )
			 (BlockingIOError
			  (logger.debug (string "Socket recv would block. This is expected if select indicated  readability but nothing was there *yet*."))
			  pass)
			 ("socket.error as e"
			  (logger.warning (fstring "Socket error receiving data: {e}")))
			 ("Exception as e"
			  (logger.exception (fstring "Error processing received data: {e}"))))
			)
		      (unless ack_received
			(time.sleep RETRY_DELAY_SECONDS)))
		 (when ack_received
		   (logger.info (fstring "UDP hole punch to {remote_if}:{target_port} successful!"))
		   (return True))
		 (logger.error (fstring "Failed to establish UDP connection after {MAX_ATTEMPTS} attempts."))
		 (return False)))
	  ("socket.error as e"
	   (logger.error (fstring "Socket setup error: {e}. Check port permission/availability."))
	   (return False))
	  ("Exception as e"
	   (logger.exception (fstring "An unexpected error occurred during UDP punching: {e}"))
	   (return False))))

       (def main ()
	 (setf parser (argparse.ArgumentParser
		       :description (string "UDP Hole Punching Tool. Runs locally and initiates remote execution via SSH.")
		       :formatter_class argparse.ArgumentDefaultsHelpFormatter))
	 (parser.add_argument (string "-H")
			      (string "--remote_spec")
			      :nargs (string "?")
			      :help (string "Remote server specification (e.g., usre@hostname.com or IP). Required unless running as --remote-instance."))
	 (parser.add_argument (string "-p" ) (string "--port")
			      :type int
			      :help (string "The UDP port number for hole punching (must be same on both ends)."))
	 (parser.add_argument (string "-r") (string "--remote-instance")
			      :action (string "store_true")
			      :help (string "Internal flag: Indicates this script instance is running on the remote server."))
	 (parser.add_argument (string "-v")
			      (string "--verbose")
			      :action (string "store_true")
			      :help (string "Enable verbose debug logging."))
	 (setf args (parser.parse_args))
	 (when args.verbose
	   (logger.setLevel logging.DEBUG)
	   (for (handler (dot logging
			      (getLogger)
			      handlers))
		(handler.setLevel logging.DEBUG))
	   (logger.debug (string "Verbose logging enabled.")))
	 (if args.remote_instance
	     (do0
	      (setf (dot threading
			 (current_thread)
			 name)
		    (string "RemotePuncher"))
	      (logger.info (fstring "Running as REMOTE instance on port {args.port}"))
	      (try
	       (do0
		(setf ssh_client_env (os.environ.get (string "SSH_CLIENT")))
		(unless ssh_client_env
		  (logger.error (string "SSH_CLIENT environment variable not found. Cannot determine client IP."))
		  (sys.exit 1))
		(setf client_ip (dot ssh_client_env (aref (split) 0)))
		(logger.info (fstring "Inferred Client IP from SSH_CLIENT: {client_ip}")))
	       ("(IndexError, AttributeError)"
		(logger.error (string "Could not parse SSH_CLIENT environment variable."))
		(sys.exit 1))
	       ("Exception as e"
		(logger.exception (fstring "Error obtaining client IP: {e}"))
		(sys.exit 1)))
	      (unless client_ip
		(logger.error (string "Client IP address could not be determined."))
		(sys.exit 1))
	      (setf sucess (perform_udp_punch client_ip args.port))
	      (sys.exit (? success 0 1))
	      )
	     (do0
	      (comments "Running on the client")
	      (setf (dot threading
			 (current_thread)
			 name)
		    (string "ClientMain"))
	      (unless args.remote_spec
		(parser.error (string "The 'remote_spec' argument is required when running as the client.")))
	      (logger.info (fstring "Running as CLIENT, targeting {args.remote_spec} on port {args.port}"))
	      (try
	       (do0
		(setf host_part (dot args
				     remote_spec
				     (aref (split (string "@"))
					   -1))
		      remote_ip (socket.gethostbyname host_part)
		      logger.info (fstring "Resolved {host_part} to IP address: {remote_ip}")))
	       ("socket.gaierror as e"
		(logger.error (fstring "Could not resolve hostname {host_part}: {e}"))
		(sys.exit 1))
	       ("Exception as e"
		(logger.exception (fstring "Error resolving remote host IP: {e}"))
		(sys.exit 1)))
	      (comments "Event to signal success from puncher thread")
	      (setf puncher_success (threading.Event))
	      (setf puncher_thread_local None
		    remote_thread None)
	      (def local_punch_target ()
		(when (perform_udp_punch remote_ip args.port)
		  (puncher_success.set)))
	      (try
	       (do0
		(setf puncher_thread_local (threading.Thread
					    :target local_punch_target
					    :name (string "LocalPuncher")
					    :daemon True))
		(puncher_thread_local.start)
		(setf remote_thread (threading.Thread
				     :target run_self_on_remote
				     :args (tuple args.remote_spec args.port)
				     :name (string "RemoteExecutor")
				     :daemon True))
		(remote_thread.start)
		(comments "Wait for local puncher OR remote executor to finish"
			  "We prioritize local success signal, but also need to wait for remote logs")
		(while (and (puncher_thread_local.is_alive)
			    (remote_thread.is_alive))
		       (when (puncher_success.is_set)
			 (logger.info (string "Local puncher signaled success. Waiting for remote process output."))
			 break)
		       (time.sleep .5))
		(logger.info (string "Waiting for remote execution thread to complet to get all logs ..."))
		(remote_thread.joint)
		(when (puncher_thread_local.is_alive)
		  (logger.info (string "Waiting for local puncher thread to complete..."))
		  (puncher_thread_local.join :timeout (* 2 DEFAULT_TIMEOUT_SECONDS)))
		(if (puncher_success.is_set)
		    (do0 (logger.info (string "UDP Hole Punching process completed successfully."))
			 (sys.exit 0))
		    (do0
		     (logger.error (string "UDP Hole Punching process failed"))
		     (sys.exit 1))))
	       ("(KeyboardInterrupt,SystemExit)"
		(logger.info (string "Caught interrupt, exiting..."))
		(sys.exit 1))
	       ("Exception as e"
		(logger.exception (string "An unexpected error occurred in the main client process."))
		(sys.exit 1))))))

       (when (== __name__ (string "main"))
	 (main))
       )))
  )
