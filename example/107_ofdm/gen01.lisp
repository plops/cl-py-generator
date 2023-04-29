(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "107_ofdm")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    msg
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    vars)))
                   (format  (- (time.time) start_time)
                            ,@vars)))))

  
  (defun doc (def)
    `(do0
      ,@(loop for e in def
	      collect
	      (destructuring-bind (&key name val (unit "-") (help name)) e
		`(do0
		  (comments ,(format nil "~a (~a)" help unit))
		  (setf ,name ,val))))))
  
  (let* ((notebook-name "ofdm")
	 #+nil (cli-args `(
			   (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/python"
       (do0
	)
       (do0
	
	(imports (			;	os
					;sys
		  time
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
					;   scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
		  (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
					;re
					;json
					; csv
					;io.StringIO
					;bs4
					;requests
					;mss
			
					;(np jax.numpy)
					;(mpf mplfinance)
		  (fft scipy.fftpack)
					;argparse
					;torch
					;(mp mediapipe)
					
		  )))
	 
       (setf start_time (time.time)
	     debug True)
       (setf
	_code_git_version
	(string ,(let ((str (with-output-to-string (s)
			      (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		   (subseq str 0 (1- (length str)))))
	_code_repository (string ,(format nil
					  "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/"
					  *project*))
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


       (do0

	(class OFDMTransmitter ()
	       (def __init__ (self n_subcarriers data_size)
		 (setf self.n_subcarriers n_subcarriers
		       self.data_size data_size))
	       (def _generate_random_data (self)
		 (setf data (np.random.randint 0 2
					       (tuple self.n_subcarriers
						      self.data_size)))
		 (return (- (* 2 data) 1)))
	       (def _ifft (self data)
		 (return (fft.ifft data :axis 0)))
	       (def _create_schmidl_cox_training_sequence (self)
		 (setf random_symbols (self._generate_random_data)
		       training_sequence (np.vstack
					  (tuple (aref random_symbols ":" 0)
						 (aref random_symbols ":" 0))))
		 (return training_sequence))
	       (def modulate (self)
		 (setf data_symbols (self._generate_random_data)
		       ifft_data (self._ifft data_symbols)
		       training_sequence (self._create_schmidl_cox_training_sequence)
		       ofdm_frame (np.hstack (tuple training_sequence
						    ifft_data))
		       serialized_data (np.reshape ofdm_frame
						   (* self.n_subcarriers
						      (+ 2 self.data_size))))
		 (return serialized_data)))

	(class OFDMReceiver ()
	       (def __init__ (self n_subcarriers data_size)
		 (setf self.n_subcarriers n_subcarriers
		       self.data_size data_size))
	       (def _fft (self data)
		 (return (fft.fft data :axis 0)))
	       (def _schmidl_cox_time_sync (self received_signal)
		 (setf half_len self.n_subcarriers
		       R (np.zeros (- received_signal.size
				      (* 2 half_len))))
		 (for (i (range R.size))
		      (setf first_half (aref received_signal (slice i (+ i half_len)))
			    second_half (aref received_signal (slice (+ i half_len)
								     (+ i (* 2 half_len))))
			    (aref R i) (np.abs (/
						(** (np.sum (* (np.conj first_half)
							       second_half))
						    2)
						(np.sum (* (** (np.abs first_half) 2)
							   (** (np.abs second_half) 2))))
					       )))
		 (setf frame_start (np.argmax R))
		 (return frame_start))
	       (def _schmidl_cox_frequency_sync (self received_signal frame_start)
		 (setf half_len self.n_subcarriers
		       first_half (aref received_signal (slice frame_start
							       (+ frame_start half_len)))
		       second_half (aref received_signal (slice (+ frame_start half_len)
								(+ frame_start (* 2 half_len))
							       ))
		       angle_sum (np.angle (np.sum (* (np.conj first_half)
						      second_half)))
		       cfo_est (/ -angle_sum
				  (* 2 np.pi half_len)))
		 (return cfo_est))
	       (def demodulate (self received_signal)
		 (setf frame_start (self._schmidl_cox_time_sync received))))

	  
	)))))

