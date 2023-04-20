(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "106_sar")
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
  
  (let* ((notebook-name "sar_simulator")
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
	  
	,(doc `((:name c :val 3e8 :unit m/s :help "speed of light" )
		(:name fc :val 5.5e9  :unit Hz :help "center frequency")
		(:name wavelength :val (/ c fc) :unit m)
		(:name platform_speed :val 100 :unit m/s )
		(:name altitude :val 5000 :unit m :help "platform altitude")
		(:name duration :val 10 :unit s :help "simulation duration")
		(:name sample_rate :val 1000 :unit Hz)
		(:name bw :val 50e6 :unit Hz :help "bandwidth")
		(:name T_p :val 10e-6 :unit s :help "pulse duration")
		(:name K :val (/ bw T_p) :unit Hz/s :help "chirp rate")
		(:name scatter_points  :val (np.array (list (list 3000 5000)
							    (list 5000 8000)
							    (list 8000 15000)))
		 :unit m :help "scattering targets on the ground plane")
		(:name tt :val (np.arange 0 duration (/ 1 sample_rate)) :unit s :help "time array")
		(:name platform_positions :val (* platform_speed tt) :unit m :help "position array")
		(:name slant_ranges :val (np.sqrt (+ (** altitude 2)
						     (** (- (aref scatter_points ":" 0)
							    (aref platform_positions ":" np.newaxis))
							 2)))
		 :unit m :help "slant ranges for each scatter point")
		(:name time_delays :val (* 2 (/ slant_ranges c)) :unit s :help "round-trip time delay")
		(:name t_chirp :val (np.arange 0 T_p (/ 1 sample_rate)) :unit s :help "time axis for pulse")
		(:name transmitted_signal :val (np.exp (* 1j np.pi K (** t_chirp 2) ))
		 :unit amplitude :help "chirped radar pulse amplitude")
		(:name v_radial :val (/ (* -platform_speed
					   (- (aref scatter_points ":" 0)
					      (aref platform_positions ":" np.newaxis)))
					slant_ranges)
		       ;; m/s * m / m
		 :unit m/s
		 :help "radial speed difference between platform and target")
		(:name doppler_shifts :val (* 2 (/ v_radial wavelength))
		       ;; m/s / m
		 :unit Hz)
		(:name received_signal :val (np.zeros (tuple (len tt)
							     (len transmitted_signal))
						      :dtype complex)
		 :unit amplitude)
			 
					;(:name :val :unit :help)
		))

	(for (scatter_idx (range (dot scatter_points
				      (aref shape 0))))
	     (setf delay_samples (dot np (round (* (aref time_delays ":" scatter_idx)
						   sample_rate))
				      (astype int)))
	     (for ((ntuple idx delay)
		   (enumerate delay_samples))
		  (when (and (<= 0 delay )
			     (< delay (len transmitted_signal)))
		    (incf (aref received_signal idx)
			  (* (np.exp (* 1j 2 np.pi
					(aref doppler_shifts idx scatter_idx)
					t_chirp))
			     (aref transmitted_signal
				   (slice delay "")
				   ;np.newaxis
				   #+nil (- "0:len(transmitted_signal)"
				      delay)
				   #+nil (- (slice 0 (len transmitted_signal))
					    delay)))))))

	,(doc
	  `(;(:name :val :unit :help)
	    (:name noise_level :val 1e-5 :unit amplitude )
	    (:name received_signal :val (+ received_signal
					   (* noise_level
					      (+ (np.random.randn *received_signal.shape)
						 (* 1j (np.random.randn *received_signal.shape))))))
	    )
	  )
	  

	  
	)))))

