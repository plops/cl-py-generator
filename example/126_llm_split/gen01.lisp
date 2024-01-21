(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "126_llm_split")
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
  
  (let* ((notebook-name "segment")
	 (cli-args `((:short "-c" :long "--chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     (:short "-p" :long "--prompt" :type str
		      :default "Summarize the following video transcript as a bullet list."
		      :help "The prompt to be prepended to the output file(s)."))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (do0
	)
       (do0
	(comments "python -m venv ~/mediapipe_env; . ~/mediapipe_env/bin/activate; python -m pip install mediapipe mss"
		  "pip install --user tiktoken"
		  )
	#+nil
	(do0
	 
	 (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
	 (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation)
					;(xrp xarray.plot)
		   ))

					;(plt.ion)
	 (plt.ioff)
	 ;;(setf font (dict ((string size) (string 6))))
	 ;; (matplotlib.rc (string "font") **font)
	 )
	#+nil 
	(imports-from  (matplotlib.pyplot
			plot imshow tight_layout xlabel ylabel
			title subplot subplot2grid grid text
			legend figure gcf xlim ylim)
		       )
	(imports (			;	os
					;sys
		  time
					;docopt
					;pathlib
					; (np numpy)
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
					;(np numpy)
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
					
					;math
			
					;(np jax.numpy)
					;(mpf mplfinance)
					;(fft scipy.fftpack)
		  argparse
					;torch
					;(mp mediapipe)
					;mss
		  ;(cv cv2)
		  tiktoken
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
       (def split_document (input_file chunk_size prompt)
	 (string3 "given an input file split it into several files with at most chunk_size words each. prepend with prompt. replace newlines with space.")
	 (with (as (open input_file (string "r"))
		   f)
	       (setf text (f.read)))
	 (setf words (text.split)
	       chunks (list)
	       current_chunk (list))
	 (setf word_count 0)
	 (for (word words)
	      (current_chunk.append word)
	      (incf word_count)
	      (when (<= chunk_size word_count)
		(chunks.append (dot (string " ")
				    (join current_chunk)))
		(setf current_chunk (list))
		(setf word_count 0)))
	 (when current_chunk
	   (chunks.append (dot (string " ")
			       (join current_chunk))))
	 (for ((ntuple i chunk)
	       (enumerate chunks))
	      (setf output_file
		    (dot (string "{}.{}")
			 (format (aref (input_file.split (string ".")) 0)
				 (dot (str (+ i 1))
				      (zfill 2)))))
	      (with (as (open output_file (string "w"))
			f)
		    (f.write (+ prompt (string "\\n```\\n")))
		    (f.write chunk)
		    (f.write (string "\\n```"))))
	 )

       (when (== __name__ (string "__main__"))
	 (do0 (setf parser (argparse.ArgumentParser :description (string "Split a document into chunks. I use this to make summaries with chatgpt4")))
	      (parser.add_argument (string "input_file")
				   :type str
				   :help (string "The input text file to split."))
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help type default required action nb-init) e
			`(parser.add_argument
			  (string ,(format nil "-~a" short))
			  (string ,(format nil "--~a" long))
			  :help (string ,help)
					;:required
			  #+nil
			  (string ,(if required
				       "True"
				       "False"))
			  :default ,(if default
					default
					"None")
			  :type ,(if type
					type
					"None")
			  :action ,(if action
				       `(string ,action)
				       "None"))))

	      (setf args (parser.parse_args))))
       ))))

