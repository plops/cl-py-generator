(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "97_makemore")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun init-lprint ()
    `(def lprint (args msg)
       (when args.verbose
	 (print (dot (string "{} {}")
                     (format  (- (time.time) start_time)
                              msg))))))
  (defun lprint (&key (msg "") vars)
    `(lprint args (dot (string ,(format nil "~a ~{~a={}~^ ~}" msg vars))
                  (format ,@vars))))

  (let* ((notebook-name "makemore")
	 (cli-args `(
		     (:short "v" :long "verbose" :help "enable verbose output" :action "store_true" :required nil :nb-init True))))
    (write-notebook
     :nb-file (format nil "~a/source/~a_~a.ipynb" *path* *idx* notebook-name)
     :nb-code
     `((python
	(export
	 ,(format nil "#|default_exp p~a_~a" *idx* notebook-name)))
       (python (export
		(do0
					;(comments "this file is based on ")
					;"%matplotlib notebook"
		 #-nil(do0

		       (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		       (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation)
					;(xrp xarray.plot)
				 ))

		       (plt.ion)
					;(plt.ioff)
		       ;;(setf font (dict ((string size) (string 6))))
		       ;; (matplotlib.rc (string "font") **font)
		       )
		 (imports (	os
					;sys
				time
					;docopt
				pathlib
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
					; (np numpy)
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

					;(np jax.numpy)
					;(mpf mplfinance)
					;selenium.webdriver ;.FirefoxOptions
				argparse
				torch
				))
		 "import torch.nn.functional as F"
		 (imports-from (torch tensor))



		 #-nil
		 (imports-from  (matplotlib.pyplot
				 plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid text
				 legend figure gcf xlim ylim)
				)

		 )
		))

       (python
	(do0
	 (class Args ()
		(def __init__ (self)
		 ,@(loop for e in cli-args
			 collect
			 (destructuring-bind (&key short long help required action nb-init) e
			   `(setf (dot self ,long) ,nb-init)))))
	 (setf args (Args))))
       (python
	(export
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

	 (setf start_time (time.time)
	       debug True)
	 ,(init-lprint)))


      

       (python
	(export
	 (do0 (setf parser (argparse.ArgumentParser))
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help required action nb-init) e
			`(parser.add_argument
			  (string ,(format nil "-~a" short))
			  (string ,(format nil "--~a" long))
			  :help (string ,help)
					;:required
			  #+nil
			  (string ,(if required
				       "True"
				       "False"))
			  :action ,(if action
				       `(string ,action)
				       "None"))))

	      (setf args (parser.parse_args)))))

       (markdown "multi layer perceptron https://www.youtube.com/watch?v=TCH_1BHY58I "
		 "reference bengio 2003 neural probabilistic language model")

       (python
	(export
	 (setf words (dot
		      (open (string "/home/martin/stage/cl-py-generator/example/97_makemore/source/names.txt")
			    (string "r"))
		      (read)
		      (splitlines)
		      ))
	 (aref words (slice "" 10)))
	)
       (python
	(export
	 (setf chars (sorted ("list" (set (dot (string "")
					       (join words))))))
	 (setf stoi (curly (for-generator ((ntuple i s)
					   (enumerate chars))
					  (slice s (+ i 1))))
	       (aref stoi (string ".")) 0
	       itos (curly (for-generator ((ntuple s i)
					   (stoi.items))
					  (slice i s))))
	 (print itos)))
       ,@(let* ((block-size 3)
		(n2 2)
		(n100 100) ;; hidden layer
		(n6 (* n2 block-size))
		(n27 27))
	   `((python
	      (export
	       (setf block_size ,block-size
		     X (list)
		     Y (list))
	       (for (w (aref words (slice "" 5)))
		    (print w)
		    (setf context (* (list 0)
				     block_size))
		    (for (ch (+ w (string ".")))
			 (setf ix (aref stoi ch))
			 (X.append context)
			 (Y.append ix)
			 (print (dot (string "")
				     (join (for-generator (i context)
							  (aref itos i))))
				(string "--->")
				(aref itos ix))
			 (setf context (+ (aref context (slice 1 ""))
					  (list ix)))))
	       (setf X (tensor X)
		     Y (tensor Y))))
	     (python
	      (export
	       (setf C (torch.randn
			(tuple ,n27 ,n2)))))
	     (python
	      (export
	       (@
		(dot (F.one_hot (tensor 5)
				:num_classes ,n27)
		     (float))
		C)
	       ))
	     (python
	      (do0
	       (setf emb (aref C X))
	       (do0
		(setf W1 (torch.randn (tuple ,n6 ,n100))
		      b1 (torch.randn (tuple ,n100)))
		(setf W2 (torch.randn (tuple ,n100 ,n27))
		      b2 (torch.randn (tuple ,n27))))

	       ))
	     (python
	      (do0
	       (setf h
		     (torch.tanh
		      (+ (@ (dot emb
				 (view -1 ,n6)) W1)
			 b1)))))
	     (python
	      (do0
	       (setf logits (+ (@ h W2)
			       b2)
		     counts (logits.exp)
		     prob (/ counts
			     (counts.sum 1 :keepdims True)))

	       ))
	     (python
	      (do0
	       (comments "negative log-likelihood loss")
	       (setf loss
		     (* -1
			(dot (aref prob
				   (torch.arange 32)
				   Y)
			     (log)
			     (mean))))))

	     (python
	      (export
	       (setf g (dot torch
			    (Generator)
			    (manual_seed 2147483647))
		     C (torch.randn (tuple ,n27 ,n2)
				    :generator g)
		     )
	       (do0
		(setf W1 (torch.randn (tuple ,n6 ,n100))
		      b1 (torch.randn (tuple ,n100)))
		(setf W2 (torch.randn (tuple ,n100 ,n27))
		      b2 (torch.randn (tuple ,n27))))
	       (setf parameters (list C W1 b1 W2 b2))
	       (setf n_parameters (sum (for-generator (p parameters)
						      (p.nelement))))
	       ,(lprint :vars `(n_parameters))
	       (setf emb (aref C X))
	       (setf logits (+ (@ h W2)
			       b2)
		     counts (logits.exp)
		     prob (/ counts
			     (counts.sum 1 :keepdims True)))
	       (setf loss
		     (* -1
			(dot (aref prob
				   (torch.arange 32)
				   Y)
			     (log)
			     (mean))))
	       ))
	     ))

       ))))

