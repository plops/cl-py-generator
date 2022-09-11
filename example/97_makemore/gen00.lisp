(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "97_makemore")
  (defparameter *idx* "00")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                            ,@rest)))))

  (let* ((notebook-name "makemore")
	 (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-notebook
     :nb-file (format nil "~a/source/~a_~a.ipynb" *path* *idx* notebook-name)
     :nb-code
     `((python
	(export
	 ,(format nil "#|default_exp p~a_~a" *idx* notebook-name)))
       (python (export
		(do0
		 (comments "this file is based on https://github.com/fastai/course22/blob/master/05-linear-model-and-neural-net-from-scratch.ipynb")
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
	       debug True)))

       (python
	(export
	 (do0 (setf parser (argparse.ArgumentParser))
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help required action) e
			`(parser.add_argument
			  (string ,short)
			  (string ,long)
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

       #+nil
       (python
	(export
	 (setf path (pathlib.Path (string "titanic")))
	 (unless (path.exists)
	   (do0
	    (imports (zipfile))
	    (dot zipfile
		 (ZipFile (fstring "/content/drive/MyDrive/{path}.zip"))
		 (extractall path))))
	 ))

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
       ,@(loop for e in `(min max)
	       collect
	       `(python
		 (export
		  (,e (for-generator (w words)
				     (len w)))
		  )))

       (python
	(do0
	 (comments "collect statistics for pairs of characters")
	 (setf b "{}")
	 (for (w words)
	      (setf chs (+ (list (string "<S>"))
			   ("list" w)
			   (list (string "<E>"))))
	      (for (bigram
		    (zip chs (aref chs (slice 1 ""))))
		   (setf (aref b bigram)
			 (+ (b.get bigram 0)
			    1))))))
       (python
	(export
	 (comments "show statistics sorted by frequency")
	 (sorted (b.items)
		 :key (lambda (kv)
			(* -1  (aref kv 1))))))
       (python
	(export
	 (setf character_set
	       (sorted ("list"
			(set
			 (dot (string "")
			      (join words))))))
	 (len character_set)))
       (python
	(export
	 (setf stoi (curly (for-generator ((ntuple i s)
					   (enumerate character_set))
					  "s:i+1")))
	 #+nil(setf (aref stoi (string "<S>")) 26
		    (aref stoi (string "<E>")) 27
		    )
	 (setf (aref stoi (string ".")) 0

	       )
	 stoi))
       (python
	(export
	 (comments "invert lookup")
	 (setf itos (curly
		     (for-generator ((ntuple s i)
				     (stoi.items))
				    "i:s")
		     ))))
       (python
	(export
	 (comments "2d array is more convenient")
	 (setf number_tokens (len stoi))
	 (setf N (torch.zeros (tuple number_tokens
				     number_tokens)
			      :dtype torch.int32))
	 (for (w words)
	      (setf chs (+ (list (string "."))
			   ("list" w)
			   (list (string "."))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (incf (aref N ix1 ix2))))))
       (python
	(do0
	 (imshow N)))

       (python
	(do0
	 (figure :figsize (tuple 16 16))
	 (imshow N :cmap (string "Blues"))
	 (for (i (range number_tokens))
	      (for (j (range number_tokens))
		   (setf chstr (+ (aref itos i)
				  (aref itos j)))
		   (text j i chstr :ha (string "center")
			 :va (string "bottom")
			 :color (string "gray"))
		   (text j i (dot (aref N i j)
				  (item))
			 :ha (string "center")
			 :va (string "top")
			 :color (string "gray"))))
	 (plt.axis (string "off"))
	 ))

       (python
	(do0
	 (setf p (dot (aref N 0)
		      (float))
	       p (/ p (p.sum)))
	 p))

       (python
	(do0
	 (setf g (dot torch
		      (Generator)
		      (manual_seed 2147483647))
	       p (dot torch (rand 3 :generator g))
	       p (/ p (p.sum)))))
       (python
	(do0
	 (torch.multinomial p
			    :num_samples 20
			    :replacement True
			    :generator g)))

       (python
	(do0
	 (comments "https://pytorch.org/docs/stable/notes/broadcasting.html")
	 (comments "adding one for model smoothing (we don't want zeros in the matrix)")
	 (setf P (dot (+ N 1)  (float))
	       P (/ P
		    (P.sum 1 :keepdim True)))
	 ))

       (python
	(do0
	 (setf log_likelihood 0.0
	       n 0)
	 (for (w (list (string "andrej")))
	      (setf chs (+ (list (string "."))
			   ("list" w)
			   (list (string "."))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (setf prob (aref P ix1 ix2)
			 logprob (torch.log prob))
		   (incf log_likelihood logprob)
		   (incf n)
		   (comments "everything with probability higher than 4% is better than random")
		   (print (fstring "{ch1}{ch2}: {prob:.4f} {logprob:.4f}"))))
	 (print (fstring "{log_likelihood=}"))
	 (comments "we are intersted in the product of all probabilities. this would be a small number so we look at the log")
	 (comments "look at negative log_likelihood. the lowest we can get is 0")
	 (setf nll (* -1 log_likelihood))
	 (print (fstring "{nll=}"))
	 (comments "normalized log likelihood is what we use")
	 (comments "normalized log likelihood of the training model is 2.454")
	 (print (fstring "{nll/n:.3f}"))
	 ))

       (python
	(do0
	 (setf xs (list)
	       ys (list))
	 (for (w (aref words (slice "" 1)))
	      (setf chs (+ (list (string "."))
			   ("list" w)
			   (list (string "."))))
	      (for ((ntuple ch1 ch2)
		    (zip chs (aref chs (slice 1 ""))))
		   (setf ix1 (aref stoi ch1)
			 ix2 (aref stoi ch2))
		   (print ch1 ch2)
		   (dot xs (append ix1))
		   (dot ys (append ix2))))
	 (setf xs (tensor xs)
	       ys (tensor ys))

	 (comments "encode integers with one-hot encoding")
	 (setf xenc (dot (F.one_hot xs :num_classes number_tokens)
			 (float)))
	 (imshow xenc)
	 ))

       (python
	(do0
	 (do0
	  (setf g (dot
		   torch
		   (Generator)
		   (manual_seed 2147483647)))
	  (setf W (torch.randn (tuple 27 27)
			       :generator g
			       :requires_grad True)))))
       (python
	(do0

	 
	 

	 (comments "output is 5x27 @ 27x27 = 5x27"
		   "27 neurons on 5 inputs"
		   "what is the firing rate of the 27 neurons on everyone of the 5 inputs"
		   "xenc @ W [3,13] indicates the firing rate of the 13 neuron for input 3. it is a dot-product of the 13th column of W with the input xenc"
		   "we exponentiate the numbers. negative numbers will be 0..1, positive numbers will be >1"
		   "we will interpret them as something equivalent to count (positive numbers). this is called logits. equivalent to the counts in the N matrix"
		   "converting logits to probabilities is called softmax")

	 (setf logits (@ xenc W)
	       counts (dot logits (exp))
	       probs (/ counts
			(counts.sum 1 :keepdims True)))
	 probs
	 (setf loss
	       (* -1 (dot
		 (aref probs
		       (torch.arange 5)
		       ys)
		 (log)
		 (mean))))
	 (print (loss.item))
	 (comments "this is the forward pass")
	 )
	)
       (python
	(do0
	 (comments "backward pass"
		   "clear gradient")
	 (setf W.grad None)
	 (loss.backward)
	 (incf W.data (* -.1 W.grad))
	 ))

       )))
  #+nil
  (sb-ext:run-program "/usr/bin/ssh"
		      `("c11"
			"cd ~/arch/home/martin/stage/cl-py-generator/example/97_makemore/source; nbdev_export")))

