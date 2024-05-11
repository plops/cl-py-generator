(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "132_makemore5")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun init-lprint ()
    `(def lprint (msg args)
       (when args.verbose
	 (print (dot (string "{} {}")
                     (format  (- (time.time) start_time)
                              msg))))))
  (defun lprint (&key msg vars)
    `(lprint (dot (string ,(format nil "~@[~a ~]~{~a={}~^ ~}" msg (mapcar (lambda (x) (emit-py :code x))
									  vars)))
                  (format ,@vars))
	     args))

  (let* ((notebook-name "makemore5")
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
				random
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
				tqdm
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

		 (imports-from  (torch
				 linspace
				 randn
				 randint
				 tanh
				 )
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

       
       (python
	"!wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
       (python
	(export
	 (comments "read in all the words")
	 (setf words (dot
		      (open (string "names.txt")
			    (string "r"))
		      (read)
		      (splitlines)
		      ))
	 ,(lprint :vars `((len words)
			 (max (for-generator (w words)
					     (len w)))))
	 (aref words (slice "" 10)))
	)
       (python
	(export
	 (comments "build the vocabulary of characters and mappings to/from integers")
	 (setf chars (sorted ("list" (set (dot (string "")
					       (join words))))))
	 (comments "Create a mapping from character to integer (stoi)"
		   "Start from 1 because 0 is reserved for the end character '.'")
	 (setf stoi (curly (for-generator ((ntuple i s)
					   (enumerate chars))
					  (slice s (+ i 1))))
	       (aref stoi (string ".")) 0
	       )
	 (comments "Create a mapping from integer to character (itos)")
	 (setf itos (curly (for-generator ((ntuple s i)
					   (stoi.items))
					  (slice i s))))
	 ,(lprint :msg "mapping from integer to character" :vars `(itos))))

       (python
	(export
	 (comments "shuffle up the words")
	 (random.seed 42)
	 (random.shuffle words)))

       (python
	(export
	 (comments "build the dataset")
	 (comments "block_size .. context length of how many characters do we take to predict the next one")
	 (setf block_size 3)
	 (def build_dataset (words)
	   (string3 "This function builds a dataset for training a model using the given list of words.
    It creates a context of a certain block size for each character in the words and uses this to predict the next character.

    Args:
        words (list): A list of words to be used for creating the dataset.

    Returns:
        tuple: A tuple containing the input tensor (X) and the target tensor (Y). X is the tensor representing the context for each character, and Y is the tensor representing the characters themselves.")
	   (setf X (list)
		 Y (list))
	   (for (w words)
		(setf context (* (list 0) block_size))
		(for (ch (+ w (string ".")))

		     (comments "The character ch is converted to an integer index ix using the stoi function.")
		     (setf ix (stoi ch))
		     (comments "The current context is appended to X, and the integer index ix is appended to Y.

The context is updated by removing the first element and appending the integer index ix at the end. This means that the context always contains the integer indices of the last block_size characters.")
		     (X.append context)
		     (Y.append ix)
		     (setf context (+ (aref context (slice 1 ""))
				      (list ix)))))
	   (setf X (torch.tensor X)
		 Y (torch.tensor Y)
		 )
	   (comments "Each element in Y is the character that should be predicted given the corresponding context in X.")
	   ,(lprint :vars `(X.shape Y.shape))
	   (return (ntuple X Y)))
	 (comments "Use 80% for training, 10% for validation and 10% for testing. We use the following indices to perform the split.")
	 (setf n1 (int (* .8 (len words)))
	       n2 (int (* .9 (len words))))
	 ,@(loop for e in `((:name tr :slice (slice "" n1) :comment "80%")
			    (:name dev :slice (slice n1 "" n2) :comment "10%")
			    (:name te :slice (slice n2 "") :comment "10%"))
		 collect
		 (destructuring-bind (&key name slice comment) e
		   (let ((x (format nil "X~a" name))
			 (y (format nil "Y~a" name)))
		    `(do0
		      (comments ,comment)
		      (setf (ntuple ,x ,y) (build_dataset (aref words ,slice)))))))
	 ))
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
	       (for (w words		;(aref words (slice "" 5))
		       )
		    (print w)
		    (setf context (* (list 0)
				     block_size))
		    (for (ch (+ w (string ".")))
			 (setf ix (aref stoi ch))
			 (X.append context)
			 (Y.append ix)
			 #+nil (print (dot (string "")
					   (join (for-generator (i context)
								(aref itos i))))
				      (string "--->")
				      (aref itos ix))
			 (setf context (+ (aref context (slice 1 ""))
					  (list ix)))))
	       (setf X (tensor X)
		     Y (tensor Y))))
	     (python
	      (do0
	       (setf C (torch.randn
			(tuple ,n27 ,n2)))))
	     (python
	      (do0
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
	       (do0
		(setf g (dot torch
			     (Generator)
			     (manual_seed 2147483647))
		      C (torch.randn (tuple ,n27 ,n2)
				     :generator g))
		(do0
		 (setf W1 (torch.randn (tuple ,n6 ,n100))
		       b1 (torch.randn (tuple ,n100)))
		 (setf W2 (torch.randn (tuple ,n100 ,n27))
		       b2 (torch.randn (tuple ,n27))))
		(setf parameters (list C W1 b1 W2 b2))
		(for (p parameters)
		     (setf p.requires_grad True))
		(do0 (setf n_parameters (sum (for-generator (p parameters)
							    (p.nelement))))
		     ,(lprint :vars `(n_parameters)))
		)))
	     (python
	      (do0

	       (do0
		(comments "forward")
		(setf emb (aref C X))
		(setf h
		      (torch.tanh
		       (+ (@ (dot emb
				  (view -1 ,n6)) W1)
			  b1)))
		(setf logits (+ (@ h W2)
				b2))
		(setf loss (F.cross_entropy logits Y)))
	       #+nil
	       (do0
		(comments "explicit cross_entropy is not efficient a lot of tensors: ")
		(comments "positive numbers can overflow logits, offset doesn't influence loss")
		(setf logits2 (- logits (torch.max logits))
		      counts (logits2.exp)
		      prob (/ counts
			      (counts.sum 1 :keepdims True)))
		(setf loss
		      (* -1
			 (dot (aref prob
				    (torch.arange 32)
				    Y)
			      (log)
			      (mean)))))
	       (do0
		(comments "backward pass")
		(comments "set gradients to zero")
		(for (p parameters)
		     (setf p.grad None))
		(loss.backward)
		(comments "update")
		(for (p parameters)
		     (incf p.data
			   (* -.1 p.grad)))
		)
	       ))

	     (python
	      (do0
	       (comments "learning loop")

	       (for (_ (range 10))
		    (do0
		     (do0
		      (comments "forward")
		      (do0 (setf emb (aref C X))
			   (setf h
				 (torch.tanh
				  (+ (@ (dot emb
					     (view -1 ,n6)) W1)
				     b1)))
			   (setf logits (+ (@ h W2)
					   b2))
			   (setf loss (F.cross_entropy logits Y))
			   ,(lprint :vars `((loss.item)))))

		     (do0
		      (comments "backward pass")
		      (for (p parameters)
			   (setf p.grad None))
		      (loss.backward)
		      (comments "update")
		      (for (p parameters)
			   (incf p.data
				 (* -.1 p.grad)))
		      )))
	       ))

	     (python
	      (export
	       (comments "learn with minibatch")
	       (for (_ (range 10_000))
		    (do0
		     (do0
		      (setf ix (torch.randint 0 (aref X.shape 0)
					      (tuple 32)))
		      (comments "forward")
		      (setf emb (aref C (aref X ix)))
		      (setf h
			    (torch.tanh
			     (+ (@ (dot emb
					(view -1 ,n6)) W1)
				b1)))
		      (setf logits (+ (@ h W2)
				      b2))
		      (setf loss (F.cross_entropy logits (aref Y ix)))
		      ,(lprint :vars `((loss.item))))

		     (do0
		      (comments "backward pass")
		      (for (p parameters)
			   (setf p.grad None))
		      (loss.backward)
		      (comments "update")
		      (for (p parameters)
			   (incf p.data
				 (* -.1 p.grad)))
		      )))
	       (do0 (comments "report loss on entire data set")
		    (setf emb (aref C X))
		    (setf h
			  (torch.tanh
			   (+ (@ (dot emb
				      (view -1 ,n6)) W1)
			      b1)))
		    (setf logits (+ (@ h W2)
				    b2))
		    (setf loss_full (F.cross_entropy logits Y))
		    ,(lprint :vars `((loss_full.item))))
	       ))

	     (python
	      (do0
	       (comments "find a good learning rate, start with a very small lr and increase exponentially")
	       (setf lre (linspace -3 0 1000)
		     lre (** 10 lre))
	       (setf lri (list)
		     lossi (list))
	       (for (lr (tqdm.tqdm lre))
		    (do0
		     (do0
		      (setf ix (torch.randint 0 (aref X.shape 0)
					      (tuple 32)))
		      (comments "forward")
		      (setf emb (aref C (aref X ix)))
		      (setf h
			    (torch.tanh
			     (+ (@ (dot emb
					(view -1 ,n6)) W1)
				b1)))
		      (setf logits (+ (@ h W2)
				      b2))
		      (setf loss (F.cross_entropy logits (aref Y ix)))
					; ,(lprint :vars `((loss.item)))
		      )

		     (do0
		      (comments "backward pass")
		      (for (p parameters)
			   (setf p.grad None))
		      (loss.backward)
		      (comments "update")
		      (for (p parameters)
			   (incf p.data
				 (* -1 lr p.grad)))

		      (comments "track stats")
		      (lri.append lr)
		      (lossi.append (loss.item))
		      )))

	       ))
	     (python
	      (do0
	       (plot lre lossi
		     :alpha .4)
	       (grid)))
	     ))

       ))))

