(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "132_makemore5")
  (defparameter *idx* "04")
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

  (defun make-to (storage)
		  `(def to (self device)
		     ,@(loop for e in storage
			     collect
			     `(unless (is (dot self ,e)
					  None)
				(setf (dot self ,e data)
				      (dot self ,e data (to device)))))))
  (let* ((notebook-name "makemore5_torch")
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
				(optim torch.optim)
				(nn torch.nn)
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
	(export 
	 (comments "This code trains a network with 20k parameters that generates character sequences that look like names.")
	 (comments "Based on the youtube video https://youtu.be/t3YJ5hKiMQ0 31:40 that explains this notebook: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb")
	 )
	)
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
	 (setf vocab_size (len itos))
	 ,(lprint :msg "mapping from integer to character" :vars `(itos vocab_size)))
	)
       (python
	(export
	 (comments "shuffle up the words")
	 (random.seed 42)
	 (random.shuffle words)))

       (python
	(export
	 (comments "build the dataset")
	 (comments "block_size .. context length of how many characters do we take to predict the next one")
	 (do0 (setf device (torch.device #+nil(string "cpu")
					 #-nil(? (torch.cuda.is_available)
					    (string "cuda")
					    (string "cpu"))))
	      (print device))

	 (setf block_size 8)
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
		     (setf ix (aref stoi ch))
		     (comments "The current context is appended to X, and the integer index ix is appended to Y.

The context is updated by removing the first element and appending the integer index ix at the end. This means that the context always contains the integer indices of the last block_size characters.")
		     (X.append context)
		     (Y.append ix)
		     (setf context (+ (aref context (slice 1 ""))
				      (list ix)))))
	   
	   (setf X (dot (torch.tensor X) (to ;:dtype torch.uint8
					     :device device))
		 Y (dot (torch.tensor Y) (to ;:dtype torch.uint8
					     :device device))
		 )
	   (comments "Each element in Y is the character that should be predicted given the corresponding context in X.")
	   ,(lprint :vars `(X.shape Y.shape))
	   (return (ntuple X Y)))
	 (comments "Use 80% for training, 10% for validation and 10% for testing. We use the following indices to perform the split.")
	 (setf n1 (int (* .8 (len words)))
	       n2 (int (* .9 (len words))))
	 ,@(loop for e in `((:name tr :slice (slice "" n1) :comment "Training 80%")
			    (:name dev :slice (slice n1  n2) :comment "Validation 10%")
			    (:name te :slice (slice n2 "") :comment "Test 10%"))
		 collect
		 (destructuring-bind (&key name slice comment) e
		   (let ((x (format nil "X~a" name))
			 (y (format nil "Y~a" name)))
		     `(do0
		       (comments ,comment)
		       (setf (ntuple ,x ,y) (build_dataset (aref words ,slice)))))))
	 ))

       (python
	(do0
	 (for ((ntuple x y)
	       (zip (aref Xtr (slice "" 20))
		    (aref Ytr (slice "" 20))))
	      (print (dot (string "")
			  (join (paren (for-generator (ix x)
						      (aref itos (ix.item))))
				))
		     (string "-->")
		     (aref itos (y.item))))
	 ))

       (python
	(export
	 (class FlattenConsecutive (nn.Module)
		(def __init__ (self n)
		  (dot (super FlattenConsecutive
			      self)
		       (__init__))
		  (setf self.n n))
		(def forward (self x)
		  (setf (ntuple B TT C) x.shape)
		  (setf x (x.view B
				  (// TT self.n)
				  (* C self.n)))
		  (when (== (aref x.shape 1) 1)
		    (setf x (x.squeeze 1)))
		  (return x)))
	 ))

       (python
	(export
	 (comments "Seed rng for reproducibility")
	 (torch.manual_seed 42)
	 ))
       (python
	(export
	 (comments "The dimensionality of the character embedding vectors.")
	 (setf n_embed 10)
	 (comments "The number of neurons in the hidden layer of the MLP")
	 (setf n_hidden0 (* 1
			    68))
	 (setf n_hidden1 (* 2 68))
	 (setf n_hidden2 (* 4 68))
	 	 ;; 4 2 1 2.075
	 ;; 1 2 4 2.05
	 (comments "Define the list of layers
The MLP consists of a linear layer, a batch normalization layer, a
tanh activation function, and another linear layer. The output of the
MLP is a probability distribution over the vocabulary.")
	 (setf model (nn.Sequential
		      (nn.Embedding vocab_size n_embed)

		      (FlattenConsecutive 2)
		      (nn.Linear (* n_embed 2)
				 n_hidden0
				 :bias False)
		      (nn.BatchNorm1d n_hidden0) 

		      (FlattenConsecutive 2)
		      (nn.Linear (* n_hidden0 2)
				 n_hidden1
				 :bias False)
		      (nn.BatchNorm1d n_hidden1)

		      (FlattenConsecutive 2)
		      (nn.Linear (* n_hidden1 2)
				 n_hidden2
				 :bias False)
		      (nn.BatchNorm1d n_hidden2)

		      (nn.Tanh)
		      (nn.Linear n_hidden2 vocab_size)))
	 
	 (setf model (model.to device))

	 (comments "Make the last layer less confident. This is done by scaling down the
weights of the last layer. This can help for the network to be initially overconfidently wrong.")
	 (with (torch.no_grad)
	       ;; i must use the *= operator here, setf is not working
	       (space (dot (aref  model -1)
			   weight)
		      *= .1s0
		     ))
	 (comments "Gather all the parameters of the model: This includes the embedding
table C and the parameters of all the layers in the MLP.")
	 (setf parameters (model.parameters))

	 ,(lprint :msg "Number of parameters in total"
		  :vars `((sum (for-generator (p parameters)
					      (p.nelement)))))
	 (for (p parameters)
	      (setf p.requires_grad True))))

       (python
	(do0
	 (comments "Look at a batch of 4 examples")
	 (setf ix (torch.randint 0 (aref Xtr.shape 0) (tuple 4))
	       Xb (aref Xtr ix)
	       Yb (aref Ytr ix)
	       logits (model Xb))
	 (comments "Print the shape of the input batch")
	 (print Xb.shape)
	 (comments "Print the input batch")
	 Xb
	 ))
       (python
	(do0
	 (comments "Print overview of the architecture and output dimensions at each stage")
	 (for (layer model.layers)
	      (print (dot layer __class__ __name__)
		     (string ":")
		     (tuple layer.out.shape)))))
       (python
	(do0
	 (comments "The batchnorm result should have only one non-unit dimension:")
	 (dot model
	      (aref layers 3)
	      running_mean
	      shape)))
       (python
	(export
	 (comments "Register forward hook for each module")
	 (setf self.hooks "{}")
	 (for ((ntuple name module)
	       (model.named_modules))
	      (setf (aref hooks name)
		    (module.register_forward_hook self hook_fn)))))
       (python
	(export
	 (comments "Maximum number of training steps")
	 (setf max_steps 200_000
	       )
	 (comments "Size of the minibatches")
	 (setf batch_size 256)
	 (comments "List to store the loss values")
	 (setf lossi (list))
	 (setf lr .1)

	 (setf optimizer (optim.SGD (model.parameters)
				    :lr lr))
	 (comments "Start the training loop")
	 (for (i (tqdm.tqdm (range max_steps)))
	      (comments "Construct a minibatch. Xb holds input data, Yb the corresponding target data")
	      (setf ix (torch.randint 0
				      (aref Xtr.shape 0)
				      (tuple batch_size))
		    Xb (dot (aref Xtr ix) (to device))
		    Yb (dot (aref Ytr ix) (to device)))
	      (comments "Forward pass")
	      (comments "Pass the data through each layer")
	      (setf logits (model Xb))
	      (comments "Compute the loss (cross entropy)")
	      (setf loss (F.cross_entropy logits Yb))

	      (comments "Backward pass")
	      #+nil (for (p parameters)
		   (comments "Clear the gradient for each parameter")
		   (setf p.grad None))
	      (optimizer.zero_grad)
	      (comments "Compute the gradient of the loss with respect to the parameters")
	      (loss.backward)
	      

	      (comments "Update the parameters using simple SGD with step learning rate decay")
	      (optimizer.step)
	      #+nil (setf lr (? (< i 150_000)
				.1s0
				.01s0))
	      (when (<= 150_000 i)
		(for (p optimizer.param_groups)
		     (setf (aref p (string "lr")
				 )
			   1s-2)))
	      #+nil
	      (for (p parameters)
		   (comments "Update the parameter using its gradient")
		   (incf p.data (* -lr p.grad)))
	      
	      (comments "Track the progress (every 10k steps)")
	      (when (== 0 (% i  10_000))
		(setf progress (/ i max_steps))
		,(lprint :vars `(progress (loss.item))))
	      (comments "Append the logarithm of the loss to the list")
	      (lossi.append (dot loss (log10) (item)))
	      
	      )))
       

       (python
	(export
	 (comments "average 1000 values into one")
	 (plt.plot (dot torch (tensor lossi)
			(view -1 1000)
			(mean 1)))))

       (python
	(export
	 (comments "Put layers into eval mode (needed for batchnorm especially)")
	 (for (layer model.layers)
	      ;; FIXME: naughty direct access of class attributes
	      (setf layer.training False))))
       (python
	(export
	 (comments "Evaluate the loss for a given data split (train, validation, or
test). This function is decorated with @torch.no_grad() to disable
gradient tracking in PyTorch, as we only want to evaluate the loss and
not perform any updates.
")
	 (@torch.no_grad)
	 (def split_loss (split)
	   (comments "Select the appropriate data based on the provided split")
	   (setf (ntuple x y)
		 (aref (dictionary :train (tuple Xtr Ytr)
				   :val (tuple Xdev Ydev)
				   :test (tuple Xte Yte))
		       split))
	   (setf logits (model x))
	   (comments "Compute cross-entropy loss between model's output and the target data")
	   (setf loss (F.cross_entropy logits y))
	   (comments "Print the loss for the current split")
	   ,(lprint :vars `(split (loss.item))))
	 (comments "Evaluate and print the loss for the training and validation splits. If
training loss is much smaller than validation loss then we overfitted
the network for the task. In this case performance could be improved
by scaling up the network.")
	 (split_loss (string "train"))
	 (split_loss (string "val"))))

       (python
	(export
	 (comments "Generate 20 words using the trained model (sample from model)")
	 (for (_ (range 20))
	      (comments "List to store the generated characters")
	      (setf out (list)
		    )
	      (comments "Initialize context with all end character '.' represented by 0")
	      (setf context (* (list 0)
			       block_size))
	      (while True
		     (comments "Forward pass through the the neural net")

		     (setf logits (model (torch.tensor (list context))))
		     (comments "Compute the softmax probabilities from the output logits")
		     (setf probs (F.softmax logits :dim 1))
		     (comments "Sample the character from the softmax distribution")
		     (setf ix (dot torch (multinomial probs
						      :num_samples 1)
				   (item))
			   )
		     (comments "Update the context by removing the first character and appending the sampled character")
		     (setf context (+ (aref context (slice 1 ""))
				      (list ix)))
		     (comments "Add the sampled character to the output list")
		     (out.append ix)
		     (comments "Break the loop if we sample the special '.' token represented by 0")
		     (when (== ix 0)
		       break))
	      (comments "Decode and print the generated word")
	      (print (dot (string "")
			  (join (for-generator (i out)
					       (aref itos i))))))))
       (python
	(export))))))

 
 
