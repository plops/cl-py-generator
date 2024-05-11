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
	(export
	 (comments "Based on a youtube video that explains this notebook: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb")
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
		     (setf ix (aref stoi ch))
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
	 (class Linear ()
		(string3 "A class representing a linear layer in a neural network. It computes a
matrix multiplication in the forward pass.

    Args:
        fan_in (int): The number of input features.
        fan_out (int): The number of output features.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
")
		(def __init__ (self fan_in fan_out &key (bias True))
		  (string3 " Initialize the linear layer with weights and bias.

        The weights are initialized using Kaiming initialization,
        which is a method of initializing neural networks to help
        ensure the signal from the input data does not vanish or
        explode as it is propagated through the network.

        Args:
            fan_in (int): The number of input features.
            fan_out (int): The number of output features.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        ")
		  (comments "note: Kaiming init")
		  (setf self.weight (/ (torch.randn (tuple fan_in
							   fan_out))
				       (** fan_in .5)))
		  (setf self.bias (? bias
				     (torch.zeros fan_out)
				     None)))
		(def __call__ (self x)
		  (string3 "Forward pass through the layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.")
		  (setf self.out (@ x self.weight))
		  (unless (is self.bias None)
		    (incf self.out self.bias))
		  (return self.out))
		(def parameters (self)
		  (string3 "Get the parameters of the layer.

        Returns:
            list: A list containing the weight tensor and, if it exists, the bias tensor.
        ")
		  (return (+ (list self.weight)
			     (paren (? (is self.bias None)
				       (list)
				       (list self.bias)))))))

	 (class BatchNorm1d ()
		(string3 "    A class representing a 1-dimensional batch normalization layer.

    Batch normalization is a technique for improving the speed,
    performance, and stability of neural networks.  It normalizes the
    input features across the mini-batch dimension, i.e., for each
    feature, it subtracts the mean and divides by the standard
    deviation, where both statistics are computed over the mini-batch.
    

    Note: The BatchNorm1d layer has different behaviors during
    training and inference.  It's crucial to set the correct
    mode (training or inference) to avoid unexpected results or bugs.
    There is state in this layer and state is (usually) harmful.

    Note: In BatchNorm1d, the batch dimension serves a specific
    purpose beyond efficiency.  It couples computations across batch
    elements to control activation statistics, which is integral to
    its functionality. 

    Args:
        dim (int): The number of features in the input.
        eps (float, optional): A small number added to the denominator for numerical stability. Defaults to 1e-5.
        momentum (float, optional): The momentum factor for the running mean and variance computation. Defaults to 0.1")
		(def __init__ (self dim &key (eps 1s-5) (momentum .1))
		  (string3 " Initialize the batch normalization layer with parameters and buffers.

        Args:
            dim (int): The number of features in the input.
            eps (float, optional): A small number added to the denominator for numerical stability. Defaults to 1e-5.
            momentum (float, optional): The momentum factor for the running mean and variance computation. Defaults to 0.1.
        ")
		  (setf self.eps eps
			self.momentum momentum
			self.training True)
		  (comments "Parameters (trained with backpropagation)")
		  (setf self.gamma (torch.ones dim)
			self.beta (torch.zeros dim))
		  (comments "Buffers (updated with a running 'momentum update')")
		  (setf self.running_mean (torch.zeros dim)
			self.running_var (torch.ones dim)))
		(def __call__ (self x)
		  (string3 "Forward pass through the layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.")
		  (if self.training
		      (setf xmean (x.mean 0 :keepdim True)
			    xvar (x.var 0 :keepdim True))
		      (setf xmean self.running_mean
			    xvar self.running_var)
		      )
		  (comments "Normalize to unit variance")
		  (setf xhat (/ (- x xmean)
				(torch.sqrt (+ xvar self.eps))))
		  
		  (setf self.out (+ (* self.gamma
				       xhat)
				    self.beta))
		  (comments "Update the buffers")
		  (when self.training
		    (with (torch.no_grad)
			  (setf self.running_mean
				(+ (* (- 1 self.momentum)
				      self.running_mean)
				   (* self.momentum xmean)))
			  (setf self.running_var
				(+ (* (- 1 self.momentum)
				      self.running_var)
				   (* self.momentum xvar)))))
		  (return self.out))
		(def parameters (self)
		  (string3 "Get the parameters of the layer.
Returns:
            list: A list containing the gamma and beta tensors.")
		  (return (list self.gamma self.beta))))

	 (class Tanh ()
		(string3
		 "A class representing the hyperbolic tangent activation function.

    The hyperbolic tangent function, or tanh, is a function that squashes its input into the range between -1 and 1.
    It is commonly used as an activation function in neural networks.")
		(def __call__ (self x)
		  (string3 "Apply the tanh function to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor, where the tanh function has been applied element-wise.
        ")
		  (setf self.out (torch.tanh x))
		  (return self.out))
		(def parameters (self)
		  (string3 "Get the parameters of the layer.

        The tanh function does not have any parameters, so this method returns an empty list.

        Returns:
            list: An empty list.")
		  (return (list))))))

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
	 (setf n_hidden 200)

	 (comments "Create the embedding table C:"
		   "C is a matrix where each row represents a character in the vocabulary,"
		   "and each column represents a dimension in the embedding space.")
	 (setf C (torch.randn (tuple vocab_size n_embed)))
	 (comments "Define the list of layers
The MLP consists of a linear layer, a batch normalization layer, a
tanh activation function, and another linear layer. The output of the
MLP is a probability distribution over the vocabulary.")
	 (setf layers
	       (list (Linear (* n_embed block_size)
			     n_hidden
			     :bias False)
		     (BatchNorm1d n_hidden)
		     (Tanh)
		     (Linear n_hidden vocab_size)))

	 (comments "Make the last layer less confident. This is done by scaling down the
weights of the last layer. This can help for the network to be initially overconfidently wrong.")
	 (with (torch.no_grad)
	       (setf (dot (aref layers -1)
			  weight)
		     (* .1s0 (dot (aref layers -1)
				  weight))))
	 (comments "Gather all the parameters of the model: This includes the embedding
table C and the parameters of all the layers in the MLP.")
	 (setf parameters
	       (+ (list C)
		  (list (for-generator (p (layer.parameters))
			 (for-generator (layer layers)
					p)))))

	 ,(lprint :msg "Number of parameters in total"
		  :vars `((sum (for-generator (p parameters)
					      (p.nelement)))))
	 (for (p parameters)
	      (setf p.requires_grad True))))
       (python
	(export
	 (comments "Maximum number of training steps")
	 (setf max_steps 200_000
	       )
	 (comments "Size of the minibatches")
	 (setf batch_size 32)
	 (comments "List to store the loss values")
	 (setf lossi (list))
	 (comments "Start the training loop")
	 (for (i (range max_steps))
	      (comments "Construct a minibatch. Xb holds input data, Yb the corresponding target data")
	      (setf ix (torch.randint 0
				      (aref Xtr.shape 0)
				      (tuple batch_size))
		    Xb (aref Xtr ix)
		    Yb (aref Ytr ix))
	      (comments "Forward pass")
	      (comments "Embed the input data into vectors")
	      (setf emb (aref C Xb))
	      (comments "Reshape the embedded data")
	      (setf x (emb.view (aref emb.shape 0)
				-1))
	      (comments "Pass the data through each layer")
	      (for (layer layers)
		   (setf x (layer x)))
	      (comments "Compute the loss (cross entropy)")
	      (setf loss (F.cross_entropy x Yb))

	      (comments "Backward pass")
	      (for (p parameters)
		   (comments "Clear the gradient for each parameter")
		   (setf p.grad None))
	      (comments "Compute the gradient of the loss with respect to the parameters")
	      (loss.backward)

	      (comments "Update the parameters using simple SGD with step learning rate decay")
	      (setf lr (? (< i 150_000)
			  .1s0
			  .01s0))

	      (for (p parameters)
		   (comments "Update the parameter using its gradient")
		   (incf p.data (* -lr p.grad)))
	      
	      (comments "Track the progress (every 10k steps)")
	      (when (== 0 (% i  10_000))
		(setf progress (/ i max_steps))
		,(lprint :vars `(progress (loss.item))))
	      (comments "Append the logarithm of the loss to the list")
	      (lossi.append (dot loss (log10) (item)))
	      )))))))

