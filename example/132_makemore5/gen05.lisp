(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "132_makemore5")
  (defparameter *idx* "05")
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
  (let* ((notebook-name "makemore5_in_jax")
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
		 (imports (os
			   time
			   pathlib
			   ;random
			   tqdm
			   argparse
			   jax
			   (jnp jax.numpy)
			   (np numpy)
			   optax
			   ))
		 
		 "from flax import linen as nn"
		 ;(imports-from (jax.nn softmax))
		 (imports-from (jax random))
		 (imports-from (flax.linen softmax))
		 (imports-from (flax.training train_state))



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
	 (comments "This code trains a network with 20k parameters that generates character sequences that look like names.")
	 (comments "Based on the youtube video https://youtu.be/t3YJ5hKiMQ0 31:40 that explains this notebook: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb"
		   "Jax code based on https://www.kaggle.com/code/shaochuanwang/makemore-jax-transformer")
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
	 (setf train_config
	       (dictionary :vocab_size vocab_size
			   :block_size 8
			   :n_embed 10
			   :n_hidden0 68
			   :n_hidden1 68
			   :n_hidden2 68
			   :learning_rate .1s0
			   :decay_rate 1s-2
			   :decay_step 150_000
			   :dtype jnp.bfloat16))))

       
       
       (python
	(export
	 (comments "build the dataset")
	 (comments "block_size .. context length of how many characters do we take to predict the next one")
	 

	 (setf block_size (aref train_config (string "block_size")))
	 
	 (def build_dataset (words)
	   (string3 "This function builds a dataset for training a model using the given list of words.
    It creates a context of a certain block size for each character in the words and uses this to predict the next character.

    Args:
        words (list): A list of words to be used for creating the dataset.

    Returns:
        tuple: A tuple containing the input tensor (jax array) (X) and the target tensor (Y). X is the tensor representing the context for each character, and Y is the tensor representing the characters themselves.")
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
	   
	   (setf X (jnp.array X)
		 Y (jnp.array Y))
	   (comments "Each element in Y is the character that should be predicted given the corresponding context in X.")
	   ,(lprint :vars `(X.shape Y.shape))
	   (return (ntuple X Y)))

	 (setf key (jax.random.PRNGKey 42))
	 (setf (ntuple key subkey)
	       (random.split key))
	 (setf rand_ixs (random.permutation subkey
					     (jnp.arange (len words))
					     :independent True))
	 
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
		       (setf (ntuple ,x ,y) (build_dataset (aref (np.array words) (aref rand_ixs ,slice))))))))
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
				       (list self.bias))))))
		,(make-to `(weight bias))
		)

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
		  (comments "After initialization the batchnorm gives each neuron a normal distributed activation (mean 0, std 1). These two parameters allow the optimizer to move the activation to whatever is best for our examples (see https://youtu.be/P6sfmUTpUmc?t=2830 ).")
		  (comments "gamma scalaes xhat and beta is an offset")
		  (setf self.gamma (torch.ones dim)
			self.beta (torch.zeros dim))
		  (comments "Buffers (updated with a running 'momentum update')")
		  (setf self.running_mean (torch.zeros dim)
			self.running_var (torch.ones dim)))
		,(make-to `(gamma beta running_mean running_var))
		(def __call__ (self x)
		  (string3 "Forward pass through the layer.

 If the model is in training mode, the mean and variance are computed
 from 'x'.  The dimension(s) along which these statistics are computed
 depend on the number of dimensions of 'x': if 'x' is 2D, the
 statistics are computed along the 0th dimension; if 'x' is 3D, they
 are computed along the 0th and 1st dimensions.


        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.")
		  (if self.training
		      (do0
		       (cond
			 ((== 2 x.ndim)
			  (setf dim 0))
			 ((== 3 x.ndim)
			  (setf dim (tuple 0 1))))
		       (setf xmean (x.mean dim :keepdim True)
			     xvar (x.var dim :keepdim True)))
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
		  (return (list))))


	 (class Embedding ()
		(string3
		 "A class representing an embedding layer in a neural network.
    This layer transforms integer indices to dense vectors of fixed size.
    
    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.") 
		(def __init__ (self num_embeddings embedding_dim)
		  (comments "Initialize the embedding weights with random values")
		  (setf self.weight (torch.randn (tuple num_embeddings embedding_dim)) ))
		,(make-to `(weight))
		(def __call__ (self IX)
		  (string3 "Perform the forward pass of the embedding layer.
        
        Args:
            IX (Tensor): A tensor containing the indices to be looked up.
        
        Returns:
            Tensor: The corresponding embedding vectors.")
		  (setf self.out (aref self.weight IX))
		  (return self.out))
		(def parameters (self)
		  (string3 "Get the parameters of the embedding layer.
        
        Returns:
            list: A list containing the embedding weights.")
		  (return (list self.weight))))

	 (class FlattenConsecutive ()
		(string3
		 "A class representing the flattening operation in a neural network.
    This operation reshapes the input tensor to a 2D tensor with shape (batch_size, -1).")
		(def __init__ (self n)
		  
		  (setf self.n n))
		
		(def __call__ (self x)
		  (string3 "Apply the flattening operation to the input tensor.
        
        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: The flattened tensor.")
		  (comments "B .. batch size, TT .. number of characters, C .. embedding of a single character")
		  (setf (ntuple B TT C) x.shape)
		  (setf x (x.view B (// TT self.n) (* C self.n)))
		  (when (== 1 (aref x.shape 1))
		    (comments "Squeeze out first dimension if it is one")
		    (setf x (x.squeeze 1)))
		  (setf self.out x)
		  (return self.out))
		(def parameters (self)
		  (string3 "As the flattening operation does not have any parameters, this method returns an empty list.")
		  (return (list))))

	 (comments "Sequential is explained here: https://youtu.be/t3YJ5hKiMQ0?t=815")
	 (class Sequential ()
		(string3
		 "A class representing a sequential container in a neural network.
    Modules will be added to it in the order they are passed in the constructor.
    Essentially, it allows us to specify a sequence of transformations that our input data will go through.

    Args:
        layers (list): A list of layers that make up the sequence.")
		(def __init__ (self layers)
		  (comments "Initialize the sequence with the provided layers")
		  (setf self.layers layers))
		

		(def __call__ (self x)
		  (string3 "Apply the sequence of transformations to the input data.
        
        Args:
            x (Tensor): The input data.
        
        Returns:
            Tensor: The output of the sequence of transformations.")
		  (for (layer self.layers)
		       (setf x (layer x)))
		  (setf self.out x)
		  (return self.out))
		(def parameters (self)
		  (string3 "Get the parameters of all the layers in the sequence.
        
        Returns:
            list: A list containing the parameters of all the layers in the sequence.
        ")
		  (return (list (for-generator (p (layer.parameters))
					       (for-generator
						(layer self.layers)
						p)))))
		(def to (self device)
		  (string3 "Move the parameters of all the layers in the sequence to the specified device.

        Args:
            device (torch.device): The device to move the parameters to.
        ")
		  (for (layer self.layers)
		       (when (hasattr layer (string "to"))
			 (layer.to device))
		       #+nil (for (param (layer.parameters))
				  (setf param.data
					(param.data.to device))))))))

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
	 (setf model (Sequential
		      (list
		       (Embedding vocab_size n_embed)

		       (FlattenConsecutive 2)
		       (Linear (* n_embed 2)
			       n_hidden0
			       :bias False)
		       (BatchNorm1d n_hidden0) 

		       (FlattenConsecutive 2)
		       (Linear (* n_hidden0 2)
			       n_hidden1
			       :bias False)
		       (BatchNorm1d n_hidden1)

		       (FlattenConsecutive 2)
		       (Linear (* n_hidden1 2)
			       n_hidden2
			       :bias False)
		       (BatchNorm1d n_hidden2)

		       (Tanh)
		       (Linear n_hidden2 vocab_size))))
	 
	 (model.to device)

	 (comments "Make the last layer less confident. This is done by scaling down the
weights of the last layer. This can help for the network to be initially overconfidently wrong.")
	 (with (torch.no_grad)
	       (setf (dot model
			  (aref layers -1)
			  weight)
		     (* .1s0 (dot model
				  (aref layers -1)
				  weight))))
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
	 (comments "Maximum number of training steps")
	 (setf max_steps 200_000
	       )
	 (comments "Size of the minibatches")
	 (setf batch_size 256)
	 (comments "List to store the loss values")
	 (setf lossi (list))
	 (comments "A4000 GPU: 937 it/s 3:18"
		   "TR7955 CPU (16 cores): 403 it/s 8:16 ")
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

 
 
