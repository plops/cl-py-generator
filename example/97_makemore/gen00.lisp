(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "96_colab_fastai")
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

  (let* ((notebook-name "titanic_from_scratch")
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
		 #+nil(do0

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

		 (imports-from (torch tensor))
		 #+nil  (imports-from (selenium webdriver)
				      (selenium.webdriver.common.keys Keys)
				      (selenium.webdriver.support.ui WebDriverWait)
				      (selenium.webdriver.common.by By)
				      (selenium.webdriver.support expected_conditions)


				      )


		 #+nil
		 (imports-from  (matplotlib.pyplot
				 plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid
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


       (python
	(export1
	 (comments "i want to run this on google colab. annoyingly i can't seem to access the titanic.zip file. it seems to be necessary to supply some kaggle login information in a json file. rather than doing this i downloaded the titanic.zip file into my google drive")
	 (imports (google.colab.drive))
	 (google.colab.drive.mount (string "/content/drive"))

	 ))

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
	 (setf path (pathlib.Path (string "titanic")))
	 (unless (path.exists)
	   (imports (zipfile
		     kaggle))
	   (dot kaggle
		api
		(competition_download_cli (str path)))
	   (dot zipfile
		(ZipFile (fstring "{path}.zip"))
		(extractall path))
	   )))



       (python
	(export
	 (imports (torch
		   (np numpy)
		   (pd pandas)))
	 (setf line_char_width
	       140)
	 (np.set_printoptions :linewidth line_char_width)
	 (torch.set_printoptions :linewidth line_char_width
				 :sci_mode False
				 :edgeitems 7)
	 (pd.set_option (string "display.width")
			line_char_width)))
       (python
	(export
	 (setf df (pd.read_csv (/ path
				  (string "train.csv"))))
	 df)
	)
       (python
	(do0
	 (dot df
	      (isna)
	      (sum))))
       (python
	(export
	 (setf modes (dot df
			  (mode)
			  (aref iloc 0)))))
       (python
	(export
	 (dot df
	      (fillna modes
		      :inplace True))))
       (python
	(do0
	 (dot df
	      (isna)
	      (sum))))

       (python
	(do0
	 (dot df
	      (describe :include (tuple np.number)))))
       (python
	(do0
	 (df.Fare.hist)))
       (python
	(export
	 (setf (aref df (string "LogFare"))
	       (np.log
		(+ 1 df.Fare)))))
       (python
	(do0
	 (comments "histogram of logarithm of prices no longer shows the 'long' tail")
	 (df.LogFare.hist)))
       (python
	(do0
	 (comments "look at the three values that are in passenger class. more details about the dataset are here: https://www.kaggle.com/competitions/titanic/data")
	 (setf pclasses (sorted (df.Pclass.unique)))
	 pclasses))
       (python
	(do0
	 (comments "look at columns with non-numeric values")
	 (df.describe :include (list object))))
       (python
	(export
	 (comments "replace non-numeric values with numbers by introducing new columns (dummies). The dummy columns will be added to the dataframe df and the 3 original columns are dropped."
		   "Cabin, Name and Ticket contain too many unique values for this approach to be useful")
	 (setf df (pd.get_dummies
		   df
		   :columns (list (string "Sex")
				  (string "Pclass")
				  (string "Embarked"))))
	 df.columns))
       (python
	(do0
	 (comments "look at the new dummy columns")
	 (setf added_columns (list ,@(loop for e in
					   `(Sex_male Sex_female
						      Pclass_1 Pclass_2 Pclass_3
						      Embarked_C Embarked_Q Embarked_S
						      )
					   collect
					   `(string ,e))))
	 (dot (aref df added_columns)
	      (head))))
       (python
	(export
	 (comments "create dependent variable as tensor")
	 (setf t_dep (tensor df.Survived))))
       (python
	(export
	 (comment "independent variables are all continuous variables of interest and the newly created columns")
	 (setf indep_columns
	       (+ (list ,@(loop for e in `(Age SibSp Parch LogFare)
				collect
				`(string ,e)))
		  added_columns))
	 (setf t_indep (tensor (dot (aref df indep_columns)
				    values)
			       :dtype torch.float)
	       )
	 t_indep))
       (python
	(do0
	 t_indep.shape))
       (python
	(do0
	 (comments "set up linear model. first we calculate manually a single step for the loss of every row in the dataset. we start with a random coefficient in (-.5,.5) for each column of t_indep")
	 (torch.manual_seed 442)
	 (setf n_coeffs (dot t_indep
			     (aref shape 1)))
	 (setf coeffs (- (dot torch
			      (rand n_coeffs))
			 .5))
	 coeffs))
       (python
	(do0
	 (comments "our predictions are formed by multiplying a row with coefficients and summing them up. we don't need to introduce a bias (or intercept) term by introducing a column containing only ones. Such a 'one' is already present in each row in either the dummy column Sex_male or Sex_female.")
	 (* t_indep coeffs)
	 ))
       (python
	(do0
	 (comments "we have a potential problem with the first column Age. Its values are bigger in average than the values in other columns"
		   "In the lecture Jeremy mentions two options to normalize Age I can think of two more methods: 1) divide by maximum or 2) subtract mean and divide by std 3) subtract median and divide by MAD 4) find lower 2 perscentile and upper 2 percentile increase the value gap by +/- 10% and use this interval to normalize the input values. In the book jeremy uses 1). 1) and 3) differ by how they handle outliers. The maximum will be influenced a lot by outliers. I would like to know if 3) is better than 1) for typical problems. I think that boils down to how big the training dataset is. Once it is big enough there may be always enough outliers to ensure even the maximum is stable.")
	 (when True
	   (do0
	    (comments "method 1)")
	    (setf (ntuple vals indices)
		  (t_indep.max :dim 0))
	    (setf t_indep (/ t_indep
			     vals))))
	 (when False
	   (do0
	    (comments "method 2)")
	    (setf (ntuple means indices1)
		  (t_indep.mean :dim 0))
	    (setf (ntuple stdts indices2)
		  (t_indep.std :dim 0))
	    (setf t_indep (/ (- t_indep
				means)
			     stds))))))

       (python
	(do0
	 (comments "create predictions by adding up the rows of the product")
	 (setf preds  (dot (* t_indep
			      coeffs)
			   (sum :axis 1)))))
       (python
	(do0
	 (comments "look at first few")
	 (aref preds (slice "" 10))
	 (comments "as the coefficents were random these predictions are no good")))
       (python
	(do0
	 (comments "in order to improve the predictions modify the coefficients with gradient descent"
		   "define the loss as the average error between predictions and the dependent")
	 (setf loss (dot torch
			 (abs (- preds
				 t_dep))
			 (mean)))
	 loss))
       (python
	(export
	 (comments "using what we learned in the previous cells create functions to compute predictions and loss")
	 (def calc_preds (&key coeffs indeps)
	   (return
	     (dot (* indeps
		     coeffs)
		  (sum :axis 1))))
	 (def calc_loss (&key coeffs indeps deps)
	   (setf preds (calc_preds :coeffs coeffs
				   :indeps indeps))
	   (setf loss (dot torch
			   (abs (- preds
				   deps))
			   (mean)))
	   (return loss))
	 ))
       (python
	(do0
	 (comments "perform a single 'epoch' of gradient descent manually"
		   "tell pytorch that we want to calculate the gradients for the coeffs object. the underscore indicates that the coeffs object will be modified in place")
	 (dot coeffs
	      (requires_grad_))

	 ))
       (python
	(do0
	 (comments "compute the loss, pytorch will perform book keeping to compute gradients later")

	 (setf loss (calc_loss :coeffs coeffs
			       :indeps t_indep
			       :deps t_dep))))
       (python
	(do0
	 (comments "compute gradient")
	 (loss.backward)

	 coeffs.grad
	 (comments "note that every call of backward() adds the gradients to grad")))
       (python
	(do0
	 (comments "calling the steps a second time will double the values in .grad")
	 (do0 (setf loss (calc_loss :coeffs coeffs
				    :indeps t_indep
				    :deps t_dep))
	      (loss.backward)
	      coeffs.grad)))
       (python
	(do0
	 (comments "we can now perform a single gradient step. the loss should reduce")
	 (do0 (setf loss (calc_loss :coeffs coeffs
				    :indeps t_indep
				    :deps t_dep))
	      (loss.backward)
	      (with (torch.no_grad)
		    (do0
		     (dot coeffs
			  (sub_
			   (* coeffs.grad
			      .1)))
		     (dot coeffs
			  grad
			  (zero_)))
		    (print (calc_loss :coeffs coeffs
				      :indeps t_indep
				      :deps t_dep)))
	      )
	 (comments "a.sub_(b) subtracts the gradient from coeffs in place (a = a - b) and zero_ clears the gradients")))

       (python
	(export
	 (comments "before we can perform training, we have to create a validation dataset"
		   "we do that in the same way as the fastai library does")
	 (imports (fastai.data.transforms))
	 (comments "get training (trn) and validation indices (val)")
	 (setf (ntuple trn
		       val)
	       ((fastai.data.transforms.RandomSplitter
		 :seed 42
		 )
		df))
	 ))
       (python
	(export
	 ,@(loop for e in `(indep dep)
		 collect
		 `(do0
		   ,@(loop for f in `(trn val)
			   collect
			   `(setf ,(format nil "~a_~a" f e)
				  (aref ,(format nil "t_~a" e)
					,f)))))
	 (ntuple (len trn_indep)
		 (len val_indep))))
       (python
	(export
	 (comments "create 3 functions for the operations that were introduced in the previous cells")
	 (def update_coeffs (&key coeffs learning_rate)
	   (do0
	    (dot coeffs
		 (sub_
		  (* coeffs.grad
		     learning_rate)))
	    (dot coeffs
		 grad
		 (zero_))))
	 (def init_coeffs ()
	   (setf coeffs (- (dot torch
				(rand n_coeffs))
			   .5))
	   (coeffs.requires_grad_)
	   (return coeffs))
	 (def one_epoch (&key coeffs learning_rate)
	   (do0 (setf loss (calc_loss :coeffs coeffs
				      :indeps trn_indep
				      :deps trn_dep))
		(loss.backward)
		(with (torch.no_grad)
		      (update_coeffs :coeffs coeffs
				     :learning_rate learning_rate)
		      )

		(print (fstring "{loss:.3f}")
		       :end (string "; "))))))

       (python
	(export
	 (comments "now use these functions to train the model")
	 (def train_model (&key (epochs 30)
				(learning_rate .01))
	   (torch.manual_seed 442)
	   (setf coeffs (init_coeffs)
		 )
	   (for (i (range epochs)
		   )
		(one_epoch :coeffs coeffs
			   :learning_rate learning_rate))
	   (return coeffs))))
       (python
	(do0
	 (comments "try training. the loss should decrease")
	 (setf coeffs (train_model :epochs 18
				   :learning_rate .2))))
       (python
	(do0
	 (def show_coeffs ()
	   (return (dict
		    (zip indep_cols
			 (map
			  (lambda (x)
			    (x.item))
			  (coeffs.requires_grad_ False))))))
	 (show_coeffs)))

       (python
	(do0
	 (comments "the kaggle competition scores accuracy -- the proportion of rows where we correctly predict survival"
		   "determine accuracy using the validation set"
		   "first compute the predictions")
	 (setf preds (calc_preds :coeffs coeffs
				 :indeps val_indep))
	 (comments "for passenger with preds > 0.5 our model predicts survival. compare this with the dependent variable")
	 (setf results (== (val_dep.bool)
			   (> preds 0.5)))
	 (aref results (slice "" 16))
	 ))
       (python
	(do0
	 (comments "compute average accuracy")
	 (dot results
	      (float)
	      (mean))))
       (python
	(export
	 (comments "create a function to compute accuracy")
	 (def acc (coeffs)
	   (setf results (== (val_dep.bool)
			     (> preds 0.5)))
	   (return (dot results
			(float)
			(mean)
			(item))))
	 (print (dot (string "{:3.2f}")
		     (format (acc coeffs))))
	 ))
       (python
	(do0
	 (comments "some predictions are >1 and some are <0. We don't want that")
	 ))
       (python
	(do0
	 (imports (sympy))
	 (sympy.plot (string "1/(1+exp(-x))")
		     :xlim (tuple -7 7))
	 (comments "pytorch contains the sigmoid function")))
       (python
	(export
	 (def calc_preds (coeffs indeps)
	   (return (torch.sigmoid
		    (dot (* indeps
			    coeffs)
			 (sum :axis 1)))))))
       (python
	(export
	 (comments "train a new model now using the updated function to calculate predictions (that will always be in (0,1))")
	 (setf coeffs (train_model :learning_rate 100))
	 
	 ))
       (python
	(do0
	 (acc coeffs)))
       
       (python
	(do0
	 (show_coeffs)
	 (comments "older people and males are less likely to survive. first class passengers are more likely to survive.")))


       (markdown "## Submitting to Kaggle")
       (python
	(do0
	 (comments "read the test set")
	 (setf tst_df (pd.read_csv (/ path
				      (string "test.csv"))))
	 (comments "clean up one missing Fare, set it to 0")
	 (setf (aref tst_df (string "Fare"))
	       (dot tst_df
		    Fare
		    (fillna 0)))))
       (python
	(do0
	 (comments "perform the same steps we did for the training set"
		   )))

       
       

       )))
  (sb-ext:run-program "/usr/bin/ssh"
		      `("c11"
			"cd ~/arch/home/martin/stage/cl-py-generator/example/96_colab_fastai/source; nbdev_export")))



