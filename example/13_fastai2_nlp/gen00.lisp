(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/13_fastai2_nlp")
  (defparameter *code-file* "run_00_nlp")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb"
	    "# export LANG=en_US.utf8"
	    
	    
	    (do0
	     (imports (matplotlib))
			      ;(matplotlib.use (string "Agg"))
			      (imports ((plt matplotlib.pyplot)))
			 (plt.ion))
	    
	    (imports (			;os
					;sys
					;time
					;docopt
					;pathlib
		      ;(np numpy)
					;serial
			    pathlib
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      ))


	    "from fastai2.text.all import *"


	    (setf path (untar_data URLs.IMDB))
	    (comment "=> Path('/home/martin/.fastai/data/imdb')")

	    #+nil (do0
	     (setf files (get_text_files path
					 :folders

					 (list (string "train")
					       (string "test")
					       (string "unsup"))))


	     (setf txts (L (for-generator (o (aref files ":2000"))
					  (dot o
					       (open)
					       (read))))))
	    
	    #+nil
	    (do0
	     (setf spacy (WordTokenizer)
		   tkn (Tokenizer spacy))

	     (do0
	      (setf sp (SubwordTokenizer :vocab_sz 1000))
	      (dot sp
		   (setup txts)))


	     (setf toks200 (dot (aref txts ":200")
				(map tkn)))
	     (do0
	      (setf num (Numericalize))
	      (num.setup toks200)
	      )
	     (do0 (setf nums200 (dot toks200
				    (map num)))
		  (setf dl (LMDataLoader nums200)))
	     
	     (comment "From the book: At every epoch we shuffle our collection of documents and concatenate them into a stream of tokens. We then cut that stream into a batch of fixed-size consecutive mini-streams. Our model will then read the mini-streams in order, and thanks to an inner state, it will produce the same activation whatever sequence length you picked."))

	    (do0
	     (setf get_imdb (partial get_text_files
				     :folders
				     (list (string "train")
					       (string "test")
					       (string "unsup")))
		   dls_lm (dot (DataBlock
				:blocks (TextBlock.from_folder
					 path
					 :is_lm True)
				:get_items get_imdb
				:splitter (RandomSplitter .1))
			       (dataloaders path
					    :path path
					    :bs 64 
					    :seq_len 80))
		   learn (dot (language_model_learner
			       dls_lm
			       AWD_LSTM
			       :drop_mult .3
			       :metrics (list accuracy
					      (Perplexity)))
			      (to_fp16)))
	     (setf
	      problem (string "imdb")
	      fn_1epoch (dot (string "{}_1epoch")
			     (format problem))
	      path_1epoch (pathlib.Path
			   (dot
			    (string "/home/martin/.fastai/data/imdb/models/{}.pth")
			    (format fn_1epoch))))
	     (if (path_1epoch.is_file)
		 (do0
		  (setf learn (learn.load fn_1epoch)))
		 (do0
		  (learn.fit_one_cycle 1 2e-2)
		  (comment "=> 16min45sec, 1min")
		  (comment "epoch     train_loss  valid_loss  accuracy  perplexity  time")
		  (comment "0         4.152357    3.935240    0.297858  51.174419   17:51")
		  (learn.save fn_1epoch))))
	    
	    (do0
	     (learn.unfreeze)
	     (learn.fit_one_cycle 10 2e-3)
	     (learn.save_encoder (dot (string "{}_finetuned")
				      (format problem))))
	    
	    ))) 
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
 

 
  
