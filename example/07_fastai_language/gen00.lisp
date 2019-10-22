(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/07_fastai_language")
  (defparameter *code-file* "run_00_nlp")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# lesson 4: transferlearning nlp"
	    "# export LANG=en_US.utf8"
	    "# https://www.youtube.com/watch?time_continue=60&v=qqt3aMPB81c"
	    "# predict next word of a sentence"
	    "# https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb"
	    
	    (do0
	     (imports (matplotlib))
			      ;(matplotlib.use (string "Agg"))
			      (imports ((plt matplotlib.pyplot)))
			 (plt.ion))
	    
	    #+nil (imports (			;os
					;sys
					;time
					;docopt
					;pathlib
		      ;(np numpy)
		      ;serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      ))


	    ;"from fastai import *"
	    "from fastai.text import *"
	    ;"from fastai.metrics import error_rate"

	    (setf path (untar_data URLs.IMDB ;_SAMPLE
				   )
		  fn (pathlib.Path (string "/home/martin/.fastai/data/imdb/data_lm.pkl")))
	    (setf bs 48)

	    (if (fn.is_file)
		(do0
		 (print (string "load lm data from pkl"))
		 (setf data_lm (load_data path (string "data_lm.pkl" :bs bs))))
		(do0
		 (print (string "load lm data from disk"))
	       #+nil (setf data (TextDataBunch.from_csv :path path :csv_name (string "texts.csv")))
	       (setf data_lm (dot (TextList.from_folder path)
			       (filter_by_folder :include (list (string "train")
								(string "test")
								(string "unsup")))
			       (split_by_rand_pct .1)
			       (label_for_lm)
			       (databunch :bs bs)))
	       (data_lm.save (string "data_lm.pkl"))))

	    
	    
	    (setf learn (language_model_learner data_lm AWD_LSTM :drop_mult .3))

	    #+nil (do0
	     (learn.lr_find)
	     (learn.recorder.plot :skip_end 15))


	    (setf fn_head (pathlib.Path (string "/home/martin/.fastai/data/imdb/models/fit_head.pth")))
	    (if (fn_head.is_file)
		(do0
		 (print (string "load language model"))
		 (learn.load (string "fit_head")))
		(do0
		 (print (string "train language model"))
		 (learn.fit_one_cycle 1 1e-2 :moms (tuple .8 .7)) ;; takes 25min
		 (learn.save (string "fit_head"))))
	    (do0
	     (print (string "unfreeze 1 fine_tuned"))
	     (setf fn_fine (pathlib.Path (string "/home/martin/.fastai/data/imdb/models/fine_tuned.pth")))
	     (if (fn_fine.is_file)
		 (do0
		  (learn.load (string "fine_tuned")))
		 (do0 (learn.unfreeze)
		      (learn.fit_one_cycle 10 1e-3 :moms (tuple .8 .7))
		      (learn.save (string "fine_tuned"))
		      (learn.save_encoder (string "fine_tuned_enc")))))
	    (do0
	     (setf text (string "I liked this movie because")
		   n_words 40
		   n_sentences 2)
	     (setf sentences (list))
	     (for (_ (range n_sentences))
		  (sentences.apppend
		   (learn.predict text n_words :temperature .75)))
	     (print (dot (string "\\n")
			 (join sentences))))
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))


