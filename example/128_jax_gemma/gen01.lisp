(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "128_jax_gemma")
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
  
  (let* ((notebook-name "gemma")
	 (cli-args `(#+nil (:short "c" :long "chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     #+nil (:short "p" :long "prompt" :type str
		      :default (string "Summarize the following video transcript as a bullet list.")
		      :help "The prompt to be prepended to the output file(s)."))))
    (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "https://youtu.be/1RcORri2ZJg?t=418")
       (comments "!pip install -q git+https://github.com/google-deepmind/gemma.git ")
       (imports (os
		 time
		 kagglehub
		 gemma
		 gemma.params
		 gemma.transformer
		 gemma.sampler
		 (spm sentencepiece)))
       (imports-from (google.colab userdata))
       ;"from gemma import params as params_lib"
       
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


       #+Nil
       ,@(loop for e in `(KAGGLE_USERNAME
			  KAGGLE_KEY)
	       collect
	       `(setf (aref os.environ (string ,e))
		      (userdata.get (string ,e))))
       
       (kagglehub.login)

       (comments "Enable GPU in Colab: Click on Edit > Notebook settings > Select T4 GPU")

       

       (comments "gemma-2b-it is 3.7Gb in size")
       (setf GEMMA_VARIANT (string "2b-it"))
       (setf GEMMA_PATH (kagglehub.model_download (fstring "google/gemma/flax/{GEMMA_VARIANT}")))
       (comments "in addition to logging in with an api key into kaggle, i also had to manually submit a consent form on the kaggle website before i was able to download the gemma data")
       ,(lprint :vars `(GEMMA_PATH))

       (comments "specify tokenizer model file and checkpoint")

       (setf CKPT_PATH (os.path.join GEMMA_PATH GEMMA_VARIANT)
	     TOKENIZER_PATH (os.path.join GEMMA_PATH (string "tokenizer.model")))
       ,(lprint :vars `(CKPT_PATH ))
       ,(lprint :vars `(TOKENIZER_PATH))


       (setf params (gemma.params.load_and_format_params CKPT_PATH))

       (comments "load tokenizer")
       (setf vocab (spm.SentencePieceProcessor))
       (vocab.Load TOKENIZER_PATH)

       (setf transformer_config (gemma.transformer.TransformerConfig.from_params
				 :params params
				 :cache_size 1024))
       (setf transformer (gemma.transformer.Transformer transformer_config))

       (comments "create sampler")

       (setf sampler (gemma.sampler.Sampler
		      :transformer transformer
		      :vocab vocab
		      :params (aref params (string "transformer"))))

       (comments "write prompt in input_batch and perform inference. total_generation_steps is limited to 100 here to preserve host memory")

       (setf prompt (list (string "\\n# What is the meaning of life?")))
       (setf reply (sampler :input_strings prompt
			    :total_generation_steps 100))
       ,(lprint :vars `(reply.text))
       )

     

     
     )))

