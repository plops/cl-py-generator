(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_data"
						 "example/161_sqlite_embed/"))
 `(do0
    (imports (;(np numpy)
	      (pd pandas)
	      sys))
    (imports-from (sqlite_minutils *)
		  (loguru logger))

    (do0
     (logger.remove)
     (logger.add
      sys.stdout
      :format (string "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>")
      :colorize True
      :level (string "DEBUG")
      ;:utc True
      ))

    (logger.info (string "Logger configured"))
    (setf db (Database (string "/home/kiel/summaries_20250929.db"))
	  tab (db.table (string "items")))
    (logger.info (string "Sqlite file opened"))
    
    ,(let ((db-cols `(identifier model summary summary_timestamp_start summary_timestamp_end summary_done summary_input_tokens summary_output_tokens host original_source_link embedding full_embedding)))
       `(do0
	 (setf cols (list ,@(loop for e in db-cols
				  collect `(string ,e))))
	 (setf sql (+ (string "SELECT ")
		      (dot (string ", ")
			   (join cols))
		      (string " FROM items WHERE summary is NOT NULL AND summary !=''")))
	 (setf df (pd.read_sql_query sql db.conn))
	 (logger.info (string "Read columns from sqlite into pandas dataframe")))
       )
    
    (comments
     "
 convert a subset of the columns to pandas (identifier model summary summary_timestamp_start summary_timestamp_end summary_done summary_input_tokens summary_output_tokens host original_source_link embedding full_embedding)
 filter out bad summaries
 compute 4d and 2d umap
 compute hdbscan clusters
 find names for the clusters

 store the umap so that new emedding entries can be assigned a 2d position and a cluster name

 visualize the full map

 display a segment of the full map in the neighborhood of a new embedding entry


")))


