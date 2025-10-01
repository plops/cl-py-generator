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
   (do0
    (imports (matplotlib))
    (matplotlib.use (string "webagg"))
    (imports ((plt matplotlib.pyplot)))
    )

   (imports (				;(np numpy)
	     (pd pandas)
	     sys
	     (plt matplotlib.pyplot)))
    
    
    
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
	 (setf (aref df (string "ts_start"))
	       (pd.to_datetime (aref df (string "summary_timestamp_start"))
			       :errors (string "coerce")))
	 (setf (aref df (string "ts_end"))
	       (pd.to_datetime (aref df (string "summary_timestamp_end"))
			       :errors (string "coerce")))
	 (setf (aref df (string "duration_s"))
	       (dot (- (aref df (string "ts_end"))
		       (aref df (string "ts_start")))
		    dt
		    (total_seconds)))
	 (logger.info (string "Read columns from sqlite into pandas dataframe")))
       )

   (do0
    (plt.hist (/ df.summary_input_tokens
		 df.duration_s)
	      :log True
	      :bins 300
	      )
    (plt.show))
   (comments "
>>> df
      identifier                                              model  ... embedding full_embedding
0              1                            gemini-1.5-pro-exp-0827  ...      None           None
1              2                            gemini-1.5-pro-exp-0827  ...      None           None
2              3                            gemini-1.5-pro-exp-0827  ...      None           None
3              4                            gemini-1.5-pro-exp-0827  ...      None           None
4              5                            gemini-1.5-pro-exp-0827  ...      None           None
...          ...                                                ...  ...       ...            ...
7972        8155  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
7973        8156  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
7974        8157  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
7975        8158  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
7976        8159  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None

[7977 rows x 12 columns]
>>> df.iloc[-1]
identifier                                                              8159
model                      gemini-2.5-pro| input-price: 1.25 output-price...
summary                    **Abstract:**\n\nThis personal essay by Georgi...
summary_timestamp_start                           2025-09-29T21:32:46.405084
summary_timestamp_end                             2025-09-29T21:33:08.668613
summary_done                                                             1.0
summary_input_tokens                                                 14343.0
summary_output_tokens                                                  742.0
host                                                          194.230.161.72
original_source_link       https://www.huffpost.com/entry/weight-loss-sur...
embedding                                                               None
full_embedding                                                          None
Name: 7976, dtype: object

")
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


