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
    (matplotlib.use (string "qtagg" ; "webagg"
			    ))
    (imports ((plt matplotlib.pyplot)))
    (plt.ion)
    )

   (imports ((np numpy)
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
	 (do0
	  (comments "Convert Timestamps from string into datetime type")
	  (setf (aref df (string "ts_start"))
		(pd.to_datetime (aref df (string "summary_timestamp_start"))
				:errors (string "coerce")
				:utc True))
	  (setf (aref df (string "ts_end"))
		(pd.to_datetime (aref df (string "summary_timestamp_end"))
				:errors (string "coerce")
				:utc True))
	  (setf (aref df (string "duration_s"))
		(dot (- (aref df (string "ts_end"))
			(aref df (string "ts_start")))
		     dt
		     (total_seconds))))
	 (do0
	  (comments "Find which inferences were performed during the work week and "))
	 (logger.info (string "Read columns from sqlite into pandas dataframe")))
       )
   #+nil
   (do0
    (comments "Add continental US workhour filter")
    (comments "define continental US timezones")
    (setf tzs (dictionary :eastern (string "US/Eastern")
			  :central (string "US/Central")
			  :mountain (string "US/Mountain")
			  :pacific (string "US/Pacific")))
    (comments "create localized columns and boolean workhour masks per timezone")
    (setf work_masks (list))
    (for ((ntuple name tz)
	  (tzs.items))
	 (setf col (fstring "ts_{name}")
	       (aref df col) (dot (aref df (string "ts_start")
					)
				  dt (tz_convert tz)))
	 (comments "workday Mon-Fri -> dayofweek 0..4")
	 (setf is_weekday (< (dot (aref df col)
				dt dayofweek )
			     5))
	 (comments "workhours 09:00 <= local_time < 17:00 (hours 9..16")
	 (setf is_workhour (dot (aref df col)
				dt hour (between 9 16)))
	 (dot work_masks
	      (append (& is_weekday is_workhour))))
    (setf (aref df (string "is_workhours_continental")
		)
	  (dot np logical_or (reduce work_masks)))
    (comments "filter invalid durations and tokens and keep only workhours rows")
    (setf df_valid
	  (aref df
		(& (aref df (string "is_workhours_continental"))
		   (dot (aref df (string "duration_s")) (notna))
		   (> (aref df (string "duration_s")) 0)
		   (dot (aref df (string "summary_input_tokens")) (notna)))
		))
    (setf df_valid_off
	  (aref df
		(& (aref ~df (string "is_workhours_continental"))
		   (dot (aref df (string "duration_s")) (notna))
		   (> (aref df (string "duration_s")) 0)
		   (dot (aref df (string "summary_input_tokens")) (notna)))
		)))
   
   (do0
    (for ((ntuple name df) (list (list (string "work") df_valid)
				 (list (string "off") df_valid_off)))
     (for (s (list (string "-flash")
		   ;(string "-pro")
		   ))
	  (do0
	   (setf mask (df.model.str.contains s
					     :case False
					     :na False))
	   (setf dfm (aref df.loc mask))
	   (setf dat (/ (+ dfm.summary_input_tokens
			   dfm.summary_output_tokens)
			dfm.duration_s))
	   (setf bins (np.linspace 0
				   (np.percentile (dat.dropna) 99)
				   300))
	   (plt.hist dat
		     :log True
		     :bins bins
		     :label (+ name s)
		     :alpha .6
		     ))))

    
    (plt.xlabel (string "tokens/s"))
    (plt.legend)
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
identifier                                                               8123
model                       gemini-2.5-pro| input-price: 1.25 output-price...
summary                     **Abstract:**\n\nThis presentation by Thomas B...
summary_timestamp_start                            2025-09-29T11:11:19.613310
summary_timestamp_end                              2025-09-29T11:12:00.599444
summary_done                                                              1.0
summary_input_tokens                                                  22888.0
summary_output_tokens                                                  1162.0
host                                                             193.8.40.126
original_source_link              https://www.youtube.com/watch?v=0CepUaVqSeQ
embedding                                                                None
full_embedding                                                           None
ts_start                                     2025-09-29 11:11:19.613310+00:00
ts_end                                       2025-09-29 11:12:00.599444+00:00
duration_s                                                          40.986134
ts_eastern                                   2025-09-29 07:11:19.613310-04:00
ts_central                                   2025-09-29 06:11:19.613310-05:00
ts_mountain                                  2025-09-29 05:11:19.613310-06:00
ts_pacific                                   2025-09-29 04:11:19.613310-07:00
is_workhours_continental                                                False
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


