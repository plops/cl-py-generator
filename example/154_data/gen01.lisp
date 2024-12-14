(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "154_data")
  (defparameter *idx* "01") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *languages* `(en de fr ch nl pt cz it jp ar))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format (- (time.time) start_time)
                          ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  
  (let* ((notebook-name "show"))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       
       (imports (
		 time
		 json
					;os
		 tqdm
		 ;subprocess
		 ;pathlib
		 ;concurrent.futures
					; re
					;markdown
					; uvicorn
					;sqlite_minutils.db
		 ;datetime
					;time
		 (pd pandas)
		 (np numpy)
		 requests
		 random
		 ))

       (imports-from (sqlite_minutils *))
       (setf db (Database (string "tide.db")))
       (setf users (Table db (string "Users")))
       #+nil (setf q ("list" (for-generator (row (tqdm.tqdm users.rows)) (json.loads (aref row (string "data"))))))
       (setf res (list))
       (for (row (tqdm.tqdm users.rows))
	    (setf q (json.loads (aref row (string "data"))))
	    (setf d (dictionary :id (aref row (string "id"))
				))
	    (do0
	     ,@(loop for e in `(name
				
				birth_date
				bio
				schools
				jobs
				_id
				locations
				distance
			       
				)
		     collect
		     `(try
		       (do0 (setf (aref d (string ,e))
				  (aref q (string ,e)))
			    )
		       ("Exception as e"
			(print (string ,(format nil "no ~a" e)))
			pass)
		       )))
	    
	    (do0 (setf images (list))
		 (setf photos (aref q (string "photos")))
		 (setf num_photos (len photos))
					;(print (fstring "{d['name']} {len(photos)}"))
		 (for (image_data photos)
		      (try
		       (do0
			;(print (type image_data))
			(setf url
			      (aref (aref (aref image_data (string "processedFiles"))
					  0)
				    (string "url"))
			      )
			
			(images.append url))
		       ("Exception as e"
			pass)))
		 (try
		  (do0
		   #+nil (setf (aref d (string "user_url"))
			 (fstring "https://api.gotinder.com/v2/profile?include=account%2Cuser/{}"))
		   (setf (aref d (string "num_photos"))
			 num_photos)
		       (setf (aref d (string "images"))
			     images))
		  ("Exception as e"
		   pass))
		 )
	    
	    (try
	     (for (s (aref q (string "selected_descriptors")))
		  (try
		   (setf (aref d (aref s (string "name")))
			 (aref (aref (aref s (string "choice_selections"))
				     0)
			       (string "name")))
		   ("Exception as e"
		    pass)))
	     ("Exception as e"
	      pass))
	    
	    (res.append d))
       (setf df2 (pd.DataFrame res))
       (setf df1 (df2.drop_duplicates :subset (string "id")
						  ))
       (setf df0 (df1.set_index (string "id")
				:drop True
				:append False
				:inplace False))
       (setf (aref df0 (string "age"))
	     (dot df0 birth_date (apply (lambda (x)
					  (? (== (type (string "bla")) (type x))
					     (- 2024 (int (aref x (slice 0 4))))
					     0
					     ))
					1)))
       ,(let ((l `((:column Smoking :values ((:value Non-smoker :weight 10))
		    :nan t)
		   (:column "Family Plans" :values ((:value "Not sure yet" :weight 3)
						    (:value  "I want children" :weight 10)
						    )
		    :nan t)
		   (:column Drinking :values ((:value "Not for me" :weight 7)
					      (:value "Sober" :weight 5))
		    :nan t)
		   (:column Workout :values ((:value "Often" :weight 4)
					     (:value "Everyday" :weight 3)
					     (:value "Gym rat" :weight 3))
		    :nan t)
		   (:column Education :values ((:value "Bachelors" :weight 5)
					       (:value "Masters" :weight 6)
					       (:value "PhD" :weight 7))
		    :nan t)
		   )))
	  `(do0
	    (setf df
		  (aref df0
			(logand
			 ,@(loop for e in l
				 collect
				 (destructuring-bind (&key column values nan) e
				   `(logior
				     ,@(loop for ev in values
					     collect
					     (destructuring-bind (&key value weight) ev
					      `(== (aref df0 (string ,column))
						   (string ,value))))
				     ,(if nan
					  `(dot (aref df0 (string ,column))
						(isna))
					  `False))))
			 )))
	    (def computeWeight (row)
	      (setf sum 0)
	      ,@(loop for e in l
		      appending
		      (destructuring-bind (&key column values nan) e
			(loop for ev in  values
			      collect
			      (destructuring-bind (&key value weight) ev
				`(try
				  (when (== (aref row (string ,column))
					    (string ,value))
				    (incf sum ,weight))
				  ("Exception as e"
				   pass))))))
	      
	      (return sum))
	    (setf (aref df (string "weight")) (dot df (apply computeWeight :axis 1)))
		  ))
       (setf df (dot df (sort_values :by (string "weight")
				     :ascending False)))
       #+nil(for ((ntuple idx row) (df.iterrows))
		 (unless (== row.bio (string " "))
		   (print row.bio)))

       (print     (aref df (list (string "name")
				 (string "weight")
				 (string "age")
				 (string "bio")
				 (string "num_photos"))))

       (setf sleep_max_for 1.3)
       (do0
	(with (as (open (string "token")) f)
	      (setf token (dot f (read) (strip))))
	(for ((ntuple idx (tuple row_idx row)) (tqdm.tqdm (enumerate (df.iterrows))))
	     (setf url (fstring "https://api.gotinder.com/like/{row._id}"))
	     (setf header "{}")
	     (setf (aref header (string "X-Auth-Token"))
		   token)
	     (setf data (dot  (requests.get url :headers header)
			      (json))
		   likes (aref data (string "likes_remaining"))
		   match (aref data (string "match"))
		   ;;1734151873712
		   limit (aref data (string "rate_limited_until")))
	     (print (fstring "match={match} {likes} likes remaining"))
	     (print data)
	   
	     (time.sleep (* (random.random) sleep_max_for)))
	)
       #+nil 
       (do0
	
	
	(for ((ntuple idx (tuple row_idx row)) (tqdm.tqdm (enumerate (df.iterrows))))
	     (for ((ntuple i url) (enumerate row.images))
		  (setf req (requests.get url :stream True))
		  (when (== req.status_code 200)
		    (with (as (open (fstring "img/{idx:04}_{row._id}_{row['name']}_{i}.jpg")
				    (string "wb"))
			      f)
			  (f.write req.content)))
		  (time.sleep (* (random.random) sleep_max_for)))))
       ))))
