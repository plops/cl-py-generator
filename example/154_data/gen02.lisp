(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more
				     :try)))
(setf *features* (set-difference *features* '(:more
					      ;:try
					      )))

(let ()
  (defparameter *project* "154_data")
  (defparameter *idx* "02") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *languages* `(en de fr ch nl pt cz it jp ar))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format; (- (time.time) start_time)
                          ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  
  (let* ((notebook-name "swipe"))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "micromamba install requests sqlite_minutils")
       (imports (
		 time
		 ;json
		 sys
					;os
		 ;tqdm
		 ;subprocess
		 ;pathlib
		 ;concurrent.futures
					; re
					;markdown
					; uvicorn
					;sqlite_minutils.db
		 ;datetime
					;time
		 ;(pd pandas)
		 ;(np numpy)
		 requests
		 random
		 datetime
		 ))

       (imports-from (sqlite_minutils *))
       (setf URL (string "https://api.gotinder.com"))
       (with (as (open (string "token")) f)
	      (setf token (dot f (read) (strip))))

       ,(let* ((l0 `((:var name :fun (data.get (string "name")
					      (string "Unknown")))
		    (:var bio :fun (data.get (string "bio")
					     (string "")))
		    (:var distance :fun (/ (data.get (string "distance_mi")
						     0)
					   1.60934))
		    (:var birth_date
		     :fun (? (data.get (string "birth_date")
				       False)
			     (int
			      (aref (aref data (string "birth_date"))
				    (slice 0 4))
			      #+nil
			      (datetime.datetime.strptime
			       (aref data (string "birth_date"))
			       (string "%Y")))
			     0)
		     :type int)
		    (:var gender :fun (aref (list (string "Male")
					     (string "Female")
					     (string "Unknown"))
				       (data.get (string "gender")
						 2)))
		    (:var images :fun ("list"
				       (map
					(lambda (p)
					  (aref p (string "url")))
					(data.get (string "photos")
						  (list)))))
		    ))
	      (l1 `((:var family_plans :json "Family Plans")
		    (:var smoking :json "Smoking")
		    (:var drinking :json "Drinking")
		    (:var workout :json "Workout")
		    (:var education :json "Education")
		    (:var personality_type :json "Personality Type")))
	       (l (append
		   (loop for e in l0
			 collect
			 (destructuring-bind (&key var fun (type 'str)) e
			   `(:var ,var :fun ,fun :type ,type :json nil)))
		   (loop for e in l1
			 collect
			 ;; data['selected_descriptors'][1].get('choice_selections')[0].get('name')

			 (destructuring-bind (&key var fun (type 'str) json) e
			   `(:var ,var 
			     :fun (next
				   (for-generator (s (aref data (string "selected_descriptors")))
						  (? (== (s.get (string "name"))
							 (string ,json))
						     (dot s
							  (aref (get (string "choice_selections"))
								0)
							  (get (string "name")))
						     (string ""))))
			     :type ,type
			     :json ,json))))))
	   `(do0
	     
	     (class Person (object)
		    (def __init__ (self data api)
		      (setf self._api api
			    self.data data
			    self.id (aref data (string "_id"))))
		    (def parse (self &key (add_data True))
		      (setf data self.data)
		      (setf d (dictionary :id self.id
					  ))
		      (when add_data
			(setf (aref d (string "data"))
			      data))
		      ,@(loop for e in l
			      collect
			      (destructuring-bind (&key var fun type json) e
				`(try
				  (do0
				   (setf (aref d (string ,var))
					 ,fun))
				  ("Exception as e"
				   ;(print (fstring ,(format nil "125: '~a' {e}" var)))
				   pass))))
		      (return d)))
	     (class API ()
		    (def __init__ (self token)
		      (setf self._token token))
		    (def get (self link)
		      
		      (setf headers "{}")
		      (setf (aref headers (string "X-Auth-Token"))
			    self._token)
		      (setf combined_url (+ URL link))
		      (print combined_url)
		      (setf data (dot
				  requests
				  (get combined_url
				       :headers headers)))
		      (return (data.json)))
		    (def like (self id)
		      (return (self.get (fstring "/like/{id}"))))
		    (def dislike (self id)
		      (return (self.get (fstring "/pass/{id}"))))
		    (def nearby_persons (self)
		      (setf data (self.get (string "/v2/recs/core")))
		      "global q"
		      (setf q data)
		      ;(print data)
		      (return ("list"
			       (map
				(lambda (user)
				  (Person (aref user (string "user"))
					  self))
				(dot data
				     (get (string "data"))
				     (get (string "results"))))))))
	      
	     (setf datetime_str (dot datetime
				     datetime
				     (now)
				     (strftime (string "%Y%m%d%H%M%S")))
		   db_fn (fstring "tide_{datetime_str}.db"))
	     (setf db (Database db_fn))
	     (setf users (Table db (string "Users")))
	     (setf schema (dictionary
			   :id str
			   :data str
			   ,@(loop for e in l
				   appending
				   (destructuring-bind (&key var fun type json) e
				     `(,(make-keyword (string-upcase (format nil "~a" var)))
				       ,type)))))
	     
	     (users.create :columns schema
			   :pk (string "id"))))

       (setf api (API token))
       ,(let ((l-choice
	       `((:column smoking :values ((:value Non-smoker :weight 10))
		  :nan t)
		 (:column family_plans :values ((:value "Not sure yet" :weight 3)
						(:value  "I want children" :weight 10)
						)
		  :nan t)
		 (:column personality_type :values ((:value "INFJ" :weight 3)
						     )
		  :nan nil)
		 (:column drinking :values ((:value "Not for me" :weight 7)
					    (:value "Sober" :weight 5))
		  :nan t)
		 (:column workout :values ((:value "Often" :weight 4)
					   (:value "Everyday" :weight 3)
					   (:value "Gym rat" :weight 3))
		  :nan t)
		 #+nil (:column education :values ((:value "Bachelors" :weight 5)
					     (:value "Masters" :weight 6)
					     (:value "PhD" :weight 7))
		  :nan t)
		 )))
	`(while True
	       (time.sleep (* (random.random) 3.2))
	       (#-try do0 #+try try 
		(do0
		 (setf persons (api.nearby_persons))
		 (setf l  (len persons)) (print (fstring "len(persons)={l}") )
		 (for (person persons)
		      (#-try do0 #+try try
		       (do0
			#+nil (do0 (comments "check if person already exists in db")
				   (setf known (bool
						(dot users
						     (select :where (dictionary :id person.id))))))
			(do0		;unless known
		       
			     (comments "don't update existing entry")
			     (setf p (person.parse))
			     (users.insert p
					   :ignore True)
			     (setf name (aref p (string "name"))
				   smoking (aref p (string "smoking"))
				   family (aref p (string "family_plans")))
			     ,(lprint :vars `(person.id
					      name
					      smoking
					      family))
			     (when (and
				    ,@(loop for e in l-choice
					    collect
					    (destructuring-bind (&key column values nan) e
					      `(or ,@(loop for f in values
							   collect
							   (destructuring-bind (&key value weight) f
							     `(== (dot p (get (string ,column)))
								  (string ,value))))
						   (== (dot p (get (string ,column)))
						       (string "")))
					      )))
			       (print (fstring "liking {p['name']}"))
			       (setf like_result (api.like person.id))
			       (print like_result)
			       (when (like_result.get (string "rate_limited_until")
						      False)
				 (sys.exit 1)))))
		       
		       #+try ("Exception as e"
			      (print (fstring "169: {e}"))
			      pass))))
		#+try ("Exception as e"
		       (print (fstring "162: {e}"))
		       pass))
	       ))
       
       #+nil (setf q ("list" (for-generator (row (tqdm.tqdm users.rows)) (json.loads (aref row (string "data"))))))

       #+nil(do0
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
				 :inplace False)))
       #+nil
       (do0
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
	     )))
       #+nil 
       (setf df (dot df (sort_values :by (string "weight")
				     :ascending False)))
       #+nil(for ((ntuple idx row) (df.iterrows))
		 (unless (== row.bio (string " "))
		   (print row.bio)))
 #+nil
       (print     (aref df (list (string "name")
				 (string "weight")
				 (string "age")
				 (string "bio")
				 (string "num_photos"))))

 #+nil(setf sleep_max_for 1.3)
     #+nil  (do0
	
	(for ((ntuple idx (tuple row_idx row)) (tqdm.tqdm (enumerate (df.iterrows))))
	     (setf url (fstring "{URL}/like/{row._id}"))
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
       )))
  )
