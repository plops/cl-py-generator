(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

;; TODO
;; [ ] list mp4 files
;; [ ] store files in database
;; [ ] populate database with information from the video files (size, duration, resolution, framerate, bitrate)
;; [ ] allow adding more files, database migration
;; [ ] allow viewing in browser
;; [ ] gui for rating (good actor, good story, tags)
;; [ ] pre-load (if necessary)
;; [ ] whisper (convert audio to searchable transcript)
;; [ ] mediapipe (count and track people, faces, hands)
;; [ ] OCR for text in the video
;; [ ] summarization (using transcript, OCR results and possibly scene descriptions using multi-modal model like llama 3.2) 

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ((db-cols `((:name identifier :type int)
		 (:name path :type str)
		 (:name tags :type str)
		 (:name video_width :type int)
		 (:name video_height :type int)
		 (:name size_bytes :type int)
		 (:name atime :type int)
		 (:name mtime :type int)
		 (:name ctime :type int)
		 (:name ffmpeg_text :type str)
		 (:name duration_s :type float)
		 (:name framerate :type float)
		 (:name video_bitrate_Mbit :type float)
		 (:name audio_bitrate_Mbit :type float)
		 (:name audio_language :type str)
		 (:name partial :type bool)
		 (:name duplicates :type str)
		 ,@(loop for e in `(actors story)
			 collect
			 `(:name ,(format nil "rating_~a" e)
			   :type int))
		 )))
  (defparameter *project* "149_host_videos")
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

  
  (let* ((notebook-name "fill"))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U python-fasthtml")

       (comments "Purpose of this file is to populate the video.db sqlite database with filenames and whatever information may be useful (filesize, video duration, dimensions)")
       (imports (
					;os
		 tqdm
		 subprocess
		 pathlib
		 concurrent.futures
					; re
					;markdown
					; uvicorn
					;sqlite_minutils.db
		 datetime
					;time
		 ))
       
       (imports-from (fasthtml.common *))

       " "
       (def find_mp4_files (directory)
	 (setf root_path (pathlib.Path directory))
	 (setf mp4_files ("list" (root_path.rglob (string "*.mp4*"))))
	 (return mp4_files))

       (setf db_exists (dot pathlib (Path (string "data/video.db"))
		    (exists)))
       (when db_exists
	 (print (string "video.db file exists. use it")))
       (unless db_exists
	 (setf mp4_files (find_mp4_files (string "videos/"))))
       
       " "
       (comments "open website")
       (comments "videos is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
       (setf (ntuple app rt videos VideoEntry)
	     (fast_app :db_file (string "data/video.db")
		       :live True
					;:render render
		       ,@(loop for e in db-cols
			       appending
			       (destructuring-bind (&key name type no-show) e
				 `(,(make-keyword (string-upcase (format nil "~a" name)))
				   ,type)))

		       
		       :pk (string "identifier")
		       ))
       (print (string "...loaded"))
       (unless db_exists
	(do0
	 (print (string "first iteration: collecting file sizes"))
	 (comments "collecting file sizes and file times (cold over network: 100it/s, hot: 1800 it/s, local: 840it/s)")
	 (for (f (tqdm.tqdm mp4_files))
	      (setf v (VideoEntry))
	      
	      (setf stat (dot f (stat)))
	      ,@(loop for (e f) in `(	;(identifier idx)
				     (path (str f))
				     (size_bytes stat.st_size)
				     (atime stat.st_atime)
				     (mtime stat.st_mtime)
				     (ctime stat.st_ctime)
					;(ffmpeg_text ffmpeg_text.stderr)
				     )
		      collect
		      `(setf (dot v ,e) ,f))
	      (videos.insert v)))
	
	)

       (def collect_video_metadata (v)
	 (declare (type VideoEntry v))
	 (when (is v.path "not None")
	   (unless v.ffmpeg_text
	     (setf ffmpeg_text (subprocess.run (list (string "ffmpeg")
						     (string "-i")
						     v.path)
					       :capture_output True
					       :text True))
	     ,@(loop for (e f) in `((ffmpeg_text ffmpeg_text.stderr))
		     collect
		     `(setf (dot v ,e) ,f))
	     (videos.upsert v)))
	 (return None))
       #+nil 
       (do0
	(comments "run ffmpeg for each video file to collect information about width, height, bitrate")
	(print (string "second iteration: collecting video metadata"))
	(with (as (concurrent.futures.ThreadPoolExecutor :max_workers 12)
		  executor)
	      (setf future_to_video "{}")
	      (for (v (videos))
		   (executor.submit collect_video_metadata
				    v))
	      (for (future (tqdm.tqdm (concurrent.futures.as_completed future_to_video)
				      :total (len (videos))))
		   (try
		       (setf result (future.result))
		     ("Exception as e"
			 (print (fstring "{vido.path} generated an exception: {e}"))))))

	#+nil
	(do0 (comments "single threaded version of the code above")
	 (for (v (tqdm.tqdm (videos)))
	      (when (is v.path "not None")
		(unless v.ffmpeg_text
		  (setf ffmpeg_text (subprocess.run (list (string "ffmpeg")
							  (string "-i")
							  v.path)
						    :capture_output True
						    :text True))
		  ,@(loop for (e f) in `((ffmpeg_text ffmpeg_text.stderr))
			  collect
			  `(setf (dot v ,e) ,f))
		  (videos.upsert v))))))

       (do0
	
	(comments "parse ffmpeg output and populate db, for now i want duration, width, height, bitrate")
	
	(def parse_ffmpeg (v)
	  (try
	      (do0
	       (setf ftext (dot (string "\\n")
				(join (aref (v.ffmpeg_text.split (string "\\n"))
					    (slice 12 "")))))
	       (comments "  creation_time   : 2018-02-13T01:59:51.000000Z
Duration: 00:02:08.46, start: 0.000000, bitrate: 442 kb/s")
	       ,@(loop for e in `(creation_time Duration)
		       collect
		       (let ((e-match (format nil "~a_match" e))
			     )
			 `(do0
			   (setf ,e-match (re.search (rstring3 ,(format nil "[ ]*~a[ ]*:[ ]*([^ ,]+)" e))
						     ftext))
			   (setf ,e (dot ,e-match (group 1))))))

	       ,@(loop for e in `(bitrate)
		       collect
		       (let ((e-match (format nil "~a_match" e))
			     (e-unit (format nil "~a_unit" e)))
			 `(do0
			   (setf ,e-match (re.search (rstring3 ,(format nil "[ ]*~a[ ]*:[ ]*(\\S+)[ ]+(\\S+)" e))
						     ftext))
			   (setf ,e (dot ,e-match (group 1)))
			   (setf ,e-unit (dot ,e-match (group 2))))))

	       ,(let ((l-video `(channel codec format  width height dimension_extra bitrate bitrate_unit framerate framerate_unit
				 )))
		 `(do0
		  (comments "Stream #0:0(und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p, 640x360 [SAR 1:1 DAR 16:9], 674 kb/s, 25 fps, 30 tbr, 90k tbn, 60 tbc (default)")
		  (do0
		   (setf video_info0 (re.search (rstring3 "Stream .*: Video:.*(\\d+)x(\\d+).*([0-9.]+) kb/s, ([0-9.]*) fps,.*")
					      ftext)))

		  ,@(loop for e in l-video
			  collect
			  `(setf ,(format nil "video_~a" e)
				 None))
		  (when video_info0
		    ,@(loop for e in `(width height bitrate framerate)
			    and e-i from 1
			    collect
			    `(setf ,(format nil "video_~a" e)
				   (dot video_info0 (group ,e-i))))
		    )
		  
		  (setf video_info (re.search (rstring3 "Stream (\\S+): Video: ([^,]+), (.*), (\\d+)x(\\d+)[ ]*([^,]*), ([0-9.]+) ([^,]*), ([\\d.]+) ([^,]*),.*")
					      ftext))
		  ,@(loop for e in l-video
			  collect
			  `(setf ,(format nil "video_~a" e)
				 (string "-1")))
		  (when video_info
		    ,@(loop for e in l-video
			    and e-i from 1
			    collect
			    `(setf ,(format nil "video_~a" e)
				   (dot video_info (group ,e-i))))
		    )))

	       (comments "convert 00:07:27.65 into seconds")
	       ;;(setf Duration (string "00:07:27.65"))
	       ;(print Duration)
	       (setf time_diff (- (dot datetime
				       datetime
				       (strptime Duration
						 (string "%H:%M:%S.%f")))
				  (dot datetime
				       datetime
				       (strptime (string "00:00:00.00")
						 (string "%H:%M:%S.%f")))))
	       (setf duration_s (dot time_diff
				     (total_seconds)))
	       (setf number_frames (* (float video_framerate)
				      duration_s))
	       (setf estimated_bytes (* duration_s
					(/ (* 1024 (int video_bitrate))
					   8)))
	       (setf bit_per_pixel (/ (* 8 (int v.size_bytes))
				      (* (int video_width)
					 (int video_height))))
	       (setf bit_per_pixel_and_frame (/ bit_per_pixel
						number_frames))
	       (setf v.duration_s duration_s
		     v.video_bitrate_mbit bitrate
		     v.video_width video_width
		     v.video_height video_height
		     v.framerate video_framerate)
	       (videos.upsert v)
	       )
	      ("Exception as e"
	       (print (fstring "exception {v.path} {e}"))
	       pass))
	     
	     )
	
	(print (string "parse ffmpeg output")) ;; 2000 it/s
	(for (v (tqdm.tqdm (videos)))
	     (parse_ffmpeg v)))
       

       )))

  (let* ((notebook-name "host")

	 )
    (write-source
     (format nil "~a/source01/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U python-fasthtml")

       (imports (
					;os
		 random
		 pathlib
		 multiprocessing
		 re
		 ;markdown
		 ; uvicorn
		 ;sqlite_minutils.db
		 datetime
		 ;time
		 ))
       
       (imports-from (fasthtml.common *))

       " "

       
       " "
       (comments "open website")
       (comments "videos is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table")
       (setf (ntuple app rt videos VideoEntry)
	     (fast_app :db_file (string "data/video.db")
		       :live True
		       ;:render render
		       ,@(loop for e in db-cols
			       appending
			       (destructuring-bind (&key name type no-show) e
				 `(,(make-keyword (string-upcase (format nil "~a" name)))
				   ,type)))

		       
		       :pk (string "identifier")
		       ))

       " "
       (def render (video)
	 (declare (type VideoEntry video))
	 (setf identifier video.identifier)
	 (setf sid (fstring "gen-{identifier}"))
	 (return (Div (NotStr sid))))
       
       
       (@rt (string "/"))
       (def get (request)
	 (declare (type Request request))
	 	 
	 
	 (setf nav (Nav
		    (Ul (Li (Strong (string "Video Viewer"))))
		    (Ul	(Li (A (string "Overview")
			       :href (string "https://www.youtube.com/watch?v=ttuDW1YrkpU")))
			(Li (A (string "Details")
			       :href (string "https://github.com/plops/gemini-competition/blob/main/README.md")))
			)))
	 
	 

	
	 (setf form
	       (Form
                (Group
		 (Div 
		  
		  (Textarea :placeholder (string "Add tags, e.g. outside mountains night")
				    
				    :name (string "tags"))
		  
		  (Div (Label (string "Language") :_for (string "audio_language"))
		       (Select
			,@(loop for e in *languages*
				collect
				`(Option (string ,e)))
			:style (string "width: 100%;")
			:name (string "audio_language")
			:id (string "audio_language"))
		       :style (string "display: flex; align-items: center; width: 100%;"))
		  ,@(loop for e in `(actors story)
			  collect
			  (let ((id (format nil "~a_rating" e)))
			   `(Div (Label (string ,(format nil "~a Rating" (string-capitalize e)))
				       :_for (string ,id))
				(Select
				 ,@(loop for e in `(1 2 3 4 5)
					 collect
					 `(Option (string ,e)))
				 :style (string "width: 100%;")
				 :name (string ,id)
				 :id (string ,id))
				:style (string "display: flex; align-items: center; width: 100%;"))))
		  ,@(loop for (e f default) in `((partial "Video is incomplete" False)
						 )
			  collect
			  `(Div
			    (Input :type (string "checkbox")
				   :id (string ,e)
				   :name (string ,e)
				   :checked ,default)
			    (Label (string ,f) :_for (string ,e))
			    :style (string "display: flex; align-items: center; width: 100%;")))
		  
		  :style (string "display: flex; flex-direction:column;"))
		 )
		:hx_post (string "/process_transcript")
		:hx_swap (string "afterbegin")
		:target_id (string "gen-list")))

	 (return (ntuple (Title (string "Video Viewer"))
			
			 (Main nav
			       (Div
				(Video
				 (Source 
				  :src (dot (random.choice (videos))
					    path)
				  :type (string "video/mp4"))
				 (NotStr (string "Your browser does not support the video tag."))
				 :style (string "margin-bottom: 20px;")
				 :height (string "auto")
				 :width (string "100%")
				 :controls True
				 :id (string "my_video")
				 ))
			       form
			       (Script (string3 "var myVideo = document.getElementById('my_video');
myVideo.muted = true;"))
			       :cls (string "container")))))
 
       " "
       (@app.post (string "/generations/{identifier}"))
       (def get (identifier)
	 (declare (type int identifier))
	 (return (generation_preview identifier)))

       " "
       (@rt (string "/process_transcript"))
       (def post (summary request)
	 (declare (type Summary summary)
		  (type Request request))
	 ,(lprint :msg "POST process_transcript" :vars `(summary request)))

       ,(let ((l-video `(channel codec format  width height dimension_extra bitrate bitrate_unit framerate framerate_unit
				 )))
	 `(do0
	   " "
	   (@rt (string "/video/{id}")
		:name (string "video"))
	   (def get (id)
	     (declare (type "int=4" id))
	     (setf v (aref videos id))
	     (setf compute_results (Div (NotStr (string "error"))))
	     (try
	      (do0
	       (setf ftext (dot (string "\\n")
				(join (aref (v.ffmpeg_text.split (string "\\n"))
					    (slice 12 "")))))
	       (comments "  creation_time   : 2018-02-13T01:59:51.000000Z
Duration: 00:02:08.46, start: 0.000000, bitrate: 442 kb/s")
	       ,@(loop for e in `(creation_time Duration)
		       collect
		       (let ((e-match (format nil "~a_match" e))
			     )
			 `(do0
			   (setf ,e-match (re.search (rstring3 ,(format nil "[ ]*~a[ ]*:[ ]*([^ ,]+)" e))
						     ftext))
			   (setf ,e (dot ,e-match (group 1))))))

	       ,@(loop for e in `(bitrate)
		       collect
		       (let ((e-match (format nil "~a_match" e))
			     (e-unit (format nil "~a_unit" e)))
			 `(do0
			   (setf ,e-match (re.search (rstring3 ,(format nil "[ ]*~a[ ]*:[ ]*(\\S+)[ ]+(\\S+)" e))
						     ftext))
			   (setf ,e (dot ,e-match (group 1)))
			   (setf ,e-unit (dot ,e-match (group 2))))))

	       (do0
		(comments "Stream #0:0(und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p, 640x360 [SAR 1:1 DAR 16:9], 674 kb/s, 25 fps, 30 tbr, 90k tbn, 60 tbc (default)")
		(setf video_info (re.search (rstring3 "Stream (\\S+): Video: ([^,]+), ([^,]+), (\\d+)x(\\d+)[ ]*([^,]*), ([0-9.]+) ([^,]*), ([\\d.]+) ([^,]*),.*")
					   
					    ftext))
		,@(loop for e in l-video
			collect
			`(setf ,(format nil "video_~a" e)
			       (string "-1")))
		(when video_info
		  ,@(loop for e in l-video
			  and e-i from 1
			  collect
			  `(setf ,(format nil "video_~a" e)
				 (dot video_info (group ,e-i))))
		  ))

	       (comments "convert 00:07:27.65 into seconds")
					;(setf Duration (string "00:07:27.65"))
	       (print Duration)
	       (setf time_diff (- (dot datetime
				       datetime
				       (strptime Duration
						 (string "%H:%M:%S.%f")))
				  (dot datetime
				       datetime
				       (strptime (string "00:00:00.00")
						 (string "%H:%M:%S.%f")))))
	       (setf duration_s (dot time_diff
				     (total_seconds)))
	       (setf number_frames (* (float video_framerate)
				      duration_s))
	       (setf estimated_bytes (* duration_s
					(/ (* 1024 (int video_bitrate))
					   8)))
	       (setf bit_per_pixel (/ (* 8 (int v.size_bytes))
				      (* (int video_width)
					 (int video_height))))
	       (setf bit_per_pixel_and_frame (/ bit_per_pixel
						number_frames))
	       (setf compute_results (Div (Pre
					   ,@(loop for e in `(size_bytes)
						   collect
						   `(NotStr (dot (string ,(format nil "~a: {}" e))
								 (format (dot v ,e)))))
					   ,@(loop for e in `(
							      atime
							      mtime
							      ctime
							      )
						   collect
						   `(NotStr (dot (string ,(format nil "~a: {}" e))
								 (format (dot datetime
									      datetime
									      (fromtimestamp (dot v ,e))
									      (isoformat)))))
						   )
					   (NotStr (fstring "creation_time: {creation_time} duration: {Duration} bitrate: {bitrate} {bitrate_unit}"))
					   
					   )
					  (Pre (NotStr (fstring ,(format nil "~{~a~^, ~}"
									 (loop for e in `(bit_per_pixel duration_s estimated_bytes number_frames bit_per_pixel_and_frame)
									       collect
									       (format nil "~a: {~a:3.2f}"
										       e e))))))

					  (Pre (NotStr (fstring ,(format nil "~{~a~^, ~}"
									 (loop for e in l-video
									       collect
									       (format nil "~a: {video_~a}"
										       e e))))))
					  (Pre ftext)) ))
	      ("Exception as e"
	       pass))
	     
	     
	     (return (Body (H4 (fstring "{v.identifier} {v.path}"))
			   (A (string "Prev")
			      :href (fstring "/video/{id-1}"))
			   (A (string "Next")
			      :href (fstring "/video/{id+1}")
			      )
			   (Div
			    (Video
			     (Source 
			      :src (fstring "/{v.path}")
			      :type (string "video/mp4"))
			     (NotStr (string "Your browser does not support the video tag."))
			     :style (string "margin-bottom: 20px;")
			     :height (string "auto")
			     :width (string "30%")
			     :controls True
			     :id (string "my_video")
			     )

			    compute_results)
			   (Script (string3 "var myVideo = document.getElementById('my_video');
myVideo.muted = true;"))
			   )))))

       (serve :host (string "0.0.0.0") :port 5001)
       
       #+nil
       (do0
	(def serve_in_process ()
	  (serve :host (string "0.0.0.0") :port 5001))

	(when (== __name__ (string "__main__"))
	  (setf process (multiprocessing.Process :target serve_in_process))
	  (process.start)))
       #+nil
       (do0
	(imports-from (httpx ASGITransport AsyncClient)
		      (anyio from_thread)
		      (functools partialmethod))
	(class Client ()
	       (def __init__ (self app &key (url (string "http://localhost:5001")))
		 (setf self.cli (AsyncClient :transport
					     (ASGITransport app)
					     :base_url url)))
	       (def _sync (self method url **kwargs)
		 (space async
			(def _request ()
			  (return (space await
					 (self.cli.request method url **kwargs)))))
		 (with (as (from_thread.start_blocking_portal)
			   portal)
		       (return (portal.call _request))))
	       )
	(for (o (dot (string "get post delete put patch options")
		     (split)))
	     (setattr Client
		      o
		      (partialmethod Client._sync o)))
	(setf cli (Client app))
	(print (dot cli
		    (get (string "/"))
		    text))

	(print (dot cli
		    (get (string "/video/32"))
		    text)))
       ))))
