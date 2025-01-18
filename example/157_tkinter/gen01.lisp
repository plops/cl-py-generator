(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "157_tkinter")
  (defparameter *idx* "01") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format 
                          ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  
  (let* ((notebook-name "show_db")
	 (cols `(summary summary_done model cost
			 summary_input_tokens
			 summary_output_tokens
			 summary_timestamp_start
			 summary_timestamp_end
			 original_source_link
			 host))
	 (cols-show `(summary_timestamp_start cost summary_input_tokens title model)))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "Visualize sqlite database with youtube video summaries")
       (imports (
		 ;time
		 ; tqdm
					;subprocess
		 ;pathlib
		 ;(cv cv2)
					;argparse
		 (pd pandas)
		 ))
       (imports-from (tkinter *)
		     (sqlite_minutils *)
		     (tkinter ttk))
       (setf db (Database (string "/home/martin/summaries.db")))
       (setf items (Table db (string "items")))
       (setf res (list))
       (for (row items.rows)
	    (setf d "{}")
	    ,@(loop for e in cols
		    collect
		    `(setf (aref d (string ,e)) (aref row (string ,e))))
	    (setf (aref d (string "title"))
		  (dot (aref row (string "summary"))
		       (aref (split (string "\\n")) 0)))
	    (res.append d))
       (setf df (pd.DataFrame res))

       (setf root (Tk))
       (setf frm (ttk.Frame root :padding 10))
       (frm.grid)

       #+nil
       (do0 (dot ttk (Label frm :text (string "Hello World"))
		 (grid :column 0 :row 0))

	    
	    (dot ttk (Button frm :text (string "Quit")
				 :command root.destroy)
		 (grid :column 1 :row 0))
	    )
       ,@(loop for e in cols-show
	       and e-i from 0
	       collect
	       `(dot ttk (Label frm
				:text (string ,e)
				;:command (lambda () ,(lprint :msg e))
				)
		     (grid :column ,e-i :row 0)))

       (setf count 1)
       (for ((ntuple idx row) (dot (aref df (slice "" "" -1))  (iterrows)))
	    ,@(loop for e in cols-show
		    and e-i from 0
		    collect
		    `(dot ttk (Label frm
				     :text (aref row (string ,e))
				     )
			  (grid :column ,e-i :row count)))
	    (incf count))
       
       (root.mainloop)
       
       ,(lprint :msg "finished")
       )))
  )
