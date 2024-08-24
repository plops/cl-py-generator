(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
 (defparameter *project* "144_fasthtml")
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

 (let* ((notebook-name "host"))
   (write-source
     (format nil "~a/source/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "pip install -U python-fasthtml")
       (comments "https://youtu.be/evAb2x34Jqk?t=312")
       ;; In vscode with alive extension press M-S-Enter to execute top-level expression

       (imports (datetime))
       (imports-from (fasthtml.common *))

       (def render (comment)
	 (return (Li (A comment.comment
			:href (fstring "/comments/{comment.id}")))))

       (setf (ntuple app rt comments Comment)
	     (fast_app (string "data/comments.db")
		       :id int
		       :comment str
		       :user str
		       :created_at str
		       :render render
		       :pk (string "id")))

       (@rt (string "/"))
       (def get ()
	 (setf create_comment (Form
			       (Input :id (string "username")
				      :name (string "user")
				      :placeholder (string "username"))
			       (Textarea :id (string "comment")
				      :name (string "comment")
				      :placeholder (string "comment"))
			       (Button (string "Add Comment"))
			       :hx_post (string "/comments")
			       :hx_target (string "#comments")
			       :hx_swap (string "afterbegin")))
	 (setf comments_list (Ul (*comments :order_by (string "id DESC"))
				 :id (string "comments")))
	 (return (Div comments_list
		      create_comment
		      :cls (string "container"))))

       (@rt (string "/comments"))
       (space async (def post (comment)
		      (declare (type Comment comment))
		      (setf comment.created_ad (dot datetime datetime
						    (now) (isoformat)))
		      (return (comments.insert comment))))

       (serve)
       ))))
