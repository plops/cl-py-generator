(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
 (defparameter *project* "143_helium_gemini")
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
       "# pip install -U google-generativeai"
       ; M-S-Enter in alive to execute top-level expression

       (imports ((genai google.generativeai)))
       (imports-from (fasthtml.common *))

       (with (as (open (string "api_key.txt"))
                 f)
             (setf api_key (dot f (read) (strip))))

       (genai.configure :api_key api_key)

       (@rt (string "/"))
       (def get ()
            (setf frm (Form
                       (Group
                        (Textarea :placeholder (string "Paste YouTube transcript here"
                                  :name (string "transcript"))
                        (Select (Option (string "gemini-1.5-pro-exp-0801")
                                                      (Option ))))))))))))