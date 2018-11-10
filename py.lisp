;(ql:quickload "optima")
;(ql:quickload "alexandria")

(in-package :cl-py-generator)
(setf (readtable-case *readtable*) :invert)

(defparameter *file-hashes* (make-hash-table))

(defun write-source (name code &optional (dir (user-homedir-pathname)))
  (let* ((fn (merge-pathnames (format nil "~a.py" name)
			      dir))
	(code-str (emit-py
		   :clear-env t
		   :code code))
	(fn-hash (sxhash fn))
	 (code-hash (sxhash code-str)))
    (multiple-value-bind (old-code-hash exists) (gethash fn-hash *file-hashes*)
     (when (or (not exists) (/= code-hash old-code-hash))
       ;; store the sxhash of the c source in the hash table
       ;; *file-hashes* with the key formed by the sxhash of the full
       ;; pathname
       (setf (gethash fn-hash *file-hashes*) code-hash)
       (with-open-file (s fn
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
	 (write-sequence code-str s))
       #+nil

       (sb-ext:run-program "/usr/bin/autopep8" (list "--max-line-length 80" (namestring fn)))
       (sb-ext:run-program "/usr/bin/yapf" (list (namestring fn)))))))

(defun print-sufficient-digits-f64 (f)
  "print a double floating point number as a string with a given nr. of
  digits. parse it again and increase nr. of digits until the same bit
  pattern."
  (let* ((ff (coerce f 'double-float))
	 (s (format nil "~E" ff)))
    #+nil (assert (= 0d0 (- ff
			    (read-from-string s))))
    (assert (< (abs (- ff
		       (read-from-string s)))
	       1d-12))
   (substitute #\e #\d s)))

;(print-sufficient-digits-f64 1d0)


(defparameter *env-functions* nil)
(defparameter *env-macros* nil)



(defun emit-py (&key code (str nil) (clear-env nil) (level 0))
  ;(format t "emit ~a ~a~%" level code)
  (when clear-env
    (setf *env-functions* nil
	  *env-macros* nil))
  (flet ((emit (code &optional (dl 0))
	   (emit-py :code code :clear-env nil :level (+ dl level))))
    (if code
	(if (listp code)
	    (case (car code)
	      (tuple (let ((args (cdr code)))
		       (format nil "(~{~a,~})" (mapcar #'emit args))))
	      (paren (let ((args (cdr code)))
		       (format nil "(~{~a~^, ~})" (mapcar #'emit args))))
	      (ntuple (let ((args (cdr code)))
		       (format nil "~{~a~^, ~}" (mapcar #'emit args))))
	      (list (let ((args (cdr code)))
		      (format nil "[~{~a~^, ~}]" (mapcar #'emit args))))
              (dict (let* ((args (cdr code)))
		      (let ((str (with-output-to-string (s)
				   (loop for (e f) in args
				      do
					(format s "(~a):(~a)," (emit e) (emit f))))))
			(format nil "{~a}" ;; remove trailing comma
				(subseq str 0 (- (length str) 1))))))
	      (indent (format nil "~{~a~}~a"
			      (loop for i below level collect "    ")
			      (emit (cadr code))))
	      (do (with-output-to-string (s)
		    (format s "~{~a~%~}" (mapcar #'(lambda (x) (emit `(indent ,x) 1)) (cdr code)))))
	      (class (destructuring-bind (name parents &rest body) (cdr code)
		       (format nil "class ~a~a:~%~a"
			       name
			       (emit `(paren ,@parents))
			       (emit `(do ,@body)))))
	      (do0 (with-output-to-string (s)
		     (format s "~a~%~{~a~%~}"
			     (emit (cadr code))
			     (mapcar #'(lambda (x) (emit `(indent ,x) 0)) (cddr code)))))
	      (lambda (destructuring-bind (lambda-list &rest body) (cdr code)
		     (multiple-value-bind (req-param opt-param res-param
						     key-param other-key-p aux-param key-exist-p)
			 (parse-ordinary-lambda-list lambda-list)
		       (declare (ignorable req-param opt-param res-param
					   key-param other-key-p aux-param key-exist-p))
		       (with-output-to-string (s)
			 (format s "lambda ~a: ~a"
				 (emit `(ntuple ,@(append req-param
							 (loop for e in key-param collect 
							      (destructuring-bind ((keyword-name name) init suppliedp)
								  e
								(declare (ignorable keyword-name suppliedp))
								`(= ,name ,init))))))
				 (if (cdr body)
				     (break "body ~a should have only one entry" body)
				     (emit (car body))))))))
	      (def (destructuring-bind (name lambda-list &rest body) (cdr code)
		     (multiple-value-bind (req-param opt-param res-param
						     key-param other-key-p aux-param key-exist-p)
			 (parse-ordinary-lambda-list lambda-list)
		       (declare (ignorable req-param opt-param res-param
					   key-param other-key-p aux-param key-exist-p))
		       (with-output-to-string (s)
			 (format s "def ~a~a:~%"
				 name
				 (emit `(paren ,@(append req-param
							 (loop for e in key-param collect 
							      (destructuring-bind ((keyword-name name) init suppliedp)
								  e
								(declare (ignorable keyword-name suppliedp))
								`(= ,name ,init)))))))
			 (format s "~a" (emit `(do ,@body)))))))
	      (= (destructuring-bind (a b) (cdr code)
		   (format nil "~a=~a" (emit a) (emit b))))
	      (in (destructuring-bind (a b) (cdr code)
		   (format nil "(~a in ~a)" (emit a) (emit b))))
	      (setf (let ((args (cdr code)))
		      (format nil "~a"
			      (emit `(do0 
				      ,@(loop for i below (length args) by 2 collect
					     (let ((a (elt args i))
						   (b (elt args (+ 1 i))))
					       `(= ,a ,b))))))))
	      (aref (destructuring-bind (name &rest indices) (cdr code)
		      (format nil "~a[~{~a~^,~}]" (emit name) (mapcar #'emit indices))))
	      (slice (let ((args (cdr code)))
		       (if (null args)
			   (format nil ":")
			   (format nil "~{~a~^:~}" (mapcar #'emit args)))))
	      (dot (let ((args (cdr code)))
		   (format nil "~{~a~^.~}" (mapcar #'emit args))))
	      (+ (let ((args (cdr code)))
		   (format nil "(~{(~a)~^+~})" (mapcar #'emit args))))
	      (- (let ((args (cdr code)))
		   (format nil "(~{(~a)~^-~})" (mapcar #'emit args))))
	      (* (let ((args (cdr code)))
		   (format nil "(~{(~a)~^*~})" (mapcar #'emit args))))
	      (== (let ((args (cdr code)))
		    (format nil "(~{(~a)~^==~})" (mapcar #'emit args))))
	      (!= (let ((args (cdr code)))
		   (format nil "(~{(~a)~^!=~})" (mapcar #'emit args))))
	      (< (let ((args (cdr code)))
		   (format nil "(~{(~a)~^<~})" (mapcar #'emit args))))
	      (<= (let ((args (cdr code)))
		   (format nil "(~{(~a)~^<=~})" (mapcar #'emit args))))
	      (/ (let ((args (cdr code)))
		   (format nil "((~a)/(~a))"
			   (emit (first args))
			   (emit (second args)))))
	      (** (let ((args (cdr code)))
		   (format nil "((~a)**(~a))"
			   (emit (first args))
			   (emit (second args)))))
	      (// (let ((args (cdr code)))
		   (format nil "((~a)//(~a))"
			   (emit (first args))
			   (emit (second args)))))
	      (% (let ((args (cdr code)))
		   (format nil "((~a)%(~a))"
			   (emit (first args))
			   (emit (second args)))))
	      (and (let ((args (cdr code)))
		     (format nil "(~{(~a)~^ and ~})" (mapcar #'emit args))))
	      (or (let ((args (cdr code)))
		    (format nil "(~{(~a)~^ or ~})" (mapcar #'emit args))))
	      (string (format nil "\"~a\"" (cadr code)))
	      (return_ (format nil "return ~a" (emit (caadr code))))
	      (return (let ((args (cdr code)))
			(format nil "~a" (emit `(return_ ,args)))))
	      (for (destructuring-bind ((vs ls) &rest body) (cdr code)
		     (with-output-to-string (s)
		       ;(format s "~a" (emit '(indent)))
		       (format s "for ~a in ~a:~%"
			       (emit vs)
			       (emit ls))
		       (format s "~a" (emit `(do ,@body))))))

	      (if (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
		    (with-output-to-string (s)
		      (format s "if ( ~a ):~%~a"
			      (emit condition)
			      (emit `(do ,true-statement)))
		      (when false-statement
			(format s "~a:~%~a"
				(emit `(indent "else"))
				(emit `(do ,false-statement)))))))
	      (import (destructuring-bind (args) (cdr code)
			(if (listp args)
			    (format nil "import ~a as ~a~%" (second args) (first args))
			    (format nil "import ~a~%" args))))
	      (imports (destructuring-bind (args) (cdr code)
			 (format nil "~{~a~}" (mapcar #'(lambda (x) (emit `(import ,x))) args))))
	      (with (destructuring-bind (form &rest body) (cdr code)
		      (with-output-to-string (s)
		       (format s "~a~a:~%~a"
			       (emit "with ")
			       (emit form)
			       (emit `(do ,@body))))))
	      (try (destructuring-bind (prog &rest exceptions) (cdr code)
		     (with-output-to-string (s)
		       (format s "~a:~%~a"
			       (emit "try")
			       (emit `(do ,prog)))
		       (loop for e in exceptions do
			    (destructuring-bind (form &rest body) e
			      (format s "~a~%"
				      (emit `(indent ,(format nil "except ~a:" (emit form)))))
			      (format s "~a" (emit `(do ,@body)))))))
	       
	       #+nil (let ((body (cdr code)))
		     (with-output-to-string (s)
		       (format s "~a:~%" (emit "try"))
		       (format s "~a" (emit `(do ,@body)))
		       (format s "~a~%~a"
			       (emit "except Exception as e:")
			       (emit `(do "print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)"))))))
	      (t (destructuring-bind (name &rest args) code
		   (let* ((positional (loop for i below (length args) until (keywordp (elt args i)) collect
					   (elt args i)))
			  (plist (subseq args (length positional)))
			  (props (loop for e in plist by #'cddr collect e)))
		     (format nil "~a~a" name
			     (emit `(paren ,@(append
					      positional
					      (loop for e in props collect
						   `(= ,(format nil "~a" e) ,(getf plist e)))))))))))
	    (cond
	      ((or (symbolp code)
		   (stringp code)) ;; print variable
	       (format nil "~a" code))
	      ((numberp code) ;; print constants
	       (cond ((integerp code) (format str "~a" code))
		     ((floatp code)
		      (format str "(~a)" (print-sufficient-digits-f64 code)))
		     ((complexp code)
		      (format str "((~a) + 1j * (~a))"
			      (print-sufficient-digits-f64 (realpart code))
			      (print-sufficient-digits-f64 (imagpart code))))))))
	"")))

