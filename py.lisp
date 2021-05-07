;(ql:quickload "optima")
;(ql:quickload "alexandria")

(in-package :cl-py-generator)
(setf (readtable-case *readtable*) :invert)

(defparameter *file-hashes* (make-hash-table))

(defun write-notebook (&key nb-file nb-code)
  "write python jupyter notebook"
  (let ((tmp (format nil "~a.tmp" nb-file)))
    (with-output-to-file (s tmp :if-exists :supersede
				:if-does-not-exist :create)
      (format s "~a~%"
	      (jonathan:to-json
	       `(:|cells|
		  ,(loop for e in nb-code
			 collect
			 (destructuring-bind (name &rest rest) e
			   (case name
			     (`markdown `(:cell_type "markdown"
					  :metadata :empty
					  :source
					  ,(loop for p in rest
						 collect
						 (format nil "~a~c" p #\Newline))))
			     (`python `(:cell_type "code"
					:metadata :empty
					:execution_count :null
					:outputs ()
					:source
					,(loop for p in rest
					       appending
					       (let ((tempfn "/dev/shm/cell"))
						 (write-source tempfn p)
						 (with-open-file (stream (format nil "~a.py" tempfn))
						   (loop for line = (read-line stream nil)
							 while line
							 collect
							 (format nil "~a~c" line #\Newline)))))))
			     )))
		  :|metadata| (:|kernelspec| (:|display_name| "Python 3"
					       :|language| "python"
					      :|name| "python3"))
		 :|nbformat| 4
		  :|nbformat_minor| 2))))
    (sb-ext:run-program "/usr/bin/jq" `("-M" "." ,tmp)
			:output nb-file
			:if-output-exists :supersede)
    (delete-file tmp)))

(defun write-source (name code &optional (dir (user-homedir-pathname))
				 ignore-hash)
  (let* ((fn (merge-pathnames (format nil "~a.py" name)
			      dir))
	(code-str (emit-py
		   :clear-env t
		   :code code))
	(fn-hash (sxhash fn))
	 (code-hash (sxhash code-str)))
    (multiple-value-bind (old-code-hash exists) (gethash fn-hash *file-hashes*)
     (when (or (not exists) ignore-hash (/= code-hash old-code-hash))
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
       #+sbcl (sb-ext:run-program "/usr/bin/yapf" (list "-i" (namestring fn)))))))

(defun print-sufficient-digits-f64 (f)
  "print a double floating point number as a string with a given nr. of                                                                                                                                             
  digits. parse it again and increase nr. of digits until the same bit                                                                                                                                              
  pattern."

  (let* ((a f)
         (digits 1)
         (b (- a 1)))
    (unless (= a 0)
      (loop while (< 1d-12
		     (/ (abs (- a b))
		       (abs a))
		    ) do
          (setf b (read-from-string (format nil "~,vG" digits a)))
           (incf digits)
	   ))
    (substitute #\e #\d (format nil "~,vG" digits a))))


;(print-sufficient-digits-f64 1d0)


(defparameter *env-functions* nil)
(defparameter *env-macros* nil)

#+nil
(defun dotry (code)
  `(try (do0
	 ,code)
	("Exception as exc"
	 ,(lprint `(exc))
	 pass)))

(defun emit-py (&key code (str nil) (clear-env nil) (level 0))
  ;(format t "emit ~a ~a~%" level code)
  (when clear-env
    (setf *env-functions* nil
	  *env-macros* nil))
  (flet ((emit (code &optional (dl 0))
	   (emit-py :code code :clear-env nil :level (+ dl level))))
    (format nil "emit-py ~a" level)
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
	      (curly (let ((args (cdr code)))
		      (format nil "{~{~a~^, ~}}" (mapcar #'emit args))))
              (dict (let* ((args (cdr code)))
		      (let ((str (with-output-to-string (s)
				   (loop for (e f) in args
				      do
					(format s "(~a):(~a)," (emit e) (emit f))))))
			(format nil "{~a}" ;; remove trailing comma
				(subseq str 0 (- (length str) 1))))))
	      (dictionary (let* ((args (cdr code)))
			    (format nil "dict~a"
				    (emit `(paren ,@(loop for (e f) on args by #'cddr
							  collect
							  `(= ,e ,f)))))))
	      (indent (format nil "~{~a~}~a"
			      (loop for i below level collect "    ")
			      (emit (cadr code))))
	      (do (with-output-to-string (s)
		    (format s "~{~&~a~}" (mapcar #'(lambda (x) (emit `(indent ,x) 1)) (cdr code)))))
	      (class (destructuring-bind (name parents &rest body) (cdr code)
		       (format nil "class ~a~a:~%~a"
			       name
			       (emit `(paren ,@parents))
			       (emit `(do ,@body)))))
	      (do0 (with-output-to-string (s)
		     (format s "~&~a~{~&~a~}"
			     (emit (cadr code))
			     (mapcar #'(lambda (x) (emit `(indent ,x) 0)) (cddr code)))))
	      (space (with-output-to-string (s)
		     (format s "~{~a~^ ~}"
			     (mapcar #'(lambda (x) (emit x)) (cdr code)))))
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
								(if init
								    `(= ,(emit name) ,init)
								    `(= ,(emit name) "None")))))))
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
				 (emit `(paren
					 ,@(append (mapcar #'emit req-param)
						   (loop for e in key-param collect 
							(destructuring-bind ((keyword-name name) init suppliedp)
							    e
							  (declare (ignorable keyword-name suppliedp))
							  (if init
							      `(= ,name ,init)
							      `(= ,name "None"))))))))
			 (format s "~a" (emit `(do ,@body)))))))
	      (= (destructuring-bind (a b) (cdr code)
		   (format nil "~a=~a" (emit a) (emit b))))
	      (in (destructuring-bind (a b) (cdr code)
		    (format nil "(~a in ~a)" (emit a) (emit b))))
	      (is (destructuring-bind (a b) (cdr code)
		    (format nil "(~a is ~a)" (emit a) (emit b))))
	      (as (destructuring-bind (a b) (cdr code)
		   (format nil "~a as ~a" (emit a) (emit b))))
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
	      (<< (let ((args (cdr code)))
		    (format nil "(~{(~a)~^<<~})" (mapcar #'emit args))))
	      (!= (let ((args (cdr code)))
		   (format nil "(~{(~a)~^!=~})" (mapcar #'emit args))))
	      (< (let ((args (cdr code)))
		   (format nil "(~{(~a)~^<~})" (mapcar #'emit args))))
	      (<= (let ((args (cdr code)))
		    (format nil "(~{(~a)~^<=~})" (mapcar #'emit args))))
	      (>> (let ((args (cdr code)))
		   (format nil "(~{(~a)~^>>~})" (mapcar #'emit args))))
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
	      (& (let ((args (cdr code)))
		   (format nil "(~{(~a)~^ & ~})" (mapcar #'emit args))))
	      (logand (let ((args (cdr code)))
			(format nil "(~{(~a)~^ & ~})" (mapcar #'emit args))))
	      (logxor (let ((args (cdr code)))
		   (format nil "(~{(~a)~^ ^ ~})" (mapcar #'emit args))))
	      (|\|| (let ((args (cdr code)))
		      (format nil "(~{(~a)~^ | ~})" (mapcar #'emit args))))
	      (^ (let ((args (cdr code)))
		      (format nil "(~{(~a)~^ ^ ~})" (mapcar #'emit args))))
	      (logior (let ((args (cdr code)))
		     (format nil "(~{(~a)~^ | ~})" (mapcar #'emit args))))
	      (or (let ((args (cdr code)))
		    (format nil "(~{(~a)~^ or ~})" (mapcar #'emit args))))
	      (comment (format nil "# ~a~%" (cadr code)))
	      (comments (let ((args (cdr code)))
			  (format nil "~{# ~a~%~}" args)))
	      (string (format nil "\"~a\"" (cadr code)))
	      (string-b (format nil "b\"~a\"" (cadr code)))
	      (string3 (format nil "\"\"\"~a\"\"\"" (cadr code)))
	      (rstring3 (format nil "r\"\"\"~a\"\"\"" (cadr code)))
	      (return_ (format nil "return ~a" (emit (caadr code))))
	      (return (let ((args (cdr code)))
			(format nil "~a" (emit `(return_ ,args)))))
	      (assert (let ((args (cdr code)))
			(format nil "assert ~a" (emit `(ntuple ,@args)))))
	      (for (destructuring-bind ((vs ls) &rest body) (cdr code)
		     (with-output-to-string (s)
		       ;(format s "~a" (emit '(indent)))
		       (format s "for ~a in ~a:~%"
			       (emit vs)
			       (emit ls))
		       (format s "~a" (emit `(do ,@body))))))
	      (for-generator
	       (destructuring-bind ((vs ls) expr) (cdr code)
		     (format nil "~a for ~a in ~a"
			     (emit expr)
			     (emit vs)
			     (emit ls))))
	      (while (destructuring-bind (vs &rest body) (cdr code)
		     (with-output-to-string (s)
		       (format s "while ~a:~%"
			       (emit `(paren ,vs)))
		       (format s "~a" (emit `(do ,@body))))))

	      (if (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
		    (with-output-to-string (s)
		      (format s "if ( ~a ):~%~a"
			      (emit condition)
			      (emit `(do ,true-statement)))
		      (when false-statement
			(format s "~&~a:~%~a"
				(emit `(indent "else"))
				(emit `(do ,false-statement)))))))
	      (when (destructuring-bind (condition &rest forms) (cdr code)
                      (emit `(if ,condition
                                 (do0
                                  ,@forms)))))
              (unless (destructuring-bind (condition &rest forms) (cdr code)
                        (emit `(if (not ,condition)
                                   (do0
                                    ,@forms)))))
	      (import-from (destructuring-bind (module &rest rest) (cdr code)
			     (format nil "from ~a import ~{~a~^, ~}" module
				     rest)))
	      (imports-from (destructuring-bind (&rest module-defs) (cdr code)
			      (with-output-to-string (s)
				(loop for e in module-defs
				      do
				      (format s "~a~%" (emit `(import-from ,@e)))))))
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
		       (format s "~&~a:~%~a"
			       (emit "try")
			       (emit `(do ,prog)))
		       (loop for e in exceptions do
			    (destructuring-bind (form &rest body) e
			      (if (member form '(else finally))
				  (format s "~&~a~%"
					  (emit `(indent ,(format nil "~a:" form))))
				  (format s "~&~a~%"
				       (emit `(indent ,(format nil "except ~a:" (emit form))))))
			      (format s "~a" (emit `(do ,@body)))))))
	       
	       #+nil (let ((body (cdr code)))
		     (with-output-to-string (s)
		       (format s "~a:~%" (emit "try"))
		       (format s "~a" (emit `(do ,@body)))
		       (format s "~a~%~a"
			       (emit "except Exception as e:")
			       (emit `(do "print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)"))))))
	      (t (destructuring-bind (name &rest args) code
		   
		   (if (listp name)
		       ;; lambda call and similar complex constructs
		       (format nil "(~a)(~a)" (emit name) (if args
							      (emit `(paren ,@args))
							      ""))
		       #+nil(if (eq 'lambda (car name))
			   (format nil "(~a)(~a)" (emit name) (emit `(paren ,@args)))
			   (break "error: unknown call"))
		       ;; function call
		       (let* ((positional (loop for i below (length args) until (keywordp (elt args i)) collect
					       (elt args i)))
			      (plist (subseq args (length positional)))
			      (props (loop for e in plist by #'cddr collect e)))
			 (format nil "~a~a" name
				 (emit `(paren ,@(append
						  positional
						  (loop for e in props collect
						       `(= ,(format nil "~a" e) ,(getf plist e))))))))))))
	    (cond
	      ((symbolp code) ;; print variable
	       (format nil "~a" code))
	      ((stringp code)
		(substitute #\: #\- (format nil "~a" code)))
	      ((numberp code) ;; print constants
	       (cond ((integerp code) (format str "~a" code))
		     ((floatp code)
		      (format str "(~a)" (print-sufficient-digits-f64 code)))
		     ((complexp code)
		      (format str "((~a) + 1j * (~a))"
			      (print-sufficient-digits-f64 (realpart code))
			      (print-sufficient-digits-f64 (imagpart code))))))))
	"")))

