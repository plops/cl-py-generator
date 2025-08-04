					;(ql:quickload "optima")
					;(ql:quickload "alexandria")

(in-package :cl-py-generator)
(defun write-notebook (&key nb-file nb-code)
	"Writes a notebook to a file.

    The notebook is written in JSON format and formatted using the jq tool.
    
	Args:
		nb-file (string): The path to the notebook file.
		nb-code (string): The code to be written to the notebook.

	Returns:
		None"
  (let ((tmp (format nil "~a.tmp" nb-file)))
    (with-output-to-file (s tmp :if-exists :supersede
			    :if-does-not-exist :create)
      (format s "~a~%"
	      (jonathan:to-json
	       `( :cells
		  ;:|cells|
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
							  (let ((tempfn #+sbcl "/dev/shm/cell"
									#+ecl (format nil "~a_tmp_cell" nb-file)))
							    (write-source tempfn p)
							    (with-open-file (stream (format nil "~a.py" tempfn))
							      (loop for line = (read-line stream nil)
								    while line
								    collect
								    (format nil "~a~c" line #\Newline)))))))
			     )))
		  #+nil (:|metadata| (:|kernelspec| (:|display_name| "Python 3"
						      :|language| "python"
						      :|name| "python3"))
			  :|nbformat| 4
			  :|nbformat_minor| 2)


		  :metadata (:kernelspec (:display_name "Python 3"
							:language "python"
							:name "python3"))
		  :nbformat 4
		  :nbformat_minor 2))))
    #+nil
    (sb-ext:run-program "/usr/bin/python3" `("-mjson.tool" ,nb-file))
    #-sbcl
    (external-program:run
     "/usr/bin/jq"
     `("-M" "." ,tmp)
     :output nb-file
     :if-output-exists :supersede
     )
    #+sbcl
    (sb-ext:run-program "/usr/bin/jq" `("-M" "." ,tmp)
			:output nb-file
			:if-output-exists :supersede)
    (delete-file tmp)))

(setf (readtable-case *readtable*) :invert)

(defparameter *warn-breaking* t)
(defparameter *file-hashes* (make-hash-table))



(defun consume-declare (body)
  "Take a list of instructions from `body`, parse type declarations,
return the `body` without them and a hash table with an environment. The
entry `return-values` contains a list of return values. Currently supports `type`, `values`.

Parameters:
- `body` (list): The list of instructions to process.

Returns:
- `new-body` (list): The modified `body` without type declarations.
- `env` (hash-table): The hash table representing the environment with captured variables and return values."
  (let ((env (make-hash-table))
	(looking-p t)
	(new-body nil))
    (loop for e in body do
      (if looking-p
	  (if (listp e)
	      (if (eq (car e) 'declare)
		  (loop for declaration in (cdr e) do
		    (when (eq (first declaration) 'type)
		      (destructuring-bind (symb type &rest vars) declaration
			(declare (ignorable symb))
			(loop for var in vars do
			  (setf (gethash var env) type))))
		    (when (eq (first declaration) 'capture)
		      (destructuring-bind (symb &rest vars) declaration
			(declare (ignorable symb))
			(loop for var in vars do
			  (push var captures))))

		    
		    (when (eq (first declaration) 'values)
		      (destructuring-bind (symb &rest types-opt) declaration
			(declare (ignorable symb))
			;; if no values specified parse-defun will emit void
			;; if (values :constructor) then nothing will be emitted
			(let ((types nil))
			  ;; only collect types until occurrance of &optional
			  (loop for type in types-opt do
			    (unless (eq #\& (aref (format nil "~a" type) 0))
			      (push type types)))
			  (setf (gethash 'return-values env) (reverse types))))))
		  (progn
		    (push e new-body)
		    (setf looking-p nil)))
	      (progn
		(setf looking-p nil)
		(push e new-body)))
	  (push e new-body)))
    (values (reverse new-body) env)))

(defun parse-defun (code emit )
  "Parse a defun expression and generate Python code.
    
    This function parses a DEFUN s-expression form and emits Python code. Optionally, it can insert type hints for parameters and the return value.

  Args:
    code: The defun expression to parse.
    emit: The function used to emit Python code for forms in the function block.

  Returns:
    The generated Python code as a string.

  Supported grammar: defun function-name lambda-list [declaration*] form*"
  (destructuring-bind (name lambda-list &rest body) (cdr code)
    (multiple-value-bind (body env) (consume-declare body)
      (multiple-value-bind (req-param opt-param res-param
			    key-param other-key-p
			    aux-param key-exist-p)
	  (parse-ordinary-lambda-list lambda-list)
	(declare (ignorable req-param opt-param res-param
			    key-param other-key-p aux-param key-exist-p))
	(with-output-to-string (s)
	  (format s "def ~a~a~@[->~a~]:~%"
		  name
		  ;; 8 positional parameters, followed by key parameters
		  (funcall emit `(paren
				  ;; positional
				  ,@(loop for p in req-param collect
							     (format nil "~a~@[: ~a~]"
								     
								     p
								     (let ((type (gethash p env)))
								       (when type
									 (funcall emit type))
								       )))
				  ;; key parameters
				  ;; http://www.crategus.com/books/alexandria/pages/alexandria.0.dev_fun_parse-ordinary-lambda-list.html
				  ;; default arguments with type hints in python: def foo(opts: dict = {}):
				  ;; https://stackoverflow.com/questions/38727520/how-do-i-add-default-parameters-to-functions-when-using-type-hinting
				  ;; syntax of keyword arguments with type hint
				  ;; <var>: <type> = <default-value>
				  
				  ,@(loop for ((keyword-name name) init supplied-p) in key-param
					  collect
					  (progn
					    (format nil "~a~a ~@[~a~]"
						    
						    name
						    (let ((type (gethash name env)))
						      (if type
							  (format nil ": ~a" (funcall emit type))
							  "" ;; can't find type for keyword parameter
							  ))
						    (format nil "= ~a" (funcall emit init)))))
				  ))
		  #+nil (emit `(paren
				,@(append (mapcar #'emit req-param)
					  (loop for e in key-param collect
								   (destructuring-bind ((keyword-name name) init suppliedp)
								       e
								     (declare (ignorable keyword-name suppliedp))
								     (if init
									 `(= ,name ,init)
									 `(= ,name "None")))))))
		  ;; return value
		  (let ((r (gethash 'return-values env)))
		    (if (< 1 (length r))
			(progn
			  ;; https://stackoverflow.com/questions/40181344/how-to-annotate-types-of-multiple-return-values
			  ;; python 3.9 supports tuple[bool, str],  previous version Tuple[bool, str]
			  (break "multiple return values unsupported: ~a"
				 r))
			(if (car r)
			    (case (car r)
			      (:constructor "") ;; (values :constructor) will not print anything
			      (t (car r)))
			    nil		;"void"
			    ))))
	  (format s "~a" (funcall emit `(do ,@body))))))))


(defun write-source (name code &optional (dir (user-homedir-pathname))
								 ignore-hash)
		"Writes the Python source code to a file.

		Args:
				name (string): The name of the file.
				code (s-expr): The Python source code.
				dir (pathname): The directory where the file will be saved. Defaults to the user's home directory.
				ignore-hash (boolean): If true, ignores the hash check and always writes the code to the file.

		Returns:
				None"

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
				(sb-ext:run-program "/home/martin/micromamba/bin/ruff" (list "format" (namestring fn)))
				#+nil (sb-ext:run-program "/usr/bin/yapf" (list "-i" (namestring fn)))
				#+nil
				(progn
					;; python3 -m pip install --user black
					;; should i use --fast option?xs
				  (sb-ext:run-program "/home/martin/.local/bin/black"
							(list "--fast"
										(namestring  fn))))))))

(defun print-sufficient-digits-f64 (f)
	"Prints a double floating point number as a string with a given number of digits.
	 Parses the string representation and increases the number of digits until the same bit pattern is obtained.

	 Args:
		 f: The double floating point number to be printed.

	 Returns:
		 The string representation of the number with sufficient digits."
	(let* ((a f)
				 (digits 1)
				 (b (- a 1)))
		(unless (= a 0)
			(loop while (< 1d-12
										 (/ (abs (- a b))
												(abs a)))
						do
						(setf b (read-from-string (format nil "~,vG" digits a)))
						(incf digits)))
		(substitute #\e #\d (format nil "~,vG" digits a))))


					;(print-sufficient-digits-f64 1d0)


(defparameter *env-functions* nil "docstring")
(defparameter *env-macros* nil)


(defun emit-py (&key code (str nil) (clear-env nil) (level 0))
	"Emit Python code based on the given parameters.

	Args:
		code (s-expr): The code to emit.
		str: A string to emit into.
		clear-env: Whether to clear the environment.
		level: The indentation level.

	Returns:
		The emitted Python code as a string."
					;(format t "emit ~a ~a~%" level code)
  (when clear-env
    (setf *env-functions* nil
	  *env-macros* nil))
  (flet ((emit (code &optional (dl 0))
	   (emit-py :code code :clear-env nil :level (+ dl level))))
					;(format nil "emit-py ~a" level)
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
	      (cell (with-output-to-string (s)
		      (format s "~a~%"
			      (emit `(do0 (comments "export")
					  ,@(cdr code))))))
	      (export (with-output-to-string (s)
			(format s "~a~%"
				(emit `(do0 (comments "|export")
					    ,@(cdr code))))))
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
	      (def (parse-defun code #'emit)
	       #+nil (destructuring-bind (name lambda-list &rest body) (cdr code)
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
	      (incf (destructuring-bind (target &optional (val 1)) (cdr code)
		      (format nil "~a += ~a" (emit target) (emit val))))
	      (decf (destructuring-bind (target &optional (val 1)) (cdr code)
		      (format nil "~a -= ~a"
			      (emit target)
			      (emit val))))
	      (aref (destructuring-bind (name &rest indices) (cdr code)
		      (format nil "~a[~{~a~^,~}]" (emit name) (mapcar #'emit indices))))
	      (slice (let ((args (cdr code)))
		       (if (null args)
			   (format nil ":")
			   (format nil "~{~a~^:~}" (mapcar #'emit args)))))
	      (dot (let ((args (cdr code)))
		     ;; don't print . for nil arguments
		     (format nil "~{~a~^.~}" (mapcar #'emit (remove-if #'null args)))))
	      (+ (let ((args (cdr code)))
		   (format nil "(~{(~a)~^+~})" (mapcar #'emit args))))
	      (- (let ((args (cdr code)))
		   (format nil "(~{(~a)~^-~})" (mapcar #'emit args))))
	      (* (let ((args (cdr code)))
		   (format nil "(~{(~a)~^*~})" (mapcar #'emit args))))
	      (@ (let ((args (cdr code)))
		   (format nil "(~{(~a)~^@~})" (mapcar #'emit args))))
	      (== (let ((args (cdr code)))
		    (format nil "(~{(~a)~^==~})" (mapcar #'emit args))))
	      (<< (let ((args (cdr code)))
		    (format nil "(~{(~a)~^<<~})" (mapcar #'emit args))))
	      (!= (let ((args (cdr code)))
		    (format nil "(~{(~a)~^!=~})" (mapcar #'emit args))))
	      (< (let ((args (cdr code)))
		   (format nil "(~{(~a)~^<~})" (mapcar #'emit args))))
	      (> (let ((args (cdr code)))
		   (format nil "(~{(~a)~^>~})" (mapcar #'emit args))))
	      (<= (let ((args (cdr code)))
		    (format nil "(~{(~a)~^<=~})" (mapcar #'emit args))))
	      (>= (let ((args (cdr code)))
		    (format nil "(~{(~a)~^>=~})" (mapcar #'emit args))))
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
			  (format nil "~{# ~a~%~}" (mapcar #'(lambda (arg)
							      (cl-ppcre:regex-replace-all
							       "\\n"
							       arg
							       (format nil "~%# ")))
							   args))))
	      (symbol (substitute #\: #\- (format nil "~a" (cadr code))))
	      (string (format nil "\"~a\"" (cadr code)))
	      (string-b (format nil "b\"~a\"" (cadr code)))
	      (fstring (format nil "f\"~a\"" (cadr code)))
	      (fstring3 (format nil "f\"\"\"~a\"\"\"" (cadr code)))
	      (string3 (format nil "\"\"\"~a\"\"\"" (cadr code)))
	      (rstring3 (format nil "r\"\"\"~a\"\"\"" (cadr code)))
	      (return_ (format nil "return ~a" (emit (caadr code))))
	      (return (let ((args (cdr code)))
			(format nil "~a" (emit `(return_ ,args)))))
	      #+nil(assert (let ((args (cdr code)))
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
	      (cond (destructuring-bind (&rest clauses) (cdr code)
		      ;; if <cond1> : <code1> elif <cond2> : <code2> else <code3>
		      (with-output-to-string (s)
			(loop for clause in clauses and i from 0
			      do
			      (destructuring-bind (condition &rest statements) clause
				(format s "~&~a:~%~a"
					(cond ((and (eq condition 't) (eq i 0))
					       ;; this special case may happen when you comment out all but the last cond clauses
					       (format nil "if ( True )"))
					      ((eq i 0) (format nil "if ( ~a )" (emit condition)))
					      ((eq condition 't) (emit `(indent "else")))
					      (t (emit `(indent ,(format nil "elif ( ~a )" (emit condition)))))
					      )
					(emit `(do ,@statements)))))
			)))
	      (? (destructuring-bind (condition true-statement false-statement)
		     (cdr code)
		   (format nil "(~a) if (~a) else (~a)"

			   (emit true-statement)
			   (emit condition)
			   (emit false-statement))))
	      (when (destructuring-bind (condition &rest forms) (cdr code)
                      (emit `(if ,condition
                                 (do0
                                  ,@forms)))))
              (unless (destructuring-bind (condition &rest forms) (cdr code)
                        (emit `(if (not ,condition)
                                   (do0
                                    ,@forms)))))
	      (import-from (destructuring-bind (module &rest rest) (cdr code)
			     (format nil "from ~a import ~{~a~^, ~}"
				     (emit module)
				     (mapcar #'emit rest))))
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
			 (format nil "~{~a~}" (append (list (emit `(import ,(first args))))
						      (mapcar #'(lambda (x) (emit `(indent (import ,x))))
							      (rest args))))))
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
	       #+nil (progn
		       (when *warn-breaking*
			 (format t "~&BREAKING CHANGE ~a is printed as string (used to be symbol, please use (symbol <code>) from now on). I seldomly used this for (\"list\" ...) ~%" code))
		       )
	       code ;(substitute #\: #\- (format nil "~a" code))
	       )
	      ((numberp code) ;; print constants
	       (cond ((integerp code) (format str "~a" code))
		     ((floatp code)
		      (format str "(~a)" (print-sufficient-digits-f64 code)))
		     ((complexp code)
		      (format str "((~a) + 1j * (~a))"
			      (print-sufficient-digits-f64 (realpart code))
			      (print-sufficient-digits-f64 (imagpart code))))))))
	"")))

