					;(ql:quickload "optima")
					;(ql:quickload "alexandria")

(in-package :cl-py-generator)

(defun notebook-cell-source (cell)
  "Convert one entry of the `nb-code' list of `write-notebook' into the list of
source lines of a notebook cell."
  (destructuring-bind (name &rest rest) cell
    ;; NOTE: the clause keys must be plain symbols. Writing `markdown here
    ;; reads as (quasiquote markdown), i.e. as a key *list* of two keys, so
    ;; every clause would additionally match the symbol QUASIQUOTE.
    (case name
      (markdown (loop for p in rest
		      collect (format nil "~a~c" p #\Newline)))
      (python (loop for p in rest
		    appending
		    (let ((tempfn (namestring
				   (merge-pathnames "cl-py-generator-cell"
						    (uiop:temporary-directory)))))
		      (write-source tempfn p)
		      (with-open-file (stream (format nil "~a.py" tempfn))
			(loop for line = (read-line stream nil)
			      while line
			      collect (format nil "~a~c" line #\Newline))))))
      (t (error "unsupported notebook cell type ~a, expected markdown or python" name)))))

(defun write-notebook (&key nb-file nb-code)
	"Writes a notebook to a file.

    The notebook is written in JSON format and pretty printed with the jq tool
    (if jq is available in PATH).

	Args:
		nb-file (string): The path to the notebook file, including the `.ipynb' suffix.
		nb-code (list): A list of cells. Every cell is a list that starts with
			the symbol `markdown' (followed by strings) or `python' (followed by
			s-expressions that are transpiled to python).

	Returns:
		The pathname of the notebook."
  ;; NOTE: the json keys are given as strings and the structure is built as
  ;; alists on purpose. jonathan prints keyword keys with the current
  ;; readtable-case, and it folds constant keys at compile time, so with the
  ;; :invert readtable of this library the top level keys used to end up as
  ;; "CELLS"/"NBFORMAT", which nbformat/jupyter rejects.
  (let ((tmp (format nil "~a.tmp" nb-file)))
    (with-output-to-file (s tmp :if-exists :supersede
			    :if-does-not-exist :create)
      (format s "~a~%"
	      (jonathan:to-json
	       (list (cons "cells"
			   (loop for e in nb-code
				 collect
				 (let ((source (notebook-cell-source e)))
				   (if (eq 'markdown (first e))
				       (list (cons "cell_type" "markdown")
					     (cons "metadata" :empty)
					     (cons "source" source))
				       (list (cons "cell_type" "code")
					     (cons "metadata" :empty)
					     (cons "execution_count" :null)
					     (cons "outputs" nil)
					     (cons "source" source))))))
		     (cons "metadata"
			   (list (cons "kernelspec"
				       (list (cons "display_name" "Python 3")
					     (cons "language" "python")
					     (cons "name" "python3")))))
		     (cons "nbformat" 4)
		     (cons "nbformat_minor" 2))
	       :from :alist)))
    ;; pretty print the json with jq. if jq is not installed, keep the
    ;; (valid, but single line) json that jonathan produced.
    (if (program-in-path-p "jq")
	(progn
	  (uiop:run-program (list "jq" "-M" "." (namestring tmp))
			    :output (pathname nb-file)
			    :if-output-exists :supersede)
	  (delete-file tmp)
	  (pathname nb-file))
	(progn
	  (warn "jq was not found in PATH, the notebook ~a is not pretty printed." nb-file)
	  (rename-file tmp nb-file)))))

(setf (readtable-case *readtable*) :invert)

(defparameter *warn-breaking* t)
(defparameter *file-hashes* (make-hash-table))



(defun consume-declare (body)
  "Take a list of instructions from `body`, parse type declarations,
return the `body` without them and a hash table with an environment. The
entry `return-values` contains a list of return values. Currently supports `type`, `values` and `capture`.

Parameters:
- `body` (list): The list of instructions to process.

Returns:
- `new-body` (list): The modified `body` without type declarations.
- `env` (hash-table): The hash table representing the environment with captured variables and return values."
  (let ((env (make-hash-table))
	(captures nil)
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
    (when captures
      (setf (gethash 'captures env) (reverse captures)))
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
			  (error "multiple return values unsupported: ~a"
				 r))
			(if (car r)
			    (case (car r)
			      (:constructor "") ;; (values :constructor) will not print anything
			      (t (car r)))
			    nil		;"void"
			    ))))
	  (format s "~a" (funcall emit `(do ,@body))))))))


(defparameter *python-format-command* nil
	"Command that is used to pretty print the generated python files.

	 A list of strings; the name of the file to format is appended to it, e.g.
	 (list \"ruff\" \"format\").  NIL means: auto-detect (`ruff' from PATH, then
	 `uvx ruff format').  :NONE disables formatting.")

(defparameter *python-format-command-detected* nil
	"Cached result of the formatter auto detection: a command list or :NONE.")

(defparameter *python-format-warned* nil
	"Remembers whether the 'no formatter found' warning was already issued.")

(defun program-in-path-p (program)
	"Return T if PROGRAM can be found in PATH."
	(multiple-value-bind (out err code)
			(uiop:run-program (list "sh" "-c" (format nil "command -v ~a" program))
												:output nil :error-output nil :ignore-error-status t)
		(declare (ignore out err))
		(eql 0 code)))

(defun python-format-command ()
	"Return the command (a list of strings) that formats a python file when the
	 file name is appended, or NIL if formatting is disabled or no formatter is
	 available."
	(cond ((eq :none *python-format-command*) nil)
				(*python-format-command* *python-format-command*)
				(t (unless *python-format-command-detected*
						 (setf *python-format-command-detected*
									 (cond ((program-in-path-p "ruff") (list "ruff" "format"))
												 ((program-in-path-p "uvx") (list "uvx" "ruff" "format"))
												 (t :none))))
					 (if (eq :none *python-format-command-detected*)
							 nil
							 *python-format-command-detected*))))

(defun format-python-file (fn)
	"Run the external python formatter on the file FN.

	 Never signals an error: a missing or failing formatter only leads to a
	 warning, the (unformatted) file stays on disk.  Returns FN."
	(let ((cmd (python-format-command)))
		(cond (cmd
					 (multiple-value-bind (out err code)
							 (uiop:run-program (append cmd (list (namestring fn)))
																 :output nil :error-output :string :ignore-error-status t)
						 (declare (ignore out))
						 (unless (eql 0 code)
							 (warn "~{~a~^ ~} failed with exit code ~a on ~a: ~a" cmd code fn err))))
					;; formatting explicitly switched off by the user
					((eq :none *python-format-command*))
					(t (unless *python-format-warned*
							 (setf *python-format-warned* t)
							 (warn "No python formatter found (tried `ruff' and `uvx'); generated code is written unformatted. Bind cl-py-generator:*python-format-command* to a command list to override."))))
		fn))

(defun write-source (name code &optional (dir (user-homedir-pathname))
								 ignore-hash)
		"Writes the Python source code to a file.

		Note that `.py' is appended to `name', so pass \"code\" and not
		\"code.py\".  A relative `name' is merged with `dir'.

		Args:
				name (string): The name of the file, without the `.py' suffix.
				code (s-expr): The Python source code.
				dir (pathname): The directory where the file will be saved. Defaults to the user's home directory.
				ignore-hash (boolean): If true, ignores the hash check and always writes the code to the file.

		Returns:
				The pathname of the generated file."

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
				(format-python-file fn)))
		fn))

(defun print-sufficient-digits-f64 (f)
	"Print a floating point number as a string that reads back as the very same
	 number (shortest round-trip representation) and that is valid Python.

	 The Common Lisp printer already guarantees read/print consistency for
	 floats (CLHS 22.1.3.1.3).  The only thing that has to be taken care of is
	 the exponent marker: it is `d0'/`s0' etc. whenever the type of the number
	 differs from `*read-default-float-format*', so bind that variable to the
	 type of `f' and normalize any remaining marker to `e'.

	 Args:
		 f: The floating point number to be printed.

	 Returns:
		 The string representation of the number."
	(let* ((*read-default-float-format* (if (typep f 'double-float)
																					'double-float
																					'single-float))
				 (s (prin1-to-string f)))
		(map 'string (lambda (c)
									 (if (member c '(#\d #\D #\f #\F #\s #\S #\l #\L))
											 #\e
											 c))
				 s)))


					;(print-sufficient-digits-f64 1d0)


(defparameter *precedence*
  `((:op (paren paren* dict list tuple curly aref dot) :assoc l)
    (:op (**) :assoc r)
    (:op (unary- unary+ ~) :assoc r)
    (:op (* @ / // %) :assoc l)
    (:op (+ -) :assoc l)
    (:op (<< >>) :assoc l)
    (:op (& logand) :assoc l)
    (:op (^ logxor) :assoc l)
    (:op (|\|| logior) :assoc l)
    (:op (< <= > >= != == in not-in is is-not) :assoc l)
    (:op (not) :assoc r)
    (:op (and) :assoc l)
    (:op (or) :assoc l)
    (:op (? ternary) :assoc r)
    (:op (= setf) :assoc r)))

(defparameter *operators*
  (loop for e in *precedence*
        append (getf e :op)))

(defun lookup-precedence (operator)
  (loop for e in *precedence*
        and e-i from 0
        do
           (destructuring-bind (&key op assoc) e
             (declare (ignore assoc))
             (when (member operator op)
               (return e-i)))))

(defun lookup-associativity (operator)
  (loop for e in *precedence*
        do
           (destructuring-bind (&key op (assoc 'l)) e
             (when (member operator op)
               (return assoc)))))

(defparameter *env-functions* nil "docstring")
(defparameter *env-macros* nil)


(defun emit-py (&key code (str nil) (clear-env nil) (level 0) (omit-redundant-parentheses t))
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
	   (emit-py :code code :clear-env nil :level (+ dl level) :omit-redundant-parentheses omit-redundant-parentheses)))
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
		      (format nil "{~{~{(~a):(~a)~}~^, ~}}" 
			      (loop for (k v) in args
				    collect (list (emit k) (emit v))))
		      ))
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
			       (if (eq 0 (length parents))
				    ""
				    (emit `(paren ,@parents)))
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
					(error "body ~a should have only one entry" body)
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
	      (not-in (destructuring-bind (a b) (cdr code)
		    (format nil "(~a not in ~a)" (emit a) (emit b))))
	      (is (destructuring-bind (a b) (cdr code)
		    (format nil "(~a is ~a)" (emit a) (emit b))))
	      (is-not (destructuring-bind (a b) (cdr code)
		    (format nil "(~a is not ~a)" (emit a) (emit b))))
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
	      (paren*
	       ;; paren* parent-op arg
	       ;; place a pair of parentheses only when needed
	       (if (not omit-redundant-parentheses)
		   (destructuring-bind (parent-op &rest args) (cdr code)
		     (declare (ignore parent-op))
		     (format nil "(~{~a~^, ~})" (mapcar #'emit args)))
		   (progn
		     (unless (eq 3 (length code))
		       (error "paren* expects only two arguments: ~a" code))
		     (destructuring-bind (parent-op arg &rest rest) (cdr code)
		       (declare (ignore rest))
		       (cond
			 ((symbolp arg)
			  (format nil "~a" (emit arg)))
			 ((numberp arg)
			  (if (<= 0 arg)
			      (format nil "~a" (emit arg))
			      (format nil " ~a" (emit arg))))
			 ((stringp arg)
			  (format nil "~a" arg))
			 ((listp arg)
			  ;; a list can be an arbitrary abstract syntax tree of operators
			  (cond
			    ((<= (length arg) 2)
			     ;; two or one elements doesn't need paren
			     (let ((op0 (car arg))
				   (rest (cdr arg)))
			       (assert (or (symbolp op0)
					   (stringp op0)))
			       (assert (listp rest))
			       (emit `(,op0 ,@rest))))
			    (t
			     (let ((op0 parent-op)
				   (rest (cdr arg)))
			       (assert (or (symbolp op0)
					   (stringp op0)))
			       (assert (listp rest))
			       (if (and (member op0 *operators*)
					(member (car arg) *operators*))
				   (let* ((p0 (lookup-precedence op0))
					  (p0assoc (lookup-associativity op0))
					  (op1 (car arg))
					  (p1 (lookup-precedence op1))
					  (p1assoc (lookup-associativity op1)))
				     (if
				      (or (< p0 p1)
					  (and (eq p0 p1)
					       (not (eq p0assoc p1assoc)))
					  (member op0 '(/ // % - **))
					  (member op1 '(/ // % - **)))
				      (format nil "(~a)" (emit `(,op1 ,@rest)))
				      (format nil "~a" (emit `(,op1 ,@rest)))))
				   (emit `(,(car arg) ,@rest)))))))
			 (t
			  (error "unsupported argument for paren* '~a' type='~a'" arg (type-of arg))))))))
	      (+ (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^+~}" (mapcar #'(lambda (x) (emit `(paren* + ,x))) args))
		       (format nil "(~{(~a)~^+~})" (mapcar #'emit args)))))
	      (- (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (if (eq 1 (length args))
			   (format nil "-~a" (emit `(paren* unary- ,(car args))))
			   (format nil "~{~a~^-~}" (mapcar #'(lambda (x) (emit `(paren* - ,x))) args)))
		       (format nil "(~{(~a)~^-~})" (mapcar #'emit args)))))
	      (* (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^*~}" (mapcar #'(lambda (x) (emit `(paren* * ,x))) args))
		       (format nil "(~{(~a)~^*~})" (mapcar #'emit args)))))
	      (@ (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^@~}" (mapcar #'(lambda (x) (emit `(paren* @ ,x))) args))
		       (format nil "(~{(~a)~^@~})" (mapcar #'emit args)))))
	      (== (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^==~}" (mapcar #'(lambda (x) (emit `(paren* == ,x))) args))
			(format nil "(~{(~a)~^==~})" (mapcar #'emit args)))))
	      (<< (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^<<~}" (mapcar #'(lambda (x) (emit `(paren* << ,x))) args))
			(format nil "(~{(~a)~^<<~})" (mapcar #'emit args)))))
	      (!= (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^!=~}" (mapcar #'(lambda (x) (emit `(paren* != ,x))) args))
			(format nil "(~{(~a)~^!=~})" (mapcar #'emit args)))))
	      (< (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^<~}" (mapcar #'(lambda (x) (emit `(paren* < ,x))) args))
		       (format nil "(~{(~a)~^<~})" (mapcar #'emit args)))))
	      (> (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^>~}" (mapcar #'(lambda (x) (emit `(paren* > ,x))) args))
		       (format nil "(~{(~a)~^>~})" (mapcar #'emit args)))))
	      (<= (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^<=~}" (mapcar #'(lambda (x) (emit `(paren* <= ,x))) args))
			(format nil "(~{(~a)~^<=~})" (mapcar #'emit args)))))
	      (>= (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^>=~}" (mapcar #'(lambda (x) (emit `(paren* >= ,x))) args))
			(format nil "(~{(~a)~^>=~})" (mapcar #'emit args)))))
	      (>> (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^>>~}" (mapcar #'(lambda (x) (emit `(paren* >> ,x))) args))
			(format nil "(~{(~a)~^>>~})" (mapcar #'emit args)))))
	      (/ (let ((args (cdr code)))
		   (when (null args)
		     (error "/ requires at least one argument: ~a" code))
		   (if omit-redundant-parentheses
		       (if (eq 1 (length args))
			   (format nil "1.0/~a" (emit `(paren* / ,(car args))))
			   ;; python's / is left associative, just like cl's:
			   ;; (/ a b c) is a/b/c
			   (format nil "~{~a~^/~}" (mapcar #'(lambda (x) (emit `(paren* / ,x))) args)))
		       (if (eq 1 (length args))
			   (format nil "(1.0/(~a))" (emit (car args)))
			   (format nil "(~{(~a)~^/~})" (mapcar #'emit args))))))
	      (** (let ((args (cdr code)))
		    (unless (eq 2 (length args))
		      ;; python's ** is right associative, cl's expt only takes
		      ;; two arguments. don't guess, let the user be explicit.
		      (error "** expects exactly two arguments: ~a" code))
		    (if omit-redundant-parentheses
			(format nil "~a**~a" (emit `(paren* ** ,(first args))) (emit `(paren* ** ,(second args))))
			(format nil "((~a)**(~a))"
				(emit (first args))
				(emit (second args))))))
	      (// (let ((args (cdr code)))
		    (when (< (length args) 2)
		      (error "// requires at least two arguments: ~a" code))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^//~}" (mapcar #'(lambda (x) (emit `(paren* // ,x))) args))
			(format nil "(~{(~a)~^//~})" (mapcar #'emit args)))))
	      (% (let ((args (cdr code)))
		   (when (< (length args) 2)
		     (error "% requires at least two arguments: ~a" code))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^%~}" (mapcar #'(lambda (x) (emit `(paren* % ,x))) args))
		       (format nil "(~{(~a)~^%~})" (mapcar #'emit args)))))
	      (not (destructuring-bind (arg) (cdr code)
		     (if omit-redundant-parentheses
			 (format nil "not ~a" (emit `(paren* not ,arg)))
			 (format nil "(not ~a)" (emit arg)))))
	      (~ (destructuring-bind (arg) (cdr code)
		   (if omit-redundant-parentheses
		       (format nil "~~~a" (emit `(paren* ~ ,arg)))
		       (format nil "(~~~a)" (emit arg)))))
	      (and (let ((args (cdr code)))
		     (if omit-redundant-parentheses
			 (format nil "~{~a~^ and ~}" (mapcar #'(lambda (x) (emit `(paren* and ,x))) args))
			 (format nil "(~{(~a)~^ and ~})" (mapcar #'emit args)))))
	      (& (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^ & ~}" (mapcar #'(lambda (x) (emit `(paren* & ,x))) args))
		       (format nil "(~{(~a)~^ & ~})" (mapcar #'emit args)))))
	      (logand (let ((args (cdr code)))
			(if omit-redundant-parentheses
			    (format nil "~{~a~^ & ~}" (mapcar #'(lambda (x) (emit `(paren* logand ,x))) args))
			    (format nil "(~{(~a)~^ & ~})" (mapcar #'emit args)))))
	      (logxor (let ((args (cdr code)))
			(if omit-redundant-parentheses
			    (format nil "~{~a~^ ^ ~}" (mapcar #'(lambda (x) (emit `(paren* logxor ,x))) args))
			    (format nil "(~{(~a)~^ ^ ~})" (mapcar #'emit args)))))
	      (|\|| (let ((args (cdr code)))
		      (if omit-redundant-parentheses
			  (format nil "~{~a~^ | ~}" (mapcar #'(lambda (x) (emit `(paren* |\|| ,x))) args))
			  (format nil "(~{(~a)~^ | ~})" (mapcar #'emit args)))))
	      (^ (let ((args (cdr code)))
		   (if omit-redundant-parentheses
		       (format nil "~{~a~^ ^ ~}" (mapcar #'(lambda (x) (emit `(paren* ^ ,x))) args))
		       (format nil "(~{(~a)~^ ^ ~})" (mapcar #'emit args)))))
	      (logior (let ((args (cdr code)))
			(if omit-redundant-parentheses
			    (format nil "~{~a~^ | ~}" (mapcar #'(lambda (x) (emit `(paren* logior ,x))) args))
			    (format nil "(~{(~a)~^ | ~})" (mapcar #'emit args)))))
	      (or (let ((args (cdr code)))
		    (if omit-redundant-parentheses
			(format nil "~{~a~^ or ~}" (mapcar #'(lambda (x) (emit `(paren* or ,x))) args))
			(format nil "(~{(~a)~^ or ~})" (mapcar #'emit args)))))
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
			 (if omit-redundant-parentheses
			     (format s "while ~a:~%" (emit vs))
			     (format s "while ~a:~%" (emit `(paren ,vs))))
			 (format s "~a" (emit `(do ,@body))))))

	      (if (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
		    (with-output-to-string (s)
		      (if omit-redundant-parentheses
			  (format s "if ~a:~%~a"
				  (emit condition)
				  (emit `(do ,true-statement)))
			  (format s "if ( ~a ):~%~a"
				  (emit condition)
				  (emit `(do ,true-statement))))
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
					       (if omit-redundant-parentheses
						   (format nil "if True")
						   (format nil "if ( True )")))
					      ((eq i 0)
					       (if omit-redundant-parentheses
						   (format nil "if ~a" (emit condition))
						   (format nil "if ( ~a )" (emit condition))))
					      ((eq condition 't) (emit `(indent "else")))
					      (t (emit `(indent ,(if omit-redundant-parentheses
								     (format nil "elif ~a" (emit condition))
								     (format nil "elif ( ~a )" (emit condition))))))
					      )
					(emit `(do ,@statements)))))
			)))
	      (? (destructuring-bind (condition true-statement &optional (false-statement "None" false-statement-supplied-p))
		     (cdr code)
		   (if omit-redundant-parentheses
		       (if false-statement-supplied-p
			   (format nil "~a if ~a else ~a"
				   (emit `(paren* ternary ,true-statement))
				   (emit `(paren* ternary ,condition))
				   (emit `(paren* ternary ,false-statement)))
			   (format nil "~a if ~a"
				   (emit `(paren* ternary ,true-statement))
				   (emit `(paren* ternary ,condition))))
		       (if false-statement-supplied-p
			   (format nil "(~a) if (~a) else (~a)"
				   (emit true-statement)
				   (emit condition)
				   (emit false-statement))
			   (format nil "~a if (~a)"
				   (emit true-statement)
				   (emit condition))))))
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
			       (if (member form
					   '(else finally))
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
				     (emit `(do "print('Error on line {}.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)"))))))
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
		      (if omit-redundant-parentheses
			  (format str "~a" (print-sufficient-digits-f64 code))
			  (format str "(~a)" (print-sufficient-digits-f64 code))))
		     ((complexp code)
		      (if omit-redundant-parentheses
			  (format str "~a + 1j * ~a"
				  (print-sufficient-digits-f64 (realpart code))
				  (print-sufficient-digits-f64 (imagpart code)))
			  (format str "((~a) + 1j * (~a))"
				  (print-sufficient-digits-f64 (realpart code))
				  (print-sufficient-digits-f64 (imagpart code)))))))))
	"")))
