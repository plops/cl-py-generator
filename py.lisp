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
  ;; Binding power of the operators, from the tightest binding group (index 0)
  ;; to the loosest one.  The index and the associativity are all that
  ;; `operand-needs-parentheses-p' needs to decide whether an operand has to be
  ;; wrapped in parentheses.
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
    (:op (= setf) :assoc r)
    ;; the loosest constructs of the language.  They are not emitted by
    ;; `emit-operator', but they must be known here: a lambda or a bare comma
    ;; list swallows everything to its right, so it always needs parentheses
    ;; when it appears inside a larger expression.
    (:op (lambda ntuple) :assoc r)))

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

(defun join-strings (separator strings)
  "Concatenate STRINGS and put SEPARATOR between two consecutive elements."
  (with-output-to-string (s)
    (loop for string in strings
          and i from 0
          do (unless (zerop i)
               (write-string separator s))
             (write-string string s))))

;;; How the operators of the DSL are printed.  This used to be ~25 nearly
;;; identical CASE clauses in `emit-py'; all that ever differed between them is
;;; collected in the two tables below.  Adding an operator means: add one line
;;; here and one group entry in `*precedence*'.
;;;
;;; Recognized keys of an entry of `*infix-operators*':
;;;   :separator            string printed between two operands (required)
;;;   :min-args, :max-args  arity check, NIL means "any number of arguments"
;;;   :style                :infix (default) parenthesizes operands only where
;;;                         the precedence requires it, :always-paren keeps one
;;;                         pair of parentheses around the whole expression
;;;                         (`in', `is', ... where python needs no parentheses
;;;                         but they have always been emitted)
;;;   :unary-format         format string (one ~a: the operand) used when the
;;;                         operator is called with a single argument, e.g.
;;;                         (- x) -> -x and (/ x) -> 1.0/x
;;;   :unary-legacy-format  same, but for the fully parenthesized output
;;;   :unary-operand-op     operator used for the precedence lookup of the
;;;                         operand of the unary form (defaults to the operator
;;;                         itself; unary minus binds tighter than binary minus)
(defparameter *infix-operators*
  '((+      :separator "+")
    (-      :separator "-"  :unary-format "-~a" :unary-operand-op unary-
            ;; NOTE: without this the fully parenthesized mode used to print
            ;; (- x) as ((x)) -- it silently dropped the minus sign
            :unary-legacy-format "(-(~a))")
    (*      :separator "*")
    (@      :separator "@")
    (/      :separator "/"  :min-args 1
            :unary-format "1.0/~a" :unary-legacy-format "(1.0/(~a))")
    ;; python's / // % are left associative, just like common lisp's:
    ;; (/ a b c) is a/b/c
    (//     :separator "//" :min-args 2)
    (%      :separator "%"  :min-args 2)
    ;; python's ** is right associative, cl's expt only takes two arguments.
    ;; don't guess, let the user be explicit.
    (**     :separator "**" :min-args 2 :max-args 2)
    (<<     :separator "<<")
    (>>     :separator ">>")
    (<      :separator "<")
    (>      :separator ">")
    (<=     :separator "<=")
    (>=     :separator ">=")
    (==     :separator "==")
    (!=     :separator "!=")
    (&      :separator " & ")
    (logand :separator " & ")
    (^      :separator " ^ ")
    (logxor :separator " ^ ")
    (|\||   :separator " | ")
    (logior :separator " | ")
    (and    :separator " and ")
    (or     :separator " or ")
    (in     :separator " in "     :min-args 2 :max-args 2 :style :always-paren)
    (not-in :separator " not in " :min-args 2 :max-args 2 :style :always-paren)
    (is     :separator " is "     :min-args 2 :max-args 2 :style :always-paren)
    (is-not :separator " is not " :min-args 2 :max-args 2 :style :always-paren)))

;;; Unary operators that are printed in front of their operand.  Recognized
;;; keys: :prefix (the string, required) and :min-args/:max-args as above.
(defparameter *prefix-operators*
  '((not :prefix "not " :min-args 1 :max-args 1)
    (~   :prefix "~"    :min-args 1 :max-args 1)))

(defparameter *chaining-operators*
  '(< <= > >= != == in not-in is is-not)
  "The comparison operators.  Python chains them: a<b==c means (a<b) and (b==c),
   so a comparison that appears as an operand of a comparison always keeps its
   parentheses, no matter on which side it stands.")

(defparameter *associative-operators*
  '((+) (*) (@) (& logand) (^ logxor) (|\|| logior) (and) (or))
  "Groups of operators that may be nested into each other without parentheses,
   i.e. where a+(b+c) may be printed as a+b+c.  Two operators of the same
   precedence that do not share a group need parentheses on the side that their
   associativity does not favour: `*' and `@' sit in one precedence row, but
   a*(b@c) is not (a*b)@c, and `&'/`logand' are two spellings of the same
   python operator and therefore associative with each other.

   NOTE: `+' and `*' are treated as associative although for floats that only
   holds up to rounding.  The emitter has always done this.")

(defun mutually-associative-p (op1 op2)
  "Return T when OP1 and OP2 may be nested without parentheses, see
   `*associative-operators*'."
  (loop for group in *associative-operators*
        thereis (and (member op1 group)
                     (member op2 group))))

(defun infix-operator-spec (name)
  "Return the property list that describes the infix operator NAME, or NIL."
  (cdr (assoc name *infix-operators*)))

(defun prefix-operator-spec (name)
  "Return the property list that describes the prefix operator NAME, or NIL."
  (cdr (assoc name *prefix-operators*)))

(defun check-operator-tables ()
  "Make sure every operator mentioned in one of the emitter tables also has an
   entry in `*precedence*'.  Without it the parenthesis elision would not know
   how tightly the operator binds and would silently drop parentheses -- a
   missing entry (or a typo) has to be loud, not subtly wrong."
  (flet ((check (op where)
           (when (and op (not (member op *operators*)))
             (error "the operator ~a of ~a is missing in *precedence*"
                    op where))))
    (loop for entry in (append *infix-operators* *prefix-operators*)
          do (destructuring-bind (name &rest spec) entry
               (check name '*infix-operators*)
               (check (getf spec :unary-operand-op) '*infix-operators*)))
    (loop for op in *chaining-operators*
          do (check op '*chaining-operators*))
    (loop for group in *associative-operators*
          do (loop for op in group
                   do (check op '*associative-operators*))))
  t)

(check-operator-tables)

(defun check-operator-arity (name args spec form)
  "Signal an error when the number of ARGS does not fit the :MIN-ARGS and
   :MAX-ARGS entries of SPEC.  FORM is only used for the error message."
  (let ((min (getf spec :min-args))
        (max (getf spec :max-args))
        (n (length args)))
    (if (and min max (eql min max))
        (unless (eql n min)
          (error "~a expects exactly ~r argument~:p: ~a" name min form))
        (progn
          (when (and min (< n min))
            (error "~a requires at least ~r argument~:p: ~a" name min form))
          (when (and max (< max n))
            (error "~a accepts at most ~r argument~:p: ~a" name max form))))))

(defun operand-needs-parentheses-p (parent-op child-op &optional position)
  "Return T when an operand that prints the operator CHILD-OP has to be wrapped
   in parentheses to keep its meaning inside PARENT-OP.

   POSITION is `:left' for the first (left) operand, `:right' for every
   following one, and NIL when the position does not matter.  With NIL only an
   operand that binds strictly looser gets parentheses; this is used for the
   object of `dot' and the sequence of `aref', where parentheses around an
   operand of the same precedence would be wrong (a.(b[i]) is not python).

   An operand that is not an operator (a function call, a literal, a form that
   brings its own parentheses) has CHILD-OP NIL and never needs parentheses.

   `/', `//', `%', `-' and `**' get parentheses whenever they take part.  The
   position rule below would be enough for the ones on the left, but the
   emitter has always been generous here and that only costs readability, so it
   stays: it keeps the output of ~165 examples stable."
  (let ((parent-precedence (lookup-precedence parent-op))
        (child-precedence (lookup-precedence child-op)))
    (when (and parent-precedence child-precedence)
      (or ;; the parent binds tighter than the child
       (< parent-precedence child-precedence)
       ;; python chains comparisons: a==(b<c) printed as a==b<c would mean
       ;; (a==b) and (b<c), so a comparison inside a comparison always keeps
       ;; its parentheses -- on either side
       (and (member parent-op *chaining-operators*)
            (member child-op *chaining-operators*))
       ;; same precedence: the operand on the side that the associativity does
       ;; not favour would be regrouped, e.g. a-(b-c) -> a-b-c or
       ;; a<<(b>>c) -> a<<b>>c or 2**(3**4) -> 2**3**4
       (and position
            (eql parent-precedence child-precedence)
            (not (mutually-associative-p parent-op child-op))
            (if (eq 'r (lookup-associativity parent-op))
                (eq :left position)
                (eq :right position)))
       (member parent-op '(/ // % - **))
       (member child-op '(/ // % - **))))))

(defun effective-operator (form)
  "Return the operator that FORM really prints, or NIL when FORM is a primary
   expression that never needs parentheses (a function call, a literal, a form
   that brings its own parentheses).

   The printed operator is not always the head of the form:

     (- x)     prints -x         -> unary-
     (/ x)     prints 1.0/x      -> /
     (+ x)     prints x          -> the operator of x, recursively
     (in a b)  prints (a in b)   -> NIL, it is parenthesized already
     (f x y)   prints f(x, y)    -> NIL

   Asking the head of the form instead (or the old shortcut `a form with two
   elements needs no parentheses') silently dropped the parentheses of unary
   forms: (** (- a) 2) has to be (-a)**2 and not -a**2."
  (unless (atom form)
    (let* ((head (first form))
           (args (rest form))
           (spec (infix-operator-spec head)))
      (cond
        ;; (in a b), (is a b), ... are printed with their own parentheses
        ((and spec (eq :always-paren (getf spec :style :infix)))
         nil)
        ((and spec (eql 1 (length args)))
         (if (getf spec :unary-format)
             ;; (- x) prints the unary operator, (/ x) prints 1.0/x
             (getf spec :unary-operand-op head)
             ;; an n-ary operator with a single argument prints just that
             ;; argument, e.g. (or 255) -> 255
             (effective-operator (first args))))
        ((member head *operators*) head)))))

(defun emit-operand (parent-op arg emit &optional position)
  "Emit ARG as an operand of PARENT-OP, wrapped in parentheses only when
   `operand-needs-parentheses-p' asks for them.  EMIT is the recursive emitter
   closure of `emit-py', POSITION is documented at
   `operand-needs-parentheses-p'.  This is what the `paren*' form of the DSL
   does."
  (flet ((parenthesize-p (child-op)
           (operand-needs-parentheses-p parent-op child-op position)))
    (cond
      ((symbolp arg)			; variable name (or NIL, which emits "")
       (funcall emit arg))
      ((complexp arg)
       ;; a complex literal is printed as the expression (re + 1j * im), so it
       ;; always needs parentheses
       (format nil "(~a)" (funcall emit arg)))
      ((and (realp arg) (minusp arg))
       ;; a negative literal prints as a unary minus expression, so (** -2 2)
       ;; must not become -2**2 (which python reads as -(2**2))
       (if (parenthesize-p 'unary-)
           (format nil "(~a)" (funcall emit arg))
           ;; keep a space, otherwise (- a -1) would turn into a--1
           (format nil " ~a" (funcall emit arg))))
      ((numberp arg)
       (funcall emit arg))
      ((stringp arg)			; raw python code, inserted verbatim
       arg)
      ((listp arg)
       (if (parenthesize-p (effective-operator arg))
           (format nil "(~a)" (funcall emit arg))
           (funcall emit arg)))
      (t (error "unsupported operand of ~a: '~a' type='~a'"
                parent-op arg (type-of arg))))))

(defun emit-infix-operator (name args emit omit-redundant-parentheses form)
  "Emit the python code for the n-ary operator NAME applied to ARGS.

   EMIT is the recursive emitter closure of `emit-py', FORM the whole
   s-expression (used for error messages only).  With
   OMIT-REDUNDANT-PARENTHESES the operands are parenthesized only where the
   operator precedence requires it, otherwise the fully parenthesized legacy
   output is produced, e.g. (+ a b) -> ((a)+(b))."
  (let* ((spec (infix-operator-spec name))
         (separator (getf spec :separator))
         (unary-format (if omit-redundant-parentheses
                           (getf spec :unary-format)
                           (getf spec :unary-legacy-format))))
    (check-operator-arity name args spec form)
    (cond
      ((eq :always-paren (getf spec :style :infix))
       ;; the whole expression is parenthesized, but an operand may still need
       ;; parentheses of its own: ((a==b) in c) must not become (a==b in c),
       ;; which python reads as the chained comparison (a==b) and (b in c)
       (format nil "(~a)"
               (join-strings separator
                             (loop for arg in args
                                   and i from 0
                                   collect (if omit-redundant-parentheses
                                               (emit-operand name arg emit
                                                             (if (zerop i)
                                                                 :left
                                                                 :right))
                                               (funcall emit arg))))))
      ((and unary-format (eql 1 (length args)))
       (format nil unary-format
               (if omit-redundant-parentheses
                   (emit-operand (getf spec :unary-operand-op name)
                                 (first args) emit :right)
                   (funcall emit (first args)))))
      (omit-redundant-parentheses
       (join-strings separator
                     (loop for arg in args
                           and i from 0
                           ;; the first operand stands on the left of the
                           ;; operator, all others on its right
                           collect (emit-operand name arg emit
                                                 (if (zerop i) :left :right)))))
      (t (format nil "(~a)"
                 (join-strings separator
                               (loop for arg in args
                                     collect (format nil "(~a)"
                                                     (funcall emit arg)))))))))

(defun emit-prefix-operator (name args emit omit-redundant-parentheses form)
  "Emit the python code for the unary prefix operator NAME applied to ARGS,
   e.g. (not x) -> not x and (~ x) -> ~x.  See `emit-infix-operator' for the
   meaning of the arguments."
  (let ((spec (prefix-operator-spec name)))
    (check-operator-arity name args spec form)
    (let ((prefix (getf spec :prefix))
          (arg (first args)))
      (if omit-redundant-parentheses
          (format nil "~a~a" prefix (emit-operand name arg emit :right))
          (format nil "(~a~a)" prefix (funcall emit arg))))))

(defun emit-operator (form emit omit-redundant-parentheses)
  "Emit FORM when its head is an operator of `*infix-operators*' or
   `*prefix-operators*', otherwise return NIL (`emit-py' then falls through to
   its CASE clauses)."
  (let ((name (first form)))
    (cond ((infix-operator-spec name)
           (emit-infix-operator name (rest form) emit
                                omit-redundant-parentheses form))
          ((prefix-operator-spec name)
           (emit-prefix-operator name (rest form) emit
                                 omit-redundant-parentheses form)))))

(defun emit-condition (keyword condition emit omit-redundant-parentheses)
  "Emit the KEYWORD (\"if\", \"elif\" or \"while\") of a python statement
   together with its CONDITION.  Without OMIT-REDUNDANT-PARENTHESES the
   condition is wrapped in parentheses."
  (if omit-redundant-parentheses
      (format nil "~a ~a" keyword (funcall emit condition))
      (format nil "~a ( ~a )" keyword (funcall emit condition))))

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
	    ;; the operators are table driven (see `*infix-operators*'), NIL
	    ;; means: FORM is not an operator, fall through to the CASE below
	    (or (emit-operator code #'emit omit-redundant-parentheses)
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
		      ;; the indices are delimited by the brackets, only the
		      ;; sequence itself may need parentheses: (aref (+ a b) i)
		      ;; is (a+b)[i]
		      (format nil "~a[~{~a~^,~}]"
			      (if omit-redundant-parentheses
				  (emit-operand 'aref name #'emit)
				  (emit name))
			      (mapcar #'emit indices))))
	      (slice (let ((args (cdr code)))
		       (if (null args)
			   (format nil ":")
			   (format nil "~{~a~^:~}" (mapcar #'emit args)))))
	      (dot (let ((args (remove-if #'null (cdr code))))
		     ;; don't print . for nil arguments.  Only the object may
		     ;; need parentheses: (dot (- a b) c) is (a-b).c, whereas
		     ;; the attributes and method calls behind the dots are
		     ;; primary expressions.
		     (format nil "~{~a~^.~}"
			     (loop for arg in args
				   and i from 0
				   collect (if (and omit-redundant-parentheses
						    (zerop i))
					       (emit-operand 'dot arg #'emit)
					       (emit arg))))))
	      (paren*
	       ;; (paren* parent-op operand &optional position)
	       ;; place a pair of parentheses only when needed, see
	       ;; `emit-operand' and `operand-needs-parentheses-p'
	       (if omit-redundant-parentheses
		   (progn
		     (unless (member (length code) '(3 4))
		       (error "paren* expects two or three arguments: ~a" code))
		     (unless (member (fourth code) '(nil :left :right))
		       (error "the operand position of paren* must be :left, :right or omitted: ~a" code))
		     (emit-operand (second code) (third code) #'emit
				   (fourth code)))
		   ;; without the precedence machinery paren* degenerates into paren
		   (format nil "(~{~a~^, ~})" (mapcar #'emit (cddr code)))))
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
			 (format s "~a:~%" (emit-condition "while" vs #'emit
							   omit-redundant-parentheses))
			 (format s "~a" (emit `(do ,@body))))))

	      (if (destructuring-bind (condition true-statement &optional false-statement) (cdr code)
		    (with-output-to-string (s)
		      (format s "~a:~%~a"
			      (emit-condition "if" condition #'emit
					      omit-redundant-parentheses)
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
					       (emit-condition "if" "True" #'emit
							       omit-redundant-parentheses))
					      ((eq i 0)
					       (emit-condition "if" condition #'emit
							       omit-redundant-parentheses))
					      ((eq condition 't) (emit `(indent "else")))
					      (t (emit `(indent ,(emit-condition
								  "elif" condition #'emit
								  omit-redundant-parentheses)))))
					(emit `(do ,@statements)))))
			)))
	      (? (destructuring-bind (condition true-statement &optional (false-statement "None" false-statement-supplied-p))
		     (cdr code)
		   (if omit-redundant-parentheses
		       ;; <true> if <condition> else <false>.  The conditional
		       ;; expression is right associative, so only the false
		       ;; branch may contain another one without parentheses.
		       (if false-statement-supplied-p
			   (format nil "~a if ~a else ~a"
				   (emit-operand 'ternary true-statement #'emit :left)
				   (emit-operand 'ternary condition #'emit :left)
				   (emit-operand 'ternary false-statement #'emit :right))
			   (format nil "~a if ~a"
				   (emit-operand 'ternary true-statement #'emit :left)
				   (emit-operand 'ternary condition #'emit :left)))
		       (if false-statement-supplied-p
			   ;; NOTE: the outer parentheses matter.  Without them a
			   ;; conditional expression used as the object of dot or
			   ;; as the sequence of aref regrouped, e.g.
			   ;; (aref (? c a b) i) became (a) if (c) else (b)[i].
			   ;; The two argument form must stay unparenthesized: it
			   ;; is the filter of a comprehension.
			   (format nil "((~a) if (~a) else (~a))"
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
							`(= ,(format nil "~a" e) ,(getf plist e)))))))))))))
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
