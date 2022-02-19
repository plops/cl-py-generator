(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  ;; the following code needs inverted readtable, otherwise symbols
  ;; and filenames may have the wrong case and everything breaks in
  ;; horrible ways
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/82_spline")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key msg vars)
    `(print (dot (string ,(format nil "{:7.6f} \\033[31m ~a \\033[0m ~{~a={}~^ ~}" msg vars))
                 (format
		  (- (time.time) start_time)
                  ,@vars))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ((lam-term `(5e9 316 3.16 ))
	     (model `(LinearGAM (+ (s 0 :n_splines 20 :lam ,(elt lam-term 0))
				   (s 1 :n_splines 20 :lam ,(elt lam-term 1))
				   (f 2 :lam ,(elt lam-term 2)))))
	     (search-up 5)
	     (search-down 5))
	(gen `(spline)
	     `((python
		(cell
		 (imports ((plt matplotlib.pyplot)))
		 (plt.ion)
		 (imports (pathlib
			   time
			   tqdm
			   (pd pandas)
			   (np numpy)
			   numpy.random
			   ))
		 (imports-from (matplotlib.pyplot plot figure scatter gca sca subplots subplots_adjust title xlabel ylabel xlim ylim grid))
		 (setf start_time (time.time))))

	       (python
		(cell
		 ,(lprint :msg "generate dataset")
		 (setf xmi .3
		       xma 6.7
		       x (np.linspace xmi xma 120)
		       y (np.sin x))
		 (plot x y)
		 (grid)))

	       ,(flet ()
		  `(python
		    (cell
		     (comments "piecewise linear basis (p. 164)")
		     (do0 (setf k 12)
			  (setf sk (np.linspace xmi xma k)
				hsk (np.diff sk)))
		     (def tent (x xj j)
		       (rstring3 "tent function from set defined by knots xj")
		       (setf dj (* xj 0)
			     (aref dj j) 1)
		       (return (np.interp x dj )))
		     #+nil (def b (j x)
			     (when (== j 1)
			       (if (< x (aref sk 2))
				   (return (/ (- (aref sk 2) x)
					      (- (aref sk 2) (aref sk 1))))
				   (return 0.0)))
			     (when (== j k)
			       (if (< (aref sk (- k 1)) x )
				   (return (/ (- x (aref sk (- k 1)))
					      (- (aref sk k) (aref sk (- k 1)))))
				   (return 0.0)))
			     (cond
			       ((< (aref sk (- j 1))
				   x
				   (aref sk j))
				(return (/ (- x (aref sk (- j 1)))
					   (- (aref sk j)
					      (aref sk (- j 1))))))
			       ((< (aref sk j)
				   x
				   (aref sk (+ j 1)))
				(return (/ (- (aref sk (+ j 1))
					      x)
					   (- (aref sk (+ j 1))
					      (aref sk j)))))
			       (t (return 0.0))
			       ))
		     )))

	       #+nil
	       ,(flet ((a- (j x)
			 `(/ (- (aref sk (+ ,j 1))
				,x)
			     (aref hsk ,j)))
		       (a+ (j x)
			 `(/ (- ,x
				(aref sk ,j))
			     (aref hsk ,j)))
		       (c- (j x)
			 `(/ (- (/ (** (- (aref sk (+ ,j 1))
					  ,x)
				       3)
				   (aref hsk ,j))
				(* (aref hsk ,j) (- (aref sk (+ ,j 1))
						    ,x)))
			     6))
		       (c+ (j x)
			 `(/ (- (/ (** (- ,x
					  (aref sk ,j))
				       3)
				   (aref hsk ,j))
				(* (aref hsk ,j) (- ,x
						    (aref sk ,j)
						    )))
			     6))
		       (D (i j)
			 (cond ((eq i j)
				`(/ 1.0 (aref hsk ,i)))
			       ((eq (+ i 1) j)
				`(- (/ 1.0 (aref hsk ,i))
				    (/ 1.0 (aref hsk ,(+ i 1)))))
			       ((eq (+ i 2) j)
				`(/ 1.0 (aref hsk ,(+ i 1))))
			       (t 0.0)))
		       (B (i j)
			 (cond ((eq i j)
				`(/ (+ (aref hsk ,i)
				       (aref hsk ,(+ 1 i)))
				    3))
			       ((eq (+ i 1) j)
				`(/ (aref hsk ,(+ i 1))
				    6)
				)
			       ((eq (+ j 1) j)
				`(/  (aref hsk ,(+ j 1))
				     6))
			       (t 0.0)))
		       )
		  (python
		   (cell
		    (do0 (setf k 12)
			 (setf sk (np.linspace xmi xma k)
			       hsk (np.diff sk)))

		    )))
	       )))))
  (sb-ext:run-program "/usr/bin/sh"
		      `("/home/martin/stage/cl-py-generator/example/82_spline/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




