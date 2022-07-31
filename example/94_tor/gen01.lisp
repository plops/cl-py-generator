(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

;; on arch linux inside termux on android phone:
;; sudo pacman -S jq

(progn
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/94_tor")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key (msg "") vars)
    `(print (dot (string ,(format nil "{:7.6f} \\033[31m ~a \\033[0m ~{~a={}~^ ~}" msg vars))
                 (format
		  (- (time.time) start_time)
                  ,@vars))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let* ((dir (format nil "~a/~a/source/"
				*repo-dir-on-host*
				*example-subdir*
				))
		   (fn  (format nil "~a/~3,'0d_~{~a~^_~}.ipynb"
				dir
				nb-counter path)))
	       (ensure-directories-exist dir)
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ()
	(gen `(tor)
	     `((python
		(cell
					;(imports ((plt matplotlib.pyplot)))
					;(plt.ion)
		 (imports (pathlib
			   time
			   (pd pandas)
			   (np numpy)
					;tqdm

			   ))
		 
		 (setf start_time (time.time))))

	       (python
		(cell

		 
		 ))
	       )))))
  #+nil (progn
   (sb-ext:run-program "/usr/bin/sh"
		       `("/home/martin/stage/cl-py-generator/example/86_playwright/source/setup01_nbdev.sh"))
   (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC )))




