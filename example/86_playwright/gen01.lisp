(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/86_playwright")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key (msg "") vars)
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
      (let* ()
	(gen `(cloud)
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
		 #+nil (imports-from (matplotlib.pyplot plot figure scatter gca sca subplots subplots_adjust title xlabel ylabel xlim ylim grid))
		 (imports-from (playwright.sync_api sync_playwright))
		 (setf start_time (time.time))))

	       (python
		(cell

		 (with (as (sync_playwright)
			   p)
		       (setf browser
			     (p.chromium.launch
			      :headless False))
		       (setf page (browser.new_page

				   ))
		       (page.goto (string "https://quotes.toscrape.com/"))
		       (do0
			(setf head_sel (string "//h1/a"))
			(setf head (page.query_selector head_sel))
			(print (dot head (inner_text))))

		       (do0
			(setf login (page.query_selector
				     (string3 "[href=\"/login\"]")))
			(login.click))
		       (do0
			(setf user (page.query_selector
				    (string3 "[id=\"username\"]")))
			(user.type (string "bla")))
		       (do0
			(setf pw (page.query_selector
				  (string3 "[id=\"password\"]")))
			(pw.type (string "bla")))
		       (dot
			page
			(query_selector
			 (string3 "[type='submit']"))
			(click))
		       (do0
			(setf logout_sel
			      (string3 "//*[@href='/logout']"))
			(try
			 (setf logout
			       (dot page
				    (wait_for_selector logout_sel
						       :timeout 5000)))
			 ("Exception as e"
			  (print (string "failed"))))
			(print (logout.inner_text)))
		       (do0
			(setf quotes
			      (dot
			       page
			       (query_selector_all
				(string3 "[class='quote']"))))
			(for (q quotes)
			     (print
			      (dot q
				   (query_selector
				    (string ".text")
				    )
				   (inner_text)
				   )))
			)
		       ;(page.wait_for_timeout 2000)
		       (browser.close))
		 ))
	       )))))
  (sb-ext:run-program "/usr/bin/sh"
		      `("/home/martin/stage/cl-py-generator/example/86_playwright/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




