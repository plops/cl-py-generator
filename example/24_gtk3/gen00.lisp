(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
 
(in-package :cl-py-generator)
;; https://python-gtk-3-tutorial.readthedocs.io/en/latest/treeview.html


(progn
  (defparameter *repo-sub-path* "24_gtk3")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *repo-sub-path*))
  (defparameter *code-file* "run_00_show")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  
  (let* ((code
	  `(do0
	    "#!/usr/bin/python3"
	    
	    (do0
	     #+nil (do0
		    (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
		    "from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
		    "from matplotlib.figure import Figure"
		    (imports ((plt matplotlib.pyplot)
			      matplotlib.colors
			      (animation matplotlib.animation)
			      (xrp xarray.plot)))
		    
		    (plt.ion)
					;(plt.ioff)
		    (setf font (dict ((string size) (string 8))))
		    (matplotlib.rc (string "font") **font)
		    )


	     
	     )
	    

	    
	    (imports (			;os
					;sys
					;traceback
					;pdb
					;time
					;docopt
					;pathlib
					;(yf yfinance)
					;(np numpy)
					;collections
					;serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration 
					;skimage.feature
					;skimage.morphology
					;skimage.measure
					; (u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;scipy.optimize
					;scipy.ndimage.morphology
					; nfft
					; ttv_driver
					;pathlib
					;re
					;requests
					;zipfile
					;io
					;sklearn
					;sklearn.linear_model
		      wx
		      ))
	    (do0
	     (imports ( gi))
	     (gi.require_version (string "Gtk") (string "3.0"))
	     "from gi.repository import Gtk")
	    
	    (do0
	     (comment "%%")
	     (setf
	      _code_git_version
	      (string ,(let ((str 
			      #-sbcl "xxx"
			      #+sbcl (with-output-to-string (s)
				       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/~a.py" *repo-sub-path* *code-file*)
				       )

	      _code_generation_time
	      (string ,(multiple-value-bind
			     (second minute hour date month year day-of-week dst-p tz)
			   (get-decoded-time)
			 (declare (ignorable dst-p))
			 (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				 hour
				 minute
				 second
				 (nth day-of-week *day-names*)
				 year
				 month
				 date
				 (- tz))))))

	    (class ButtonWindow (Gtk.Window)
		   (def __init__ (self)
		     (Gtk.Window.__init__ self :title (string "hello world"))
		     (setf self.button (Gtk.Button :label (string "click here")))
		     (self.button.connect (string "clicked")
					  self.on_button_clicked)
		     (self.add self.button))
		   (def on_button_clicked (self widget)
		     (print (string "hello world"))))
	    
	    #+nil(do0
	     (setf win (ButtonWindow)	; (Gtk.Window)
		   )
	     (win.connect (string "destroy")
			  Gtk.main_quit)
	     (win.show_all)
	     (Gtk.main))
	    (do0 
	     ;; treeview has associated model
	     ;; liststore has rows of data with no children
	     ;; treestore contains rows and rows may have child rows
	     (setf store (Gtk.ListStore str str float)
		   treeiter (store.append (list (string "art of prog")
						(string "knuth")
						24.45)))
	     ;; store.get_iter() will return a TreeIter
	     ;; or use treepath
	     (setf tree (Gtk.TreeView store)))
	    
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))





