(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)

;;http://www.celles.net/wiki/Python/raw

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/05_trellis_qt")
  (defparameter *code-file* "run_trellis_gui")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

  (let* ((code
	  `(do0
	    "#!/usr/bin/env python2"


	    (string3 ,(format nil "trellis dataflow gui.
Usage:
  ~a [-vh]

Options:
  -h --help               Show this screen
  -v --verbose            Print debugging output
"
			      *code-file*))
	    
	    "# martin kielhorn 2019-02-14"
	    "# pip2 install --user PySide2"
	    "#  The scripts pyside2-lupdate, pyside2-rcc and pyside2-uic are installed in /home/martin/.local/bin"
	    "# example from https://pypi.org/project/Trellis/0.7a2/"
	    "#  pip install --user Trellis==0.7a2"
	    "# https://github.com/PEAK-Legacy/Trellis"
	    "# to install from github: pip2 install --user ez_setup"
	    "# wget http://peak.telecommunity.com/snapshots/Contextual-0.7a1.dev-r2695.tar.gz http://peak.telecommunity.com/snapshots/Trellis-0.7a3-dev-r2610.tar.gz "
	    "#  pip2 install --user Contextual-0.7a1.dev-r2695.tar.gz"
	    "# i installed trellis by commenting out the contextual line in its setup.py and then extracting the egg file into ~/.local/lib/python2.7/site-packages/peak"
	    "from peak.events import trellis"

	    (imports (os
		      sys
		      docopt
		      (np numpy)
		      pathlib
		      re
		      ))

	    
	    (imports (traceback))

	    (imports ((qw PySide2.QtWidgets)
		      (qc PySide2.QtCore)))
	    
	    "from peak.events import trellis"
	    (setf args (docopt.docopt __doc__ :version (string "0.0.1")))
	    (if (aref args (string "--verbose"))
		(print args))
	    (class Rectangle (trellis.Component)
		   (setf x (trellis.maintain
			    (lambda (self)
			      (+ self.x_min (* .5 self.x_span)))
			    :initially 0)
			 x_span (trellis.maintain
				 (lambda (self)
				   (- self.x_max self.x_min))
				 :initially 0)
			 x_min (trellis.maintain
				(lambda (self)
				  (- self.x (* .5 self.x_span)))
				:initially 0)
			 x_max (trellis.maintain
				(lambda (self)
				  (+ self.x (* .5 self.x_span)))
				:initially 0))
		   "@trellis.perform"
		   (def show_value (self)
		     (print (dot (string "rect {}-{} {}:{}")
				 (format self.x self.x_span
					 self.x_min
					 self.x_max)))))

	    (setf r (Rectangle :x_min 1 :x_max 10))
	    #+nil
	    (do0 ;if (== __name__ (string "__main__"))
	     (setf app (qw.QApplication sys.argv)
		   label (qw.QLabel (string "Hello World"))
		   )
	     (label.show)
	     (sys.exit (app.exec_))))))
    (write-source *source* code)))
