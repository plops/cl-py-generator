(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)

;;http://www.celles.net/wiki/Python/raw
; https://stackoverflow.com/questions/44603119/how-to-display-a-pandas-data-frame-with-pyqt5
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
		      (pd pandas)
		      pathlib
		      re
		      ))

	    
	    (imports (traceback))

	    (imports ((qw PySide2.QtWidgets)
		      (qc PySide2.QtCore)
		      (qg PySide2.QtGui)))

	    (imports (select))
	    "# pip2 install systemd-python"
	    "from systemd import journal"
	    
	    "from peak.events import trellis"
	    (setf args (docopt.docopt __doc__ :version (string "0.0.1")))
	    (if (aref args (string "--verbose"))
		(print args))
	    
	    ;; https://github.com/vfxpipeline/Python-MongoDB-Example/blob/master/lib/customModel.py
	    (class CustomTableModel (qc.QAbstractTableModel)
		   (def __init__ (self dataframe)
		     (qc.QAbstractTableModel.__init__ self)
		     (setf self.dataframe dataframe))
		   (def flags (self index)
		     (if (not (index.isValid))
			 (return None))
		     (return (or qc.Qt.ItemIsEnabled qc.Qt.ItemIsSelectable)))
		   (def rowCount (self *args **kwargs)
		     (return (len self.dataframe.index)))
		   (def columnCount (self *args **kwargs)
		     (return (len self.dataframe.columns)))
		   (def headerData (self section orientation
					 &key (role qc.Qt.DisplayRole))
		     (if (!= qc.Qt.DisplayRole role)
			 (return None))
		     (try
		      (do0
		       (if (== qc.Qt.Horizontal orientation)
			   (return (aref ("list" self.dataframe.columns) section)))
		       (if (== qc.Qt.Vertical orientation)
			   (return (aref ("list" self.dataframe.index) section))))
		      (IndexError
		       (return None))))
		   (def data (self index role)
		     (if (!= qc.Qt.DisplayRole role)
			 (return None))
		     (if (not (index.isValid))
			 (return None))
		     (return (str (aref self.dataframe.ix
					(index.row)
					(index.column))))))

	    (class PandasView (qw.QMainWindow)
		   (def __init__ (self dataframe)
		     (dot (super PandasView self)
			  (__init__))
		     (setf self.model (CustomTableModel dataframe)
			   self.table_view (qw.QTableView)
			   )
		     (self.table_view.setModel self.model)
		     ))
	    (class PandasWindow (qw.QWidget)
		   (def __init__ (self dataframe *args)
		     (qw.QWidget.__init__ self *args)
		     (self.setGeometry 70 150 800 600)
		     (self.setWindowTitle (string "title"))
		     (setf self.pandas_view (PandasView dataframe))
		     (setf self.layout (qw.QVBoxLayout self))
		     (self.layout.addWidget self.pandas_view)
		     (self.setLayout self.layout)))
	    ,(let ((coords `(x y)))
	     `(do0
	      (class Rectangle (trellis.Component)
		     ,(flet ((self (e &optional name)
			       (if name
				   (format nil "self.~a_~a" e name)
				   (format nil "self.~a" e)))
			     (variable (e &optional name)
                               (if name
				   (format nil "~a_~a" e name)
				   (format nil "~a" e))))
			`(do0
			  ,@(loop for e in coords collect
				 `(setf ,e (trellis.maintain
					    (lambda (self)
					      (+ ,(self e "min") (* .5 ,(self e "span"))))
					    :initially 0)
					,(variable e "span") (trellis.maintain
							      (lambda (self)
								(- ,(self e "max") ,(self e "min")))
							      :initially 0)
					,(variable e "min") (trellis.maintain
							     (lambda (self)
							       (- ,(self e)  (* .5 ,(self e "span"))))
							     :initially 0)
					,(variable e "max") (trellis.maintain
							     (lambda (self)
							       (+ ,(self e) (* .5 ,(self e "span"))))
							     :initially 0))
				 )
			  ,(let ((l (loop for e in coords append
					 (loop for f in '(nil span min max) collect
					      (self e f))))
				 (l-var (loop for e in coords append
					     (loop for f in '(nil span min max) collect
						  (variable e f)))))
			     `(do0
			       "@trellis.perform"
			       (def show_value (self)
				 (print (dot (string ,(format nil "rect ~{~a~^ ~}" (loop for e in l-var collect (format nil "~a={}" e))))
					     (format ,@l))))))
			  (do0
			   "@trellis.modifier"
			   (def translate (self r)
			     ,@(loop for e in coords and i from 0 collect
				    `(setf ,(self e) (+ ,(self e)
							(aref r ,i))))))
			  (do0
			   "@trellis.modifier"
			   (def grow (self r)
			     ,@(loop for e in coords and i from 0 collect
				    `(setf ,(self e "span") (+ ,(self e "span")
							       (aref r ,i)))))))))

	      (def make_rect_c (&key (r (np.array (list 0.0 0.0)))
				    (r_span (np.array (list 1.0 1.0))))
		(return (Rectangle ,@(loop for e in coords and i from 0 append
				    `(,(make-keyword (string-upcase e)) (aref r ,i)))
			     ,@(loop for e in coords and i from 0 append
				    `(,(make-keyword (string-upcase (format nil "~a_span" e))) (aref r_span ,i))))))
	      (def make_rect (&key (min (np.array (list 0.0 0.0)))
				  (max (np.array (list 1.0 1.0))))
		(return (Rectangle ,@(loop for e in coords and i from 0 append
				    `(,(make-keyword (string-upcase (format nil "~a_min" e))) (aref min ,i)))
			     ,@(loop for e in coords and i from 0 append
				    `(,(make-keyword (string-upcase (format nil "~a_max" e))) (aref max ,i))))))))

	    (setf r (make_rect_c))
	    (do0
	     "#  https://stackoverflow.com/questions/26331116/reading-systemd-journal-from-python-script"
	     (setf j (journal.Reader))
	     (j.log_level journal.LOG_INFO)
	     (j.seek_tail)
	     (j.get_next)
	     (setf res (list))
	     (while (j.get_next)
	       (for (e j)
		    (if (!= (string "") (aref e (string MESSAGE)))
			(do0
			 (res.append
			  (dict ((string "time") (aref e (string __REALTIME_TIMESTAMP)))
				((string "message") (aref e (string MESSAGE)))))
			 #+nil
			 (print (dot (string "{} {}")
				     (format (aref e (string __REALTIME_TIMESTAMP))
					     (aref e (string MESSAGE))
					     ))))))))
	    (setf df (pd.DataFrame res))
	    
	    (do0		 ;if (== __name__ (string "__main__"))
	     (setf app (qw.QApplication sys.argv)
		   ;label (qw.QLabel (string "Hello World"))
		   )
					;(label.show)
	    ;(setf pv (PandasView df))  (pv.show)
	     (setf win (PandasWindow df))
	     (win.show)
	     (sys.exit (app.exec_))))))
    (write-source *source* code)))
