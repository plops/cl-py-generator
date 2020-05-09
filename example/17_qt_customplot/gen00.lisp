(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *repo-sub-path* "17_qt_customplot")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *repo-sub-path*))
  (defparameter *code-file* "run_00_plot")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  
(let* ((code
	`(do0
	  (comment "pip3 install --user QCustomPlot2")
	  (comment "change gui font size in linux: xrandr --output HDMI-0 --dpi 55")
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

	   (imports (PyQt5)
		    ))
	  "from PyQt5 import QtCore, QtGui, QtWidgets"
	  "from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel"
	  "from PyQt5.QtCore import QAbstractTableModel, Qt"
	  

	  (imports (			;os
					;sys
					;traceback
					;pdb
		    time
					;docopt
					;pathlib
		    (np numpy)
					;serial
		    (pd pandas)
		    (xr xarray)
		    (xrp xarray.plot)
		    skimage.restoration 
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
		    pathlib
		    re
		    requests
		    zipfile
		    io
					;sklearn
					;sklearn.linear_model
		    ))

	  (comment "%%")

	  (do0
	   

	   (do0
	    (setf output_path (string  "/dev/shm"))
	    )
	   )
	  
	  (setf
	   _code_git_version
	   (string ,(let ((str (with-output-to-string (s)
				 (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		      (subseq str 0 (1- (length str)))))
	   _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/example/~a/~a.py" *repo-sub-path* *code-file*)
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
			      (- tz)))))

	  (class DataFrameModel (QtCore.QAbstractTableModel)
		 "# this is boiler plate to render a dataframe as a QTableView"
		 "# https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/"
		 
		 (def __init__ (self &key (df (pd.DataFrame)) (parent None))
		   (QAbstractTableModel.__init__ self)
		   (setf self._dataframe df))


		 (def rowCount (self &key (parent None))
		   (return (aref self._dataframe.shape 0)))

		 (def columnCount (self &key (parent None))
		   (return (aref self._dataframe.shape 1)))

		 (def data (self index &key (role QtCore.Qt.DisplayRole))
		   (when (index.isValid)
		     (when (== role QtCore.Qt.DisplayRole)
		       (return (str (aref self._dataframe.iloc
					  (index.row)
					  (index.column))))))
		   (return None))
		 
		 
		 (def headerData (self
				  col
				  orientation role)
		   (when (and (== orientation QtCore.Qt.Horizontal)
			      (== role QtCore.Qt.DisplayRole))
		     (return (aref self._dataframe.columns col)))
		   (return None)))
	  
	  
	  (do0
	   (comment "%% open gui windows")
	   (setf app (QApplication (list (string ""))))
	   
	   (do0
	    (setf window (QWidget)
		  layout_h (QHBoxLayout window)
		  layout (QVBoxLayout))
	    (layout_h.addLayout layout)
	    (window.setWindowTitle (string ,*code-file*))
	    (setf table (QtWidgets.QTableView window))
	    (do0
	     "# select whole row when clicking into table"
	     (table.setSelectionBehavior QtWidgets.QTableView.SelectRows))

	    
	    ,@(loop for e in `(table)
		 collect
		   `(dot layout (addWidget ,e)))
	    
	    (window.show)
	    
	    (def selectionChanged (selected deselected)
	      (do0
	       "global other_table, df"
	       
	       (unless "other_table is None"
		 (do0
		  "# https://stackoverflow.com/questions/5889705/pyqt-how-to-remove-elements-from-a-qvboxlayout/5890555"
		  (zip_table.setParent None))))

	      

	      
	      (setf row  (aref df.iloc (dot (aref (selected.indexes) 0)
					    (row))))
	      
	      )
	    
	    (do0
	     (do0
	      (setf df (pd.DataFrame ("list" (dot (pathlib.Path (string "."))
						  (glob (string "*"))))))
	      (setf model (DataFrameModel df))
	      (table.setModel model))
	     (dot table
		  (selectionModel)
		  selectionChanged
		  (connect selectionChanged))))
	   (def run0 ()
	     (app.exec_))))))
  (write-source (format nil "~a/source/~a" *path* *code-file*) code)
  #+nil (sb-ext:run-program "/usr/bin/scp" `("-C"
					     ,(format nil "source/~a.py" *code-file*) "10.1.99.22:./"))))





