(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
 
(in-package :cl-py-generator)


;; emerge -av wxpython


(progn
  (defparameter *repo-sub-path* "23_wx")
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
	    (comments "pip3 install --user helium")
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
	    
	    #+nil
	    (do0
	     (imports (PyQt5)
		      )
	     "from PyQt5 import QtCore, QtGui, QtWidgets"
	     "from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel"
	     "from PyQt5.QtCore import QAbstractTableModel, Qt")

	    "from helium import *"
	    
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

	    
	    #+nil
	    (do0
	     (comment "%%")
	     (setf app (wx.App)
		   frm (wx.Frame None :title (string "Hello World")))
	     (frm.Show)
	     (app.MainLoop)
	     )

	    (do0
	     (comment "%% https://wxpython.org/pages/overview/index.html ")
	     (class HelloFrame (wx.Frame)
		    (def __init__ (self *args **kw)
		      (dot (super HelloFrame self)
			   (__init__ *args **kw))
		      (setf pnl (wx.Panel self)
			    st (wx.StaticText pnl :label (string "hello world"))
			    font (st.GetFont)
			    )
		      (setf font.PointSize (+ font.PointSize 10))
		      (setf font (font.Bold))
		      (st.SetFont font)
		      (setf sizer (wx.BoxSizer wx.VERTICAL))
		      (sizer.Add st (dot wx
					 (SizerFlags)
					 (Border (logior wx.TOP
							 wx.LEFT)
						 25)))
		      (pnl.SetSizer sizer)
		      (self.makeMenuBar)
		      (self.CreateStatusBar)
		      (self.SetStatusText (string "Welcome to wxPython"))
		      )
		    (def makeMenuBar (self)
		      (setf fileMenu (wx.Menu)
			    helloItem (fileMenu.Append -1
						       (string "&Hello...\\tCtrl-H")
						       (string "Help string shown in status bar"))
			    )
		      (fileMenu.AppendSeparator)
		      (setf exitItem (fileMenu.Append wx.ID_EXIT))
		      (setf helpMenu (wx.Menu))
		      (setf aboutItem (helpMenu.Append wx.ID_ABOUT))
		      (setf menuBar (wx.MenuBar))
		      (menuBar.Append fileMenu (string "&File"))
		      (menuBar.Append helpMenu (string "&Help"))
		      (self.SetMenuBar menuBar)
		      (self.Bind wx.EVT_MENU self.OnHello helloItem)
		      (self.Bind wx.EVT_MENU self.OnExit exitItem)
		      (self.Bind wx.EVT_MENU self.OnAbout aboutItem))
		    (def OnExit (self event)
		      (self.Close True))
		    (def OnHello (self event)
		      (wx.MessageBox (string "Hello again"))
		      )
		    (def OnAbout (self event)
		      (wx.MessageBox (string "hello sample")
				     (string "about hello world 2")
				     (logior wx.OK
					     wx.ICON_INFORMATION))))
	     (setf app (wx.App)
		   frm (HelloFrame None :title (string "Hello World 2"))
		   )
	     (frm.Show)
	     (app.MainLoop)
	     ))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))





