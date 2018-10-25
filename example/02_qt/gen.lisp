(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))
;;https://matplotlib.org/2.1.0/gallery/user_interfaces/embedding_in_qt5_sgskip.html
(in-package :cl-py-generator)

(start-python)

(let ((code
       `(do0
	 (imports (sys
		   os
		   random
		   matplotlib))
	 (matplotlib.use (string "Qt5Agg"))
	 (imports ((qw PySide2.QtWidgets)
		   (qc PySide2.QtCore)
		   (np numpy)
		   (pd pandas)
		   pathlib
		   (agg matplotlib.backends.backend_qt5agg)
		   (mf matplotlib.figure)))
	 (class PlotCanvas (agg.FigureCanvasQTAgg)
		(def __init__ (self &key
				    (parent None)
				    (width 5)
				    (height 4)
				    (dpi 100))
		  (setf fig (mf.Figure
			     :figsize (tuple width height)
			     :dpi dpi)
			self.axes (fig.add_subplot 111))
		  (self.compute_initial_figure)

		  (agg.FigureCanvasQTAgg.__init__ self fig)
		  (self.setParent parent)
		  (agg.FigureCanvasQTAgg.setSizePolicy
		   self qw.QSizePolicy.Expanding
		   qw.QSizePolicy.Expanding)
		  (agg.FigureCanvasQTAgg.updateGeometry self))
		(def compute_initial_figure (self)
		  pass))
	 (class StaticCanvas (PlotCanvas)
		(def compute_initial_figure (self)
		  (setf t (np.arange 0 3 .01)
			s (np.sin (* 2 np.pi t))
			)
		  (self.axes.plot t s)))
	 (class DynamicCanvas (PlotCanvas)
		(def __init__ (self *args **kwargs)
		  (PlotCanvas.__init__ self *args **kwargs)
		  (setf timer (qc.QTimer self))
		  (timer.timeout.connect self.update_figure)
		  (timer.start 1000))
		(def compute_initial_figure (self)
		  (self.axes.plot (list 0 1 2 3)
				  (list 1 2 0 4)
				  (string "r")))
		(def update_figure (self)
		  (setf l (list))
		  (for (i (range 4))
		       (l.append (random.randint 0 10)))
		  (self.axes.cla)
		  (self.axes.plot (list 0 1 2 3)
				  l
				  (string "r"))
		  (self.draw)))
	 (class ApplicationWindow (qw.QMainWindow)
		(def __init__ (self)
		  (qw.QMainWindow.__init__ self)
		  (self.setAttribute qc.Qt.WA_DeleteOnClose)
		  (setf self.main_widget (qw.QWidget self)
			l (qw.QVBoxLayout self.main_widget)
			sc (StaticCanvas self.main_widget
					 :width 5
					 :height 4
					 :dpi 100)
			dc (DynamicCanvas self.main_widget
					 :width 5
					 :height 4
					 :dpi 100))
		  (l.addWidget sc)
		  (l.addWidget dc)
		  (self.main_widget.setFocus)
		  (self.setCentralWidget self.main_widget)
		  ))
	 (setf qApp (qw.QApplication sys.argv)
	       aw (ApplicationWindow)
	       )
	 (aw.show)
	 (sys.exit (qApp.exec_))
	 )))
  ;(run code)
  (write-source "/home/martin/stage/cl-py-generator/example/02_qt/source/code" code))
