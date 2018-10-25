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
	 (imports (PySide2.QtWidgets
		   PySide2.QtCore
		   (np numpy)
		   (pd pandas)
		   pathlib
		   matplotlib.backends.backend_qt5agg
		   matplotlib.figure))
	 (class PlotCanvas (matplotplib.backends.backend_qt5agg.FigureCanvasQTAgg)
		(def __init__ (self &key
				    (parent None)
				    (width 5)
				    (height 4)
				    (dpi 100))
		  (setf fig (matplotlib.figure.Figure
			     :figsize (tuple width height)
			     :dpi dpi)
			self.axes (fig.add_subplot 111))
		  (self.compute_initial_figure)
		  (matplotplib.backends.backend_qt5agg.FigureCanvasQTAgg.__init__ self fig)
		  (self.setParent parent))))))
  ;(run code)
  (write-source "/home/martin/stage/cl-py-generator/example/02_qt/source/code" code))
