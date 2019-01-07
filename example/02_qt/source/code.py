import sys
import os
import random
import matplotlib
matplotlib.use("Qt5Agg")
import PySide2.QtWidgets as qw
import PySide2.QtCore as qc
import numpy as np
import pandas as pd
import pathlib
import matplotlib.backends.backend_qt5agg as agg
import matplotlib.figure as mf
class PlotCanvas(agg.FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig=mf.Figure(figsize=(width,height,), dpi=dpi)
        self.axes=fig.add_subplot(111)
        self.compute_initial_figure()
        agg.FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)
        agg.FigureCanvasQTAgg.setSizePolicy(self, qw.QSizePolicy.Expanding, qw.QSizePolicy.Expanding)
        agg.FigureCanvasQTAgg.updateGeometry(self)
    def compute_initial_figure(self):
        pass
class StaticCanvas(PlotCanvas):
    def compute_initial_figure(self):
        t=np.arange(0, 3, (9.999999776482582e-3))
        s=np.sin(((2)*(np.pi)*(t)))
        self.axes.plot(t, s)
class DynamicCanvas(PlotCanvas):
    def __init__(self, *args, **kwargs):
        PlotCanvas.__init__(self, *args, **kwargs)
        timer=qc.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)
    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], "r")
    def update_figure(self):
        l=[]
        for i in range(4):
            l.append(random.randint(0, 10))
        self.axes.cla()
        self.axes.plot([0, 1, 2, 3], l, "r")
        self.draw()
class ApplicationWindow(qw.QMainWindow):
    def __init__(self):
        qw.QMainWindow.__init__(self)
        self.setAttribute(qc.Qt.WA_DeleteOnClose)
        self.main_widget=qw.QWidget(self)
        l=qw.QVBoxLayout(self.main_widget)
        sc=StaticCanvas(self.main_widget, width=5, height=4, dpi=100)
        dc=DynamicCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(sc)
        l.addWidget(dc)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
qApp=qw.QApplication(sys.argv)
aw=ApplicationWindow()
aw.show()
sys.exit(qApp.exec_())