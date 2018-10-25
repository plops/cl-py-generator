import sys
import os
import random
import matplotlib

matplotlib.use("Qt5Agg")
import PySide2.QtWidgets
import PySide2.QtCore
import numpy as np
import pandas as pd
import pathlib
import matplotlib.backends.backend_qt5agg
import matplotlib.figure

class PlotCanvas(matplotplib.backends.backend_qt5agg.FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig=matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        self.axes=fig.add_subplot(111)

        self.compute_initial_figure()
        matplotplib.backends.backend_qt5agg.FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)


