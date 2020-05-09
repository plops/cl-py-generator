# pip3 install --user QCustomPlot2
# change gui font size in linux: xrandr --output HDMI-0 --dpi 55
# https://pypi.org/project/QCustomPlot2/
# https://osdn.net/users/salsergey/pf/QCustomPlot2-PyQt5/scm/blobs/master/examples/plots/mainwindow.py
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QPen, QBrush, QColor
from QCustomPlot2 import *
import numpy as np
import collections
import pandas as pd
import pathlib
# %%
output_path="/dev/shm"
_code_git_version="4fcdc05aeb4fc0f8846b3335936d6732fc652741"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/17_qt_customplot/source/run_00_plot.py"
_code_generation_time="11:36:29 of Saturday, 2020-05-09 (GMT+1)"
class DataFrameModel(QtCore.QAbstractTableModel):
    # this is boiler plate to render a dataframe as a QTableView
    # https://learndataanalysis.org/display:pandas:dataframe:with:pyqt5:qtableview:widget/
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self)
        self._dataframe=df
    def rowCount(self, parent=None):
        return self._dataframe.shape[0]
    def columnCount(self, parent=None):
        return self._dataframe.shape[1]
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if ( index.isValid() ):
            if ( ((role)==(QtCore.Qt.DisplayRole)) ):
                return str(self._dataframe.iloc[index.row(),index.column()])
        return None
    def headerData(self, col, orientation, role):
        if ( ((((orientation)==(QtCore.Qt.Horizontal))) and (((role)==(QtCore.Qt.DisplayRole)))) ):
            return self._dataframe.columns[col]
        return None
# %% open gui windows
app=QApplication([""])
window=QWidget()
layout_h=QHBoxLayout(window)
layout=QVBoxLayout()
layout_h.addLayout(layout)
window.setWindowTitle("run_00_plot")
table=QtWidgets.QTableView(window)
# select whole row when clicking into table
table.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
custom_plot=QCustomPlot()
custom_plot.setFixedHeight(250)
graph=custom_plot.addGraph()
x=np.linspace(-3, 3, 300)
graph.setPen(QPen(Qt.blue))
custom_plot.rescaleAxes()
custom_plot.setInteractions(QCP.Interactions(((QCP.iRangeDrag) | (QCP.iRangeZoom) | (QCP.iSelectPlottables))))
layout.addWidget(table)
layout.addWidget(custom_plot)
window.show()
def selectionChanged(selected, deselected):
    global other_table, df
    if ( not(other_table is None) ):
        # https://stackoverflow.com/questions/5889705/pyqt:how:to:remove:elements:from:a:qvboxlayout/5890555
        zip_table.setParent(None)
    row=df.iloc[selected.indexes()[0].row()]
# the only realtime data source i can think of: power consumption of my cpu
def read_from_file(fn):
    with open(fn, "r") as file:
        return file.read().replace("\n", "")
df=pd.DataFrame({("input_fn"):(list(map(str, pathlib.Path("/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0/").glob("*input"))))})
df["base_fn"]=df.input_fn.str.extract("(.*)_input")
df["label_fn"]=df.base_fn.apply(lambda x: ((x)+("_label")))
df["label"]=df.label_fn.apply(read_from_file)
df["value"]=df.input_fn.apply(read_from_file)
df["values"]=df.input_fn.apply(lambda x: collections.deque(maxlen=1000))
model=DataFrameModel(df)
table.setModel(model)
table.selectionModel().selectionChanged.connect(selectionChanged)
def update_values():
    global df, graph, custom_plot
    for (idx,row,) in df.iterrows():
        df.loc[idx,"value"]=read_from_file(row.input_fn)
        row["values"].append(int(read_from_file(row.input_fn)))
    model=DataFrameModel(df)
    table.setModel(model)
    y=df.iloc[1]["values"]
    graph.setData(range(len(y)), y)
    custom_plot.rescaleAxes()
    custom_plot.replot()
timer=PyQt5.QtCore.QTimer()
timer.setInterval(10)
timer.timeout.connect(update_values)
timer.start()
def run0():
    # apparently i don't need to call this. without it i can interact with python -i console
    app.exec_()