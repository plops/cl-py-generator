# pip3 install --user QCustomPlot2
# change gui font size in linux: xrandr --output HDMI-0 --dpi 55
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QAbstractTableModel, Qt
import numpy as np
import pandas as pd
import pathlib
# %%
output_path="/dev/shm"
_code_git_version="e084c672689298af3167bacb3fd1a6f9e0edb086"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/17_qt_customplot/source/run_00_plot.py"
_code_generation_time="09:18:38 of Saturday, 2020-05-09 (GMT+1)"
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
layout.addWidget(table)
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
model=DataFrameModel(df)
table.setModel(model)
table.selectionModel().selectionChanged.connect(selectionChanged)
def run0():
    app.exec_()