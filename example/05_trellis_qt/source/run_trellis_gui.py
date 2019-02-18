#!/usr/bin/env python2
"""trellis dataflow gui.
Usage:
  run_trellis_gui [-vh]

Options:
  -h --help               Show this screen
  -v --verbose            Print debugging output
"""
# martin kielhorn 2019-02-14
# pip2 install --user PySide2
#  The scripts pyside2-lupdate, pyside2-rcc and pyside2-uic are installed in /home/martin/.local/bin
# example from https://pypi.org/project/Trellis/0.7a2/
#  pip install --user Trellis==0.7a2
# https://github.com/PEAK-Legacy/Trellis
# to install from github: pip2 install --user ez_setup
# wget http://peak.telecommunity.com/snapshots/Contextual-0.7a1.dev-r2695.tar.gz http://peak.telecommunity.com/snapshots/Trellis-0.7a3-dev-r2610.tar.gz 
#  pip2 install --user Contextual-0.7a1.dev-r2695.tar.gz
# i installed trellis by commenting out the contextual line in its setup.py and then extracting the egg file into ~/.local/lib/python2.7/site-packages/peak
from peak.events import trellis
import os
import sys
import docopt
import numpy as np
import pandas as pd
import pathlib
import re
import traceback
import PySide2.QtWidgets as qw
import PySide2.QtCore as qc
import PySide2.QtGui as qg
import select
# pip2 install systemd-python
from systemd import journal
from peak.events import trellis
args=docopt.docopt(__doc__, version="0.0.1")
if ( args["--verbose"] ):
    print(args)
class PandasTableModel(qc.QAbstractTableModel):
    def __init__(self, dataframe, parent=None):
        qc.QAbstractTableModel.__init__(self)
        self.dataframe=dataframe
    def flags(self, index):
        if ( not(index.isValid()) ):
            return None
        return ((qc.Qt.ItemIsEnabled) or (qc.Qt.ItemIsSelectable))
    def rowCount(self, *args, **kwargs):
        return len(self.dataframe.index)
    def columnCount(self, *args, **kwargs):
        return len(self.dataframe.columns)
    def headerData(self, section, orientation, role):
        if ( ((qc.Qt.DisplayRole)!=(role)) ):
            return None
        try:
            if ( ((qc.Qt.Horizontal)==(orientation)) ):
                return list(self.dataframe.columns)[section]
            if ( ((qc.Qt.Vertical)==(orientation)) ):
                return list(self.dataframe.index)[section]
        except IndexError:
            return None
    def data(self, index, role):
        if ( ((qc.Qt.DisplayRole)!=(role)) ):
            return None
        if ( not(index.isValid()) ):
            return None
        return str(self.dataframe.ix[index.row(),index.column()])
class PandasView(qw.QWidget):
    def __init__(self, df):
        print("PandasView.__init__")
        super(PandasView, self).__init__()
        self.model=PandasTableModel(df)
        self.table_view=qw.QTableView()
        self.table_view.setModel(self.model)
        self.main_layout=qw.QHBoxLayout()
        self.main_layout.addWidget(self.table_view)
        self.setLayout(self.main_layout)
#  https://stackoverflow.com/questions/26331116/reading-systemd-journal-from-python-script
j=journal.Reader()
j.log_level(journal.LOG_INFO)
j.seek_tail()
j.get_next()
res=[]
while (j.get_next()):
    for e in j:
        if ( (("")!=(e["MESSAGE"])) ):
            res.append({("time"):(e["__REALTIME_TIMESTAMP"]),("message"):(e["MESSAGE"])})
df=pd.DataFrame(res)
class MainWindow(qw.QMainWindow):
    def __init__(self, widget):
        super(MainWindow, self).__init__()
        self.setCentralWidget(widget)
    @qc.Slot()
    def exit_app(self, checked):
        sys.exit()
app=qw.QApplication(sys.argv)
widget=PandasView(df)
win=MainWindow(widget)
win.show()
sys.exit(app.exec_())