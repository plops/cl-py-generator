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
class CustomTableModel(qc.QAbstractTableModel):
    def __init__(self, dataframe):
        qc.QAbstractTableModel.__init__(self)
        self.dataframe(dataframe)
    def flags(self, index):
        return qc.Qt.ItemIsSelectable
    def rowCount(self, *args, **kwargs):
        return len(self.dataframe)
    def columnCount(self, *args, **kwargs):
        return len(self.dataframe.columns)
    def headerData(self, section, orientation, role=qc.Qt.DisplayRole):
        if ( ((((qc.Qt.Horizontal)==(orientation))) and (((qc.Qt.DisplayRole)==(role)))) ):
            return self.dataframe.columns[section]
    def data(self, index, role):
        
class Rectangle(trellis.Component):
    x=trellis.maintain(lambda self: ((self.x_min)+((((5.e-1))*(self.x_span)))), initially=0)
    x_span=trellis.maintain(lambda self: ((self.x_max)-(self.x_min)), initially=0)
    x_min=trellis.maintain(lambda self: ((self.x)-((((5.e-1))*(self.x_span)))), initially=0)
    x_max=trellis.maintain(lambda self: ((self.x)+((((5.e-1))*(self.x_span)))), initially=0)
    y=trellis.maintain(lambda self: ((self.y_min)+((((5.e-1))*(self.y_span)))), initially=0)
    y_span=trellis.maintain(lambda self: ((self.y_max)-(self.y_min)), initially=0)
    y_min=trellis.maintain(lambda self: ((self.y)-((((5.e-1))*(self.y_span)))), initially=0)
    y_max=trellis.maintain(lambda self: ((self.y)+((((5.e-1))*(self.y_span)))), initially=0)
    @trellis.perform
    def show_value(self):
        print("rect x={} x_span={} x_min={} x_max={} y={} y_span={} y_min={} y_max={}".format(self.x, self.x_span, self.x_min, self.x_max, self.y, self.y_span, self.y_min, self.y_max))
    @trellis.modifier
    def translate(self, r):
        self.x=((self.x)+(r[0]))
        self.y=((self.y)+(r[1]))
    @trellis.modifier
    def grow(self, r):
        self.x_span=((self.x_span)+(r[0]))
        self.y_span=((self.y_span)+(r[1]))
def make_rect_c(r=np.array([(0.0e+0), (0.0e+0)]), r_span=np.array([(1.e+0), (1.e+0)])):
    return Rectangle(x=r[0], y=r[1], x_span=r_span[0], y_span=r_span[1])
def make_rect(min=np.array([(0.0e+0), (0.0e+0)]), max=np.array([(1.e+0), (1.e+0)])):
    return Rectangle(x_min=min[0], y_min=min[1], x_max=max[0], y_max=max[1])
r=make_rect_c()
#  https://stackoverflow.com/questions/26331116/reading-systemd-journal-from-python-script
j=journal.Reader()
j.log_level(journal.LOG_INFO)
j.seek_tail()
j.get_next()
while (j.get_next()):
    for e in j:
        if ( (("")!=(e["MESSAGE"])) ):
            print("{} {}".format(e["__REALTIME_TIMESTAMP"], e["MESSAGE"]))