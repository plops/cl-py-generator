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
import pathlib
import re
import traceback
import PySide2.QtWidgets as qw
import PySide2.QtCore as qc
from peak.events import trellis
args=docopt.docopt(__doc__, version="0.0.1")
if ( args["--verbose"] ):
    print(args)
class Rectangle(trellis.Component):
    x=trellis.maintain(lambda self: ((self.x_min)+((((5.e-1))*(self.x_span)))), initially=0)
    x_span=trellis.maintain(lambda self: ((self.x_max)-(self.x_min)), initially=0)
    x_min=trellis.maintain(lambda self: ((self.x)-((((5.e-1))*(self.x_span)))), initially=0)
    x_max=trellis.maintain(lambda self: ((self.x)+((((5.e-1))*(self.x_span)))), initially=0)
    @trellis.perform
    def show_value(self):
        print("rect {}-{} {}:{}".format(self.x, self.x_span, self.x_min, self.x_max))
r=Rectangle(x_min=1, x_max=10)
r2=Rectangle(x=3, x_span=2)
r3=Rectangle(x=3, x_min=1)