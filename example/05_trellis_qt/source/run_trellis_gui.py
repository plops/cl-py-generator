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
from peak.events import trellis
args=docopt.docopt(__doc__, version="0.0.1")
if ( args["--verbose"] ):
    print(args)
class TempConverter(trellis.Component):
    F=trellis.maintain(lambda self: ((32)+((((1.799999952316284e+0))*(self.C)))), initially=32)
    C=trellis.maintain(lambda self: ((((self.F)-(32)))/((1.799999952316284e+0))), initially=0)
    @trellis.perform
    def show_values(self):
        print("Celsius    .. {}".format(self.C))
        print("Fahrenheit .. {}".format(self.F))
tc=TempConverter(C=float(args["-c"]))
tc.F=32
tc.C=-40