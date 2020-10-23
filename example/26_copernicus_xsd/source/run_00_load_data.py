# %% imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.ion()
import sys
import time
import pathlib
import numpy as np
import serial
import pandas as pd
from generated import *
_code_git_version="66e6811232d925e0f4a0f5c8f0da62125ed58de6"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/26_copernicus_xsd/source/run_00_load_data.py"
_code_generation_time="19:41:48 of Friday, 2020-10-23 (GMT+1)"
fns=list(pathlib.Path("./").glob("S1*RAW*.SAFE/*.dat"))