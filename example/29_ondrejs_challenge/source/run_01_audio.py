# %% imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.ion()
import sys
import time
import pathlib
import numpy as np
import pandas as pd
import scipy.io.wavfile
import libtiff
_code_git_version="ac8a07b2c1d68aed56bddb6880a4c679594d001b"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="13:54:38 of Saturday, 2020-11-28 (GMT+1)"
fn="supplementary_materials/zdravice.wav"
a=scipy.io.wavfile.read(fn)