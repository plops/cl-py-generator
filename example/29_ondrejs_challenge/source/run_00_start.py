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
import libtiff
_code_git_version="ca25841473f7cf53ca092bda599ff3df7004a9ce"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py"
_code_generation_time="19:03:47 of Thursday, 2020-11-26 (GMT+1)"
fn="./supplementary_materials/photos/RIMG1832-1.tiff"
# pip3 install --user libtiff
tif=libtiff.TIFF.open(fn)
im=tif.read_image()