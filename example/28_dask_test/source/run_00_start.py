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
import dask.distributed
import dask.array as da
_code_git_version="07f588cb83174cbac2f134044004c5112935bc67"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py"
_code_generation_time="21:56:23 of Sunday, 2020-11-01 (GMT+1)"
client=dask.distributed.Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit="2GB")
x=da.random.random((10000,10000,), chunks=(1000,1000,))
y=((x)+(x.T))
z=y[::2,5000:].mean(axis=1)