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
import dask.array
_code_git_version="4bf9df4b615ba49978a3a5bf5ffeec8ef6f18414"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py"
_code_generation_time="21:49:34 of Sunday, 2020-11-01 (GMT+1)"
client=dask.distributed.Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit="2GB")