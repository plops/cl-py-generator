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
_code_git_version="0e1b23217ae5e1838bcb303d2191b9a234908a3d"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/26_copernicus_xsd/source/run_00_load_data.py"
_code_generation_time="07:21:11 of Wednesday, 2020-10-28 (GMT+1)"
df=pd.read_csv(pathlib.Path("~/fanspeed_log.csv"))
df["timestamp"]=pd.to_datetime(df.time)
plt.plot(df.timestamp, df.fan)
plt.grid()