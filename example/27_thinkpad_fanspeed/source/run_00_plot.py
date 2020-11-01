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
_code_git_version="4bf9df4b615ba49978a3a5bf5ffeec8ef6f18414"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/26_copernicus_xsd/source/run_00_load_data.py"
_code_generation_time="21:48:21 of Sunday, 2020-11-01 (GMT+1)"
df=pd.read_csv(sys.argv[1], sep=r"""speed:\s+""", skipinitialspace=True)
df.columns=["time", "fan"]
df=df.iloc[::1]
df["timestamp"]=pd.to_datetime(df.time)
plt.plot(df.timestamp, df.fan)
plt.grid()
plt.title("/proc/acpi/ibm/fan")