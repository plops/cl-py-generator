import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
plt.ion()
x=np.linspace(0, (2.e+0), 30)
y=np.sin(x)
plt.plot(x, y)
plt.grid()