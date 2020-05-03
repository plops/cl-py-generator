# 
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import scipy.sparse.linalg
import time
from numpy import *
Nx=((12)+(5)+(13))
Ny=((5)+(15))
RES=np.array((Nx,Ny,))
SIG=np.zeros(RES)
SIG[13:13+5,5]=1
GND=np.ones(RES)
GND[1:((Nx)-(1)),1:((Ny)-(1))]=0
ERxx=np.ones(RES)
ERxx[:,0:5]=6
ERyy=ERxx.copy()