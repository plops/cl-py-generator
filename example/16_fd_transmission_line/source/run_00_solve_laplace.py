# 
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import scipy.sparse.linalg
import time
from numpy import *
from scipy.sparse import *
Nx=((12)+(5)+(13))
Ny=((5)+(15))
N=array((Nx,Ny,))
TM=((Nx)*(Ny))
SIG=zeros(N)
SIG[13:13+5,5]=1
GND=ones(N)
GND[1:((Nx)-(1)),1:((Ny)-(1))]=0
F=((SIG)+(GND))
V0=(1.0    )
vf=((((SIG)*(V0)))+(((GND)*(0))))
ERxx=ones(N)
ERxx[:,0:5]=6
ERyy=ERxx.copy()
SIG=diags((SIG.ravel(),), (0,))
GND=diags((GND.ravel(),), (0,))
F=diags((F.ravel(),), (0,))
ERxx=diags((ERxx.ravel(),), (0,))
ERyy=diags((ERyy.ravel(),), (0,))
vf=vf.ravel()
b=((F)*(vf))
L_=((F)+(((((eye(TM))-(F)))*(L))))