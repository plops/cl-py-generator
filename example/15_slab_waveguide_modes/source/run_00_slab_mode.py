# 
# export LANG=en_US.utf8
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import scipy.sparse.linalg
import time
# simulation parameters
lam0=(1.0    )
n1=(2.0    )
n2=(1.0    )
a=((3)*(lam0))
b=((5)*(lam0))
dx=((lam0)/(20))
M=5
print("lam0={} n1={} n2={} a={} b={} dx={} M={}".format(lam0, n1, n2, a, b, dx, M))
# compute grid
Sx=((a)+(((2)*(b))))
Nx=int(np.ceil(((Sx)/(dx))))
Sx=((Nx)*(dx))
xa=((dx)*(np.linspace((0.50    ), ((Nx)-((0.50    ))), ((Nx)+(-1)))))
xa=((xa)-(np.mean(xa)))
# start and stop indices (centered in grid)
nx=int(np.round(((a)/(dx))))
nx1=int(np.round(((((Nx)-(nx)))/(2))))
nx2=((nx1)+(nx)+(-1))
N=np.ones(Nx)
N[0:nx1:2]=n2
N[nx1:nx2]=n1
N[nx2+1:Nx]=n2
print("Sx={} Nx={} nx={} nx1={} nx2={}".format(Sx, Nx, nx, nx1, nx2))
# perform fd analysis
k0=((((2)*(np.pi)))/(lam0))
DX2=(((((1.0    ))/(((((k0)*(dx)))**(2)))))*(scipy.sparse.diags((((1)*(np.ones(((Nx)-(1))))),((-2)*(np.ones(((Nx)-(0))))),((1)*(np.ones(((Nx)-(1))))),), (-1,0,1,))))
N2=scipy.sparse.diags((((N)**(2)),), (0,))
A=((DX2)+(N2)).astype(np.float32)
Afull=A.toarray()
start=time.clock()
(D,V,)=np.linalg.eig(Afull)
end=time.clock()
duration_full=((end)-(start))
print("duration_full={}".format(duration_full))
NEFF=np.real(np.sqrt(((0j)+(D))))
start=time.clock()
(Ds,Vs,)=scipy.sparse.linalg.eigs(A, M, which="LR")
end=time.clock()
duration_sparse=((end)-(start))
print("duration_sparse={}".format(duration_sparse))
NEFFs=np.sqrt(Ds)
# plot
ind=np.flip(np.argsort(NEFF))
NEFF1=np.flip(np.sort(NEFF))
V1=V[:,ind]
# substrate
plt.axhline(y=((-b)-(((a)/(2)))))
plt.axhline(y=((b)+(((a)/(2)))))
# core
plt.axhline(y=((-a)/(2)))
plt.axhline(y=((a)/(2)))
for m in range(M):
    x0=((2)*(m))
    y0=(((0.50    ))*(((a)+(b))))
    x=((x0)+(((3)*(V1[:,m]))))
    xs=((x0)+(((3)*(Vs[:,m]))))
    y=np.linspace(((-b)-(((a)/(2)))), ((b)+(((a)/(2)))), Nx)
    plt.plot(x, y)
    plt.plot(xs, y)
    plt.text(x0, y0, "mode={}\n{:6.4f}\n{:6.4f}".format(m, NEFF1[m], NEFFs[m]))