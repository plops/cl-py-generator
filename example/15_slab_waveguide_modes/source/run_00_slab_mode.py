# 
# export LANG=en_US.utf8
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import scipy.sparse.linalg
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
A=((DX2)+(N2))
(D,V,)=np.linalg.eig(A.toarray())
NEFF=np.real(np.sqrt(((0j)+(D))))
# plot
ind=np.flip(np.argsort(NEFF))
NEFF1=np.flip(np.sort(NEFF))
V1=V[:,ind]
for m in range(M):
    plt.plot(V1[:,m])