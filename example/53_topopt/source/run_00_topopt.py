import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import datetime
import time
_code_git_version="c4e9cfdde50d6135bed970993b5a0acac807556a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="23:28:14 of Thursday, 2024-05-09 (GMT+1)"
nelx=180
nely=60
volfrac=(0.40    )
rmin=(5.40    )
penal=(3.0    )
ft=1
def lk():
    E=1
    nu=(0.30    )
    k=np.array([(((0.50    ))-(((nu)/(6)))), (((0.1250    ))+(((nu)/(8)))), (((-0.250    ))-(((nu)/(12)))), (((-0.1250    ))+(((3)*(((nu)/(8)))))), (((-0.250    ))+(((nu)/(12)))), (((-0.1250    ))-(((nu)/(8)))), ((nu)/(6)), ((1/8)-(((3)*(((nu)/(8))))))])
    KE=((((E)/(((1)-(((nu)**(2)))))))*(np.array([k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2], k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1], k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4], k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3], k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6], k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5], k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]])))
    return KE
def oc(nelx = , nely = , x = , volfrac = , dc = , dv = , g = ):
    l1=0
    l2=(1.0e+9)
    move=(0.20    )
    xnew=np.zeros(((nelx)*(nely)))
    while ((((1.00e-3))<(((((l2)-(l1)))/(((l1)+(l2))))))):
        lmid=(((0.50    ))*(((l2)+(l1))))
        xnew[:]=np.maximum((0.    ), np.maximum(((x)-(move)), np.minimum((1.0    ), np.minimum(((x)+(move)), ((x)*(np.sqrt(((-dc)/(((dv)*(lmid)))))))))))
        gt=((g)+(np.sum(((dv)*(((xnew)-(x)))))))
    if ( ((0)<(gt)) ):
        l1=lmid
    else:
        l2=lmid
    return (xnew,gt,)
Emin=(1.00e-9)
Emax=(1.0    )
ndof=((2)*(((nelx)+(1)))*(((nely)+(1))))
x=((volfrac)*(np.ones(((nely)*(nelx)), dtype=float)))
xold=x.copy()
xPhys=x.copy()
g=0
dc=np.zeros((nely,nelx,), dtype=float)
KE=lk()
edofMat=np.zeros([((nelx)*(nely)), 8], dtype=int)
for elx in range(nelx):
    for ely in range(nely):
        el=((ely)+(((elx)*(nely))))
        n1=((((((nely)+(1)))*(elx)))+(ely))
        n2=((((((nely)+(1)))*(((elx)+(1)))))+(ely))
        edofMat[el,:]=np.array([((((2)*(n1)))+(2)), ((((2)*(n1)))+(3)), ((((2)*(n2)))+(2)), ((((2)*(n2)))+(3)), ((((2)*(n2)))+(0)), ((((2)*(n2)))+(1)), ((((2)*(n1)))+(0)), ((((2)*(n1)))+(1))])
iK=np.kron(edofMat, np.ones((8,1,))).flatten()
jK=np.kron(edofMat, np.ones((1,8,))).flatten()
nfilter=int(((nelx)*(nely)*(((((1)+(((2)*(((np.ceil(rmin))-(1)))))))**(2)))))
print("nfilter={}".format(nfilter))
iH=np.zeros(nfilter)
jH=np.zeros(nfilter)
sH=np.zeros(nfilter)
cc=0
for i in range(nelx):
    for j in range(nely):
        row=((((i)*(nely)))+(j))
        crmin=np.ceil(rmin)
        crmin1=((crmin)-(1))
        kk1=int(np.maximum(0, ((i)-(crmin1))))
        kk2=int(np.maximum(nelx, ((i)+(crmin))))
        ll1=int(np.maximum(0, ((j)-(crmin1))))
        ll2=int(np.maximum(nely, ((j)+(crmin))))
        for k in range(kk1, kk2):
            for l in range(ll1, ll2):
                col=((l)+(((k)*(nely))))
                fac=((rmin)-(np.hypot(((i)-(k)), ((j)-(l)))))
                e
                iH[cc]=row
                jH[cc]=col
                sH[cc]=np.maximum((0.    ), fac)
                cc=((cc)+(1))