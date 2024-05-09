import matplotlib
import matplotlib.pyplot as plt
import xarray.plot as xrp
plt.ion()
font={("size"):("6")}
matplotlib.rc("font", **font)
import pandas as pd
import xarray as xr
import xarray.plot as xrp
import numpy as np
def osa_index_nl_to_j(n, l):
    return ((((((n)*(((n)+(2)))))+(l)))//(2))
def osa_index_j_to_nl(j):
    lut=[[0, 0], [1, -1], [1, 1], [2, -2], [2, 0], [2, 2], [3, -3], [3, -1], [3, 1], [3, 3], [4, -4], [4, -2], [4, 0], [4, 2], [4, 4], [5, -5], [5, -3], [5, -1], [5, 1], [5, 3], [5, 5], [6, -6], [6, -4], [6, -2], [6, 0], [6, 2], [6, 4], [6, 6], [7, -7], [7, -5], [7, -3], [7, -1], [7, 1], [7, 3], [7, 5], [7, 7], [8, -8], [8, -6], [8, -4], [8, -2], [8, 0], [8, 2], [8, 4], [8, 6], [8, 8], [9, -9], [9, -7], [9, -5], [9, -3], [9, -1], [9, 1], [9, 3], [9, 5], [9, 7], [9, 9], [10, -10], [10, -8], [10, -6], [10, -4], [10, -2], [10, 0], [10, 2], [10, 4], [10, 6], [10, 8], [10, 10]]
    return lut[j]
def fringe_index_nl_to_j(n, l):
    return ((((((1)+(((((n)+(np.abs(l))))/(2)))))**(2)))+(((-2)*(np.abs(l))))+(((np.sign(l))*(((((1)-(np.sign(l))))/(2))))))
def zernike(rho, phi, n = 0, l = 0):
    # n in [0 .. 10], l in [-10 .. 10]
    arg=((phi)*(abs(l)))
    if ( ((l)<(0)) ):
        azi=np.sin(arg)
    else:
        azi=np.cos(arg)
    # polynomial coefficients in order of increasing degree, (1 2 3) is 1 + 2*x + 3*x**2
    coef=[[1], [0, 1], [0, 1], [0, 0, 1], [-1, 0, 2], [0, 0, 1], [0, 0, 0, 1], [0, -2, 0, 3], [0, -2, 0, 3], [0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, -3, 0, 4], [1, 0, -6, 0, 6], [0, 0, -3, 0, 4], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, -4, 0, 5], [0, 3, 0, -12, 0, 10], [0, 3, 0, -12, 0, 10], [0, 0, 0, -4, 0, 5], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -5, 0, 6], [0, 0, 6, 0, -20, 0, 15], [-1, 0, 12, 0, -30, 0, 20], [0, 0, 6, 0, -20, 0, 15], [0, 0, 0, 0, -5, 0, 6], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, -6, 0, 7], [0, 0, 0, 10, 0, -30, 0, 21], [0, -4, 0, 30, 0, -60, 0, 35], [0, -4, 0, 30, 0, -60, 0, 35], [0, 0, 0, 10, 0, -30, 0, 21], [0, 0, 0, 0, 0, -6, 0, 7], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -7, 0, 8], [0, 0, 0, 0, 15, 0, -42, 0, 28], [0, 0, -10, 0, 60, 0, -105, 0, 56], [1, 0, -20, 0, 90, 0, -140, 0, 70], [0, 0, -10, 0, 60, 0, -105, 0, 56], [0, 0, 0, 0, 15, 0, -42, 0, 28], [0, 0, 0, 0, 0, 0, -7, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, -8, 0, 9], [0, 0, 0, 0, 0, 21, 0, -56, 0, 36], [0, 0, 0, -20, 0, 105, 0, -168, 0, 84], [0, 5, 0, -60, 0, 210, 0, -280, 0, 126], [0, 5, 0, -60, 0, 210, 0, -280, 0, 126], [0, 0, 0, -20, 0, 105, 0, -168, 0, 84], [0, 0, 0, 0, 0, 21, 0, -56, 0, 36], [0, 0, 0, 0, 0, 0, 0, -8, 0, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, -9, 0, 10], [0, 0, 0, 0, 0, 0, 28, 0, -72, 0, 45], [0, 0, 0, 0, -35, 0, 168, 0, -252, 0, 120], [0, 0, 15, 0, -140, 0, 420, 0, -504, 0, 210], [-1, 0, 30, 0, -210, 0, 560, 0, -630, 0, 252], [0, 0, 15, 0, -140, 0, 420, 0, -504, 0, 210], [0, 0, 0, 0, -35, 0, 168, 0, -252, 0, 120], [0, 0, 0, 0, 0, 0, 28, 0, -72, 0, 45], [0, 0, 0, 0, 0, 0, 0, 0, -9, 0, 10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    osa_index=osa_index_nl_to_j(n, l)
    radial=np.polynomial.polynomial.polyval(rho, coef[osa_index])
    mask=np.where(((rho)<(1)), (1.0    ), np.nan)
    return ((mask)*(radial)*(azi))
x=np.linspace(-1, 1, 128)
y=np.linspace(-1, 1, 128)
rho=np.hypot(x, y[:,np.newaxis])
phi=np.arctan2(y[:,np.newaxis], x)
zval=zernike(rho, phi, n=1, l=1)
xs=xr.DataArray(data=zval, coords=[y, x], dims=["y", "x"])
def xr_zernike(n = 0, l = 0, x = np.linspace(-1, 1, 64), y = np.linspace(-1, 1, 64)):
    """return xarray with evaluated zernike polynomial"""
    rho=np.hypot(x, y[:,np.newaxis])
    phi=np.arctan2(y[:,np.newaxis], x)
    zval=zernike(rho, phi, n=n, l=l)
    xs=xr.DataArray(data=zval, coords=[y, x], dims=["y", "x"])
    return xs
plt.figure(figsize=[19, 10])
zernike_names=pd.DataFrame({("n"):([0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8]),("l"):([0, 1, 0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 7, 0]),("name"):(["piston", "tilt", "defocus", "primary-astigmatism", "primary-coma", "trefoil", "primary-spherical", "secondary-astigmatism", "tetrafoil", "secondary-coma", "secondary-trefoil", "pentafoil", "secondary-spherical", "tertiary-astigmatism", "secondary-trefoil", "hexafoil", "tertiary-coma", "tertiary-trefoil", "secondary-pentafoil", "heptafoil", "tretiary-spherical"])})
for j in range(0, ((4)*(9))):
    ax=plt.subplot(4, 9, ((j)+(1)))
    ax.set_aspect("equal")
    (n,l,)=osa_index_j_to_nl(j)
    xs=xr_zernike(n, l)
    xs.plot(vmin=-1, vmax=1, add_colorbar=False)
    cs=xrp.contour(xs, colors="k")
    plt.clabel(cs, inline=True)
    plt.grid()
    lookup=zernike_names[((((zernike_names.n)==(n))) & (((zernike_names.l)==(np.abs(l)))))]
    if ( ((1)==(len(lookup))) ):
        plt.title("j={} n={} l={}\n{}".format(j, n, l, lookup.name.item()))
    else:
        plt.title("j={} n={} l={}".format(j, n, l))
    if ( not(((((j)%(9)))==(0))) ):
        plt.ylabel(None)
    if ( not(((((((j)//(9)))%(4)))==(3))) ):
        plt.xlabel(None)
plt.tight_layout()
plt.savefig("zernikes.png")