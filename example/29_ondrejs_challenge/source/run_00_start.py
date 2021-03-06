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
import libtiff
_code_git_version="ade9e676d7bb9ac6beaec961e2daf535f4a10784"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="20:00:58 of Thursday, 2020-11-26 (GMT+1)"
fns=list(pathlib.Path("./supplementary_materials/").glob("**/*.tif*"))
for fn in fns:
    print(fn)
    # pip3 install --user libtiff
    tif=libtiff.TIFF.open(fn)
    im=tif.read_image()
    # (256,512) float64
# assume it is real and imag next to each other
    k=(((((1.0    ))*(im[:,:256])))+(((1j)*(im[:,256:]))))
    # a dot is in the middle, so i will need fftshift
    sk=np.fft.ifftshift(k)
    ik=np.fft.ifft2(sk)
    pl=(2,3,)
    plt.close("all")
    plt.figure(0, (16,9,))
    ax=plt.subplot2grid(pl, (0,0,))
    plt.title(fn)
    plt.imshow(np.log(np.abs(k)))
    ax=plt.subplot2grid(pl, (1,0,))
    plt.title("fftshift")
    plt.imshow(np.log(np.abs(sk)))
    ax=plt.subplot2grid(pl, (0,1,))
    g=np.real(ik)
    highs=((127)+(((128)-(((((g)<(0)))*(g)*(-1))))))
    highs_mask=((g)<(0))
    mi=np.min(highs)
    ma=np.max(highs)
    plt.title("inverse fft (neg) {}..{}".format(int(mi), int(ma)))
    # this is the bright part of the image
    plt.imshow(highs, cmap="gray")
    ax=plt.subplot2grid(pl, (1,1,))
    lows=((((0)<(g)))*(g))
    lows_mask=((0)<(g))
    mi=np.min(lows)
    ma=np.max(lows)
    plt.title("inverse fft (pos) {}..{}".format(int(mi), int(ma)))
    plt.imshow(lows, cmap="gray")
    ax=plt.subplot2grid(pl, (1,2,))
    plt.title("inverse fft (bright and shadow together)")
    plt.imshow(((((highs_mask)*(highs)))+(((lows_mask)*(lows)))), cmap="gray")
    plt.savefig("/dev/shm/{}.png".format(fn.stem))