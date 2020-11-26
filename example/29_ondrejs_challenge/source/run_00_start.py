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
_code_git_version="e86a8370055c34c4649e2d35cbe3794328fcae50"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py"
_code_generation_time="19:41:25 of Thursday, 2020-11-26 (GMT+1)"
fns=list(pathlib.Path("./supplementary_materials/video").glob("*.tif"))
for fn in [fns[1]]:
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
    plt.title("inverse fft (neg)")
    # this is the bright part of the image
    g=np.real(ik)
    highs=np.abs(((((g)<(0)))*(g)*(-1)))
    plt.imshow(highs, cmap="gray")
    ax=plt.subplot2grid(pl, (1,1,))
    plt.title("inverse fft (pos)")
    lows=((((0)<(g)))*(g))
    plt.imshow(lows, cmap="gray")
    ax=plt.subplot2grid(pl, (1,2,))
    plt.title("inverse fft (bright and shadow together)")
    plt.imshow(((highs)+(lows)), cmap="gray")
    plt.savefig("/dev/shm/{}.png".format(fn.stem))