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
_code_git_version="2c5b6f9fdf704d60fa02fa8c94e2791d8dc5c523"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py"
_code_generation_time="19:37:54 of Thursday, 2020-11-26 (GMT+1)"
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
    pl=(2,2,)
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
    g=np.real(ik)
    plt.imshow(((((g)<(0)))*(g)*(-1)), cmap="gray")
    ax=plt.subplot2grid(pl, (1,1,))
    plt.title("inverse fft (pos)")
    plt.imshow(((((0)<(g)))*(g)), cmap="gray")
    plt.savefig("/dev/shm/{}.png".format(fn.stem))