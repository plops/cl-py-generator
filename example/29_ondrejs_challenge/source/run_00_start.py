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
_code_git_version="f7bfce24659ae17f2f3e02c31377c401ef0eac70"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/28_dask_test/source/run_00_start.py"
_code_generation_time="19:15:40 of Thursday, 2020-11-26 (GMT+1)"
fns=list(pathlib.Path("./supplementary_materials/photos/").glob("*.tiff"))
for fn in fns:
    # pip3 install --user libtiff
    tif=libtiff.TIFF.open(fn)
    im=tif.read_image()
    # (256,512) float64
# assume it is real and imag next to each other
    k=((((1j)*(im[:,:256])))+((((1.0    ))*(im[:,256:]))))
    # a dot is in the middle, so i will need fftshift
    sk=np.fft.ifftshift(k)
    ik=np.fft.ifft(sk)
    plt.close("all")
    plt.title(fn)
    plt.imshow(np.abs(ik))
    plt.savefig("/dev/shm/{}.png".format(fn.stem))