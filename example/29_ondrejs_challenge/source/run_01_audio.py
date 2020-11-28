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
import scipy.io.wavfile
import libtiff
_code_git_version="11d174e8861127a6b334e9795795573452655401"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="14:01:13 of Saturday, 2020-11-28 (GMT+1)"
fn="supplementary_materials/zdravice.wav"
rate, a=scipy.io.wavfile.read(fn)
# => 11025, array with 242550x2 elements, int16
z=((a[:,0])+(((1j)*(a[:,1]))))
zs=np.fft.fftshift(z)
k=np.fft.ifft(zs)
kr=np.real(k)
ki=np.imag(k)
scipy.io.wavfile.write("/dev/shm/r.wav", rate, kr)