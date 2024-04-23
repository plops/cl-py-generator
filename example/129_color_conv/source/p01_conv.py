#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2 as cv
import lmfit
start_time=time.time()
debug=True
_code_git_version="bcf9dd8229907f9ffd7c856f6c05783c542bb493"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/129_color_conv/source/"
_code_generation_time="23:16:48 of Tuesday, 2024-04-23 (GMT+1)"
bgr=np.array([10, 120, 13])
cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)