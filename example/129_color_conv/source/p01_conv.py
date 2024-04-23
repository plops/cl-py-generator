#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2 as cv
import pandas as pd
import lmfit
start_time=time.time()
debug=True
_code_git_version="294bbeae1d9553a5aac51ccb18edd3cc0bb8191d"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/129_color_conv/source/"
_code_generation_time="23:28:42 of Tuesday, 2024-04-23 (GMT+1)"
bgr=np.array([10, 120, 13])
cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)
def bgr_to_ycbcr_model(bgr, coeff_matrix, offsets, gains, gamma):
    """Model for BGR to YCbCr color transformation with adjustable parameters.

  Args:
    bgr: A numpy array of shape (3,) representing B, G, R values.
    coeff_matrix: A 3x3 numpy array representing the transformation coefficients.
    offsets: A numpy array of shape (3,) representing the offsets for each channel.
    gains: A numpy array of shape (3,) representing the gains for each channel.
    gamma: The gamma correction value.

  Returns:
    A numpy array of shape (3,) representing the Y, Cb, Cr values."""
    bgr_gamma=np.power(((bgr)/((255.    ))), (((1.0    ))/(gamma)))
    ycbcr=((((np.dot(coeff_matrix, bgr_gamma))*(gains)))+(offsets))
    return ycbcr
def fit_bgr_to_ycbcr(df):
    """Fits the BGR to YCbCr model to data using lmfit.

  Args:
    df: A pandas DataFrame with columns 'B', 'G', 'R', 'Y', 'Cb', 'Cr'.

  Returns:
    An lmfit ModelResult object containing the fitted parameters."""
    model=lmfit.Model(bgr_to_ycbcr_model)
    params=model.make_params(coeff_matrix=np.identity(3), offsets=np.zeros(3), gains=np.ones(3), gamma=(2.20    ))
    result=model.fit(df[["Y", "Cb", "Cr"]].values, params, bgr=df[["B", "G", "R"]].values)
    return result