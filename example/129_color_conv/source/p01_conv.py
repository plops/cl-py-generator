#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2 as cv
import pandas as pd
import lmfit
start_time=time.time()
debug=True
_code_git_version="fbe9ab259e457c53f1559ae626c647d1da67da16"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/129_color_conv/source/"
_code_generation_time="23:35:40 of Tuesday, 2024-04-23 (GMT+1)"
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
num_colors=100
bgr_colors=np.random.randint(0, 256, size=(num_colors,3,))
res=[]
for bgr in bgr_colors:
    ycbcr=cv.cvtColor(np.uint8([[bgr]]), cv.COLOR_BGR2YCrCb)[0,0]
    res.append(dict(B=bgr[0], G=bgr[1], R=bgr[2], Y=ycbcr[0], Cb=ycbcr[1], Cr=ycbcr[2]))
df=pd.DataFrame(res)
result=fit_bgr_to_ycbcr(df)
print(result.fit_report())