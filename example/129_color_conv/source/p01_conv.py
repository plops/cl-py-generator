#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2 as cv
import pandas as pd
import lmfit
 
def bgr_to_ycrcb_model(bgr, coeff_matrix0, coeff_matrix1, coeff_matrix2, coeff_matrix3, coeff_matrix4, coeff_matrix5, coeff_matrix6, coeff_matrix7, coeff_matrix8, offsets9, offsets10, offsets11, gamma_bgr12):
    """Model for BGR to Ycrcb color transformation with adjustable parameters.

  Args:
    bgr: A numpy array of shape (3,) representing B, G, R values.
    coeff_matrix: A 3x3 numpy array representing the transformation coefficients.
    offsets: A numpy array of shape (3,) representing the offsets for each channel.
    gains: A numpy array of shape (3,) representing the gains for each channel.
    gamma: The gamma correction value.

  Returns:
    A numpy array of shape (3,) representing the Y, Cb, Cr values."""
    params=np.array([coeff_matrix0, coeff_matrix1, coeff_matrix2, coeff_matrix3, coeff_matrix4, coeff_matrix5, coeff_matrix6, coeff_matrix7, coeff_matrix8, offsets9, offsets10, offsets11, gamma_bgr12])
    coeff_matrix=params[0:9].reshape((3,3,))
    offsets=params[9:12].reshape((3,))
    gamma_bgr=params[12:13].reshape((1,))
    bgr_gamma=np.power(((bgr)/((255.    ))), (((1.0    ))/(gamma_bgr)))
    ycrcb=((np.dot(bgr_gamma, coeff_matrix.T))+(offsets))
    return ycrcb
 
 
num_colors=1000
bgr_colors=np.random.randint(0, 256, size=(num_colors,3,))
res=[]
for bgr in bgr_colors:
    ycrcb=cv.cvtColor(np.uint8([[bgr]]), cv.COLOR_BGR2YCrCb)[0,0]
    res.append(dict(B=bgr[0], G=bgr[1], R=bgr[2], Y=ycrcb[0], Cr=ycrcb[1], Cb=ycrcb[2]))
 
df=pd.DataFrame(res)
"""Fits the BGR to Ycrcb model to data using lmfit.

  Args:
    df: A pandas DataFrame with columns 'B', 'G', 'R', 'Y', 'Cb', 'Cr'.

  Returns:
    An lmfit ModelResult object containing the fitted parameters."""
model=lmfit.Model(bgr_to_ycrcb_model)
params=lmfit.Parameters()
params.add("coeff_matrix0", value=1, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix1", value=0, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix2", value=0, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix3", value=0, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix4", value=1, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix5", value=0, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix6", value=0, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix7", value=0, vary=True, min=-np.inf, max=np.inf)
params.add("coeff_matrix8", value=1, vary=True, min=-np.inf, max=np.inf)
params.add("offsets9", value=0, vary=False, min=-np.inf, max=np.inf)
params.add("offsets10", value=128, vary=False, min=-np.inf, max=np.inf)
params.add("offsets11", value=128, vary=False, min=-np.inf, max=np.inf)
params.add("gamma_bgr12", value=1, vary=False, min=(0.10    ), max=3)
result=model.fit(df[["Y", "Cb", "Cr"]].values, params, bgr=df[["B", "G", "R"]].values)
print(result.fit_report())