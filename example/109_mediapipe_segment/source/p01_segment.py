#!/usr/bin/env python3
# python3 -m pip install --user mediapipe
# wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
# 16 MB download
import time
import numpy as np
import math
import mediapipe as mp
import mss
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
start_time=time.time()
debug=True
_code_git_version="6d78d2089a17e64fe3c5c8bf7135fca4429d846e"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/109_mediapipe_segment/source/"
_code_generation_time="09:32:26 of Saturday, 2023-06-10 (GMT+1)"
BaseOptions=mp.tasks.BaseOptions
ImageSegmenter=mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions=mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode=mp.tasks.vision.RunningMode
def print_result(result: list[mp.Image], output_image: mp.Image, timestamp_ms: int):
    print("segmented mask size: {}".format(len(result)))
DESIRED_HEIGHT=256
DESIRED_WIDTH=256
def resize(image):
    h, w=image.shape[:2]
    if ( ((h)<(w)) ):
        img=cv.resize(image, (DESIRED_WIDTH,math.floor(((h)/(((w)/(DESIRED_WIDTH))))),))
    else:
        img=cv.resize(image, (math.floor(((w)/(((h)/(DESIRED_HEIGHT))))),DESIRED_HEIGHT,))

bo = mp.tasks.BaseOptions()

options=ImageSegmenterOptions(base_options=BaseOptions(model_asset_path="selfie_multiclass_256x256.tflite"
                                                    ))
with ImageSegmenter.create_from_options(options) as segmenter:
    with mss.mss() as sct:
        grb=sct.grab(dict(top=160, left=0, width=960, height=540))
        img=np.array(grb.pixels)
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        segmentation_result=segmenter.segment(mp_image)
        category_mask=segmentation_result.category_mask
        print("{} result segmentation_result[0]={}".format(((time.time())-(start_time)), segmentation_result[0]))