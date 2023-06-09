#!/usr/bin/env python3
# python3 -m pip install --user mediapipe
# wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
# 16 MB download
import time
import mediapipe as mp
import mss
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
start_time=time.time()
debug=True
_code_git_version="793da2a1b7f70cd0827f9ca88035b3bc9dbdbee0"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/109_mediapipe_segment/source/"
_code_generation_time="22:57:57 of Friday, 2023-06-09 (GMT+1)"
BaseOptions=mp.tasks.BaseOptions
ImageSegmenter=mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions=mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode=mp.tasks.vision.RunningMode
def print_result(result: list[mp.Image], output_image: mp.Image, timestamp_ms: int):
    print("segmented mask size: {}".format(len(result)))
options=ImageSegmenterOptions(base_options=BaseOptions(model_asset_path="selfie_multiclass_256x256.tflite", output_category_mask=True, result_callback=lambda : print("{} result ".format(((time.time())-(start_time))))))
with ImageSegmenter.create_from_options(options) as segmenter:
    with mss.mss() as sct:
        img=np.array(sct.grab(dict(top=160, left=0, width=960, height=540)))
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        segmenter.segment_async(mp_image)