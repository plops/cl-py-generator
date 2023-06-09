#!/usr/bin/env python3
# python3 -m pip install --user mediapipe
# wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
# 16 MB download
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
start_time=time.time()
debug=True
_code_git_version="0925ff46e412e2bbc5a1d9c8fd83b7c70fce047d"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/109_mediapipe_segment/source/"
_code_generation_time="22:45:49 of Friday, 2023-06-09 (GMT+1)"
BaseOptions=mp.tasks.BaseOptions
ImageSegmenter=mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions=mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode=mp.tasks.vision.RunningMode
def print_result(result: List[Image], output_image: Image, timestamp_ms: int):
    print("segmented mask size: {}".format(len(result)))
options=ImageSegmenterOptions(base_options=BaseOptions(model_asset_path="selfie_multiclass_256x256.tflite", running_mode=VisionRunningMode.VIDEO, output_category_mask=True))
with ImageSegmenter.create_from_options(options) as segmenter:
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    segmenter.segment_async(mp_image, frame_timestamp_ms)