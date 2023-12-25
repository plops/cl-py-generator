#!/usr/bin/env python3
# python -m venv ~/mediapipe_env; . ~/mediapipe_env/bin/activate; python -m pip install mediapipe mss
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
_code_git_version="aed7593e5862919772ceafdc2ed4205aa76ebcee"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/125_mediapipe_seg/source/"
_code_generation_time="02:34:19 of Monday, 2023-12-25 (GMT+1)"
roi=dict(top=100, left=100, width=640, height=480)
base_options=python.BaseOptions(model_aasset_path="deeplabv3.tflite")