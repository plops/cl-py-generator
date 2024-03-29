import numpy as np
import cv2 as cv
import copy

_code_git_version = "30b5a88e76ee57aae97a49489d438e789bb6b047"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time = "06:28:34 of Wednesday, 2022-01-19 (GMT+1)"
cap = cv.VideoCapture("/dev/video0")
r = cap.set(cv.CAP_PROP_MODE, 0)
if not (r):
    print("problem with MODE")
r = cap.get(cv.CAP_PROP_MODE)
print("MODE={}".format(r))
r = cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
if not (r):
    print("problem with FRAME_WIDTH")
r = cap.get(cv.CAP_PROP_FRAME_WIDTH)
print("FRAME_WIDTH={}".format(r))
r = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
if not (r):
    print("problem with FRAME_HEIGHT")
r = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print("FRAME_HEIGHT={}".format(r))
r = cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
if not (r):
    print("problem with AUTO_EXPOSURE")
r = cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
print("AUTO_EXPOSURE={}".format(r))
r = cap.set(cv.CAP_PROP_EXPOSURE, 50)
if not (r):
    print("problem with EXPOSURE")
r = cap.get(cv.CAP_PROP_EXPOSURE)
print("EXPOSURE={}".format(r))
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.resizeWindow("image", ((1920) // (2)), 1080)
while True:
    ret, image = cap.read()
    if not (ret):
        break
    image.flags.writeable = False
    key = cv.waitKey(1)
    if (27) == (key):
        break
    cv.imshow("image", image)
cap.release()
cv.destroyAllWindows()
