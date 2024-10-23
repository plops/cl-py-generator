import numpy as np
import cv2 as cv
import copy

_code_git_version = "30b5a88e76ee57aae97a49489d438e789bb6b047"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time = "06:28:34 of Wednesday, 2022-01-19 (GMT+1)"
cap = cv.VideoCapture("/dev/video2")
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
cv.resizeWindow("image", 256, 384)
cap.set(cv.CAP_PROP_CONVERT_RGB, 0)

# from  https://github.com/LeoDJ/P2Pro-Viewer
P2Pro_resolution = (256, 384)
P2Pro_fps = 25.0
P2Pro_usb_id = (0x0bda, 0x5830)  # VID, PID


while True:
    ret, image = cap.read()
    # image.shape => (384, 256, 2)
    if not (ret):
        break
    image.flags.writeable = False
    key = cv.waitKey(1)
    if (27) == (key):
        break
    frame = image

    # split video frame (top is pseudo color, bottom is temperature data)
    frame_mid_pos = int(len(frame) / 2)
    #picture_data = frame[0:frame_mid_pos]
    thermal_data = frame[frame_mid_pos:]

    # convert buffers to numpy arrays
    #yuv_picture = np.frombuffer(picture_data, dtype=np.uint8).reshape((P2Pro_resolution[1] // 2, P2Pro_resolution[0], 2))
    #rgb_picture = cv2.cvtColor(yuv_picture, cv2.COLOR_YUV2RGB_YUY2)

    thermal_picture_16 = np.frombuffer(thermal_data.ravel(), dtype=np.uint16).reshape((P2Pro_resolution[1] // 2, P2Pro_resolution[0]))

    ltemp = thermal_picture_16
    print(ltemp.min(), ltemp.max())
    ma = 19900 # ltemp.max()
    mi = 18700 # ltemp.min()
    #print(mi,ma)
    v = (ltemp-mi)/(ma-mi)
    scaled_image = cv.resize(v, (0, 0), fx=3, fy=3)

    cv.imshow("image", scaled_image)

cap.release()
cv.destroyAllWindows()

