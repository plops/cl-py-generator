#!/usr/bin/env python3
# load a slow-mo video from iphone that with structured illumination
import cv2 as cv
# load video file
fn="IMG_3355.mov"
v=cv.VideoCapture(fn)
if ( not(v.isOpened()) ):
    print("video could not be opened fn={}".format(fn))
w=v.get(cv.CAP_PROP_FRAME_WIDTH)
h=v.get(cv.CAP_PROP_FRAME_HEIGHT)
win="screen"
cv.namedWindow(win, cv.WINDOW_AUTOSIZE)
while (True):
    a, frame=v.read()
    if ( (frame is None) ):
        break
    cv.imshow(win, frame)
    k=cv.waitKey(30)
    if ( ((k)==(27)) ):
        break