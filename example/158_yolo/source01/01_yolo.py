import cv2 as cv
import numpy as np
#from PIL import Image
from ultralytics import YOLO

model = YOLO('/home/martin/yolo/yolo11n-pose.pt')
#model = YOLO('/home/martin/yolo/yolo11m-pose.pt')
#fn = '/home/martin/yolo/bus.jpg'
#src = cv.imread(fn)
src = '/dev/shm/a.mp4'
results = model.predict(source=src,show=True,vid_stride=10,imgsz=(1280//2,720//2),stream_buffer=False,visualize=False)
#results = model.track(source=src,show=True) # lap module missing

