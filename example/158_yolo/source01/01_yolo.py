import cv2 as cv
import numpy as np
#from PIL import Image
from ultralytics import YOLO

model = YOLO('/home/martin/yolo/yolo11n-pose.pt')
fn = '/home/martin/yolo/bus.jpg'
img = cv.imread(fn)
results = model.predict(source=img,save=True)

