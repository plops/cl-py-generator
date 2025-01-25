import cv2 as cv
import mss
import time
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("/home/martin/yolo/yolo11n.pt")
model = YOLO("/home/martin/yolo/yolo11s.pt")
model = YOLO("/home/martin/yolo/yolo11n-pose.pt")
#model = YOLO("/home/martin/yolo/yolo11m-pose.pt")
names = model.names

# capture screenshots and run yolo
start_time=time.time()
loop_time=time.time()
with mss.mss() as sct:
    loop_start=time.time()
    while (True):
        img=np.ascontiguousarray(np.array(sct.grab(dict(top=160, left=0, width=1000, height=700)))[:,:,0:3])

        timestamp_ms=int(((1000)*(((time.time())-(loop_start)))))
        annotator = Annotator(img,pil=False)
        #results = model.predict(img)
        # track needs micromamba install lap
        results = model.track(img,stream=False,persist=True)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        kpts = results[0].keypoints

        for box, cls in zip(boxes, clss):
            annotator.box_label(box, label=names[int(cls)])
        for i in range(kpts.xy.shape[0]):
            annotator.kpts(kpts.xy[i],shape=kpts.orig_shape)
        
        cv.imshow("Ultralytics circle annotation", img)
        #cv.imshow("screen", img)
        delta=((time.time())-(loop_time))
        target_period=((((1)/((60.    ))))-((1.00e-4)))
        if ( ((delta)<(target_period)) ):
            time.sleep(((target_period)-(delta)))
        fps=((1)/(delta))
        fps_wait=((1)/(((time.time())-(loop_time))))
        loop_time=time.time()
        if ( ((0)==(((timestamp_ms)%(2000)))) ):
            print("{} nil fps={} fps_wait={}".format(((time.time())-(start_time)), fps, fps_wait))
        if ( ((ord("q"))==(cv.waitKey(1))) ):
            cv.destroyAllWindows()
            break
