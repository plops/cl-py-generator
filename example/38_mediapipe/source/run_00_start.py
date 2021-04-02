import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import xarray as xr
import xarray.plot as xrp
import scipy.optimize
import numpy as np
import cv2 as cv
import mediapipe as mp
import copy
from mss import mss
_code_git_version="d1fe65da2a4f27df5fe3627d03a0c12224220710"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="21:47:42 of Tuesday, 2021-03-30 (GMT+1)"
def calc_bounding_rect(image, landmarks):
    height, width=image.shape[0:2]
    landmark_array=np.empty((0,2,), int)
    for landmark in landmarks.landmark:
        lx=min(int(((landmark.x)*(width))), ((width)-(1)))
        ly=min(int(((landmark.y)*(height))), ((height)-(1)))
        landmark_point=[np.array((lx,ly,))]
        landmark_array=np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h=cv.boundingRect(landmark_array)
    return [x, y, ((x)+(w)), ((y)+(h))]
def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0],brect[1],), (brect[2],brect[3],), (0,255,0,), 2)
    return image
def draw_landmarks(image, landmarks):
    height, width=image.shape[0:2]
    for l in landmarks.landmark:
        if ( ((((l.visibility)<(0))) or (((l.presence)<(0)))) ):
            continue
        lx=min(int(((l.x)*(width))), ((width)-(1)))
        ly=min(int(((l.y)*(height))), ((height)-(1)))
        cv.circle(image, (lx,ly,), 1, (0,255,0,), 1)
    return image
bbox={("top"):(180),("left"):(10),("width"):(512),("height"):(512)}
sct=mss()
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(max_num_faces=3, min_detection_confidence=(0.70    ), min_tracking_confidence=(0.50    ))
while (True):
    sct_img=sct.grab(bbox)
    image=np.array(sct_img)
    debug_image=copy.deepcopy(image)
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results=face_mesh.process(image)
    if ( not((results.multi_face_landmarks is None)) ):
        for face_landmarks in results.multi_face_landmarks:
            brect=calc_bounding_rect(debug_image, face_landmarks)
            debug_image=draw_landmarks(debug_image, face_landmarks)
    key=cv.waitKey(1)
    if ( ((27)==(key)) ):
        break
    cv.imshow("face mesh", debug_image)
cap.release()
cv.destroyAllWindows()