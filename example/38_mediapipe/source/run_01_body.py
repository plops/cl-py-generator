import numpy as np
import cv2 as cv
import mediapipe as mp
import copy
from mss import mss
_code_git_version="d836113e90419830b996dc14d91b5fcd8f4617b4"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="22:52:45 of Saturday, 2021-04-03 (GMT+1)"
cam=False
if ( cam ):
    cap=cv.VideoCapture("/dev/video0")
bbox={("top"):(180),("left"):(10),("width"):(512),("height"):(512)}
sct=mss()
mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
holistic=mp_holistic.Holistic(min_detection_confidence=(0.70    ), min_tracking_confidence=(0.50    ))
drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cv.namedWindow("image", cv.WINDOW_NORMAL)
while (True):
    if ( cam ):
        ret, image=cap.read()
        if ( not(ret) ):
            break
    else:
        sct_img=sct.grab(bbox)
        image=np.array(sct_img)
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=holistic.process(image)
    image.flags.writeable=True
    image=cv.cvtColor(image, cv.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image=image, landmark_list=results.face_landmarks, connections=mp_holistic.FACE_CONNECTIONS, connection_drawing_spec=drawing_spec)
    mp_drawing.draw_landmarks(image=image, landmark_list=results.right_hand_landmarks, connections=mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=drawing_spec)
    mp_drawing.draw_landmarks(image=image, landmark_list=results.left_hand_landmarks, connections=mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=drawing_spec)
    mp_drawing.draw_landmarks(image=image, landmark_list=results.pose_landmarks, connections=mp_holistic.POSE_CONNECTIONS, connection_drawing_spec=drawing_spec)
    key=cv.waitKey(1)
    if ( ((27)==(key)) ):
        break
    cv.imshow("image", image)
if ( cam ):
    cap.release()
cv.destroyAllWindows()
