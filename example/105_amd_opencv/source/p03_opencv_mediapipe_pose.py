#!/usr/bin/python
#|default_exp p03_opencv_mediapipe_pose
# sudo pacman -S python-opencv rocm-opencl-runtime python-mss
import time
import numpy as np
import cv2 as cv
import mss
import mediapipe as mp
start_time=time.time()
debug=True
_code_git_version="d4be7d78df1910bb3d7c4ef4ca49726f196287e1"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="23:03:56 of Friday, 2023-04-14 (GMT+1)"
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_pose=mp.solutions.pose
clahe=cv.createCLAHE(clipLimit=(15.    ), tileGridSize=(32,18,))
with mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=(0.150    ), min_tracking_confidence=(0.150    )) as pose:
    # https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
    with mss.mss() as sct:
        loop_start=time.time()
        while (True):
            img=np.array(sct.grab(dict(top=160, left=0, width=((1920)//(2)), height=((1080)//(2)))))
            mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            timestamp_ms=int(((1000)*(((time.time())-(loop_start)))))
            imgr=cv.cvtColor(img, cv.COLOR_BGR2RGB)
            pose_results=pose.process(imgr)
            lab=cv.cvtColor(img, cv.COLOR_RGB2LAB)
            lab_planes=cv.split(lab)
            lclahe=clahe.apply(lab_planes[0])
            lab=cv.merge([lclahe, lab_planes[1], lab_planes[2]])
            imgr=cv.cvtColor(lab, cv.COLOR_LAB2RGB)
            if ( pose_results.pose_landmarks ):
                mp_drawing.draw_landmarks(imgr, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv.imshow("screen", imgr)
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