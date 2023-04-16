#!/usr/bin/python
#|default_exp p04_opencv_mediapipe_face
# sudo pacman -S python-opencv rocm-opencl-runtime python-mss
import time
import numpy as np
import cv2 as cv
import mss
import mediapipe as mp
start_time=time.time()
debug=True
_code_git_version="82e6432ae1954d4b9052235500c4482b984632b4"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="14:05:30 of Sunday, 2023-04-16 (GMT+1)"
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_face_mesh=mp.solutions.face_mesh
clahe=cv.createCLAHE(clipLimit=(15.    ), tileGridSize=(32,18,))
drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
gpu_options=mp.GpuAccelerationOptions(enable=True, device_id=0)
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, gpu_options=gpu_options, refine_landmarks=True, min_detection_confidence=(0.150    ), min_tracking_confidence=(0.150    )) as face_mesh:
    # https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
# select the attention model using refine_landmarks option to home improve accuracy around lips, eyes and irises
    with mss.mss() as sct:
        loop_start=time.time()
        while (True):
            img=np.array(sct.grab(dict(top=160, left=0, width=((1920)//(2)), height=((1080)//(2)))))
            timestamp_ms=int(((1000)*(((time.time())-(loop_start)))))
            imgr=cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results=face_mesh.process(imgr)
            lab=cv.cvtColor(img, cv.COLOR_RGB2LAB)
            lab_planes=cv.split(lab)
            lclahe=clahe.apply(lab_planes[0])
            lab=cv.merge([lclahe, lab_planes[1], lab_planes[2]])
            imgr=cv.cvtColor(lab, cv.COLOR_LAB2RGB)
            if ( results.multi_face_landmarks ):
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=imgr, landmark_list=face_landmarks, landmark_drawing_spec=None, connections=mp_face_mesh.FACEMESH_TESSELATION, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(image=imgr, landmark_list=face_landmarks, landmark_drawing_spec=None, connections=mp_face_mesh.FACEMESH_CONTOURS, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(image=imgr, landmark_list=face_landmarks, landmark_drawing_spec=None, connections=mp_face_mesh.FACEMESH_IRISES, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
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