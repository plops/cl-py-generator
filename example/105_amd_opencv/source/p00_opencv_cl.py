#|default_exp p00_opencv_cl
# sudo pacman -S python-opencv rocm-opencl-runtime python-mss
import time
import numpy as np
import cv2 as cv
import mss
start_time=time.time()
debug=True
_code_git_version="36478221d3569c8c2be1328529e1c5c2eecdfcac"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="23:31:28 of Saturday, 2023-04-01 (GMT+1)"
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
clahe=cv.createCLAHE(clipLimit=(2.0    ), tileGridSize=(8,8,))
with mss.mss() as sct:
    while (True):
        img=np.array(sct.grab(dict(top=40, left=0, width=800, height=640)))
        lab=cv.cvtColor(img, cv.COLOR_RGB2LAB)
        lab_planes=cv.split(lab)
        lclahe=clahe.apply(lab_planes[0])
        lab=cv.merge([lclahe, lab_planes[1], lab_planes[2]])
        imgr=cv.cvtColor(lab, cv.COLOR_LAB2RGB)
        cv.imshow("screen", imgr)
        fps=((1)/(((time.time())-(loop_time))))
        loop_time=time.time()
        print("{} nil fps={}".format(((time.time())-(start_time)), fps))
        if ( ((ord("q"))==(cv.waitKey(1))) ):
            cv.destroyAllWindows()
            break