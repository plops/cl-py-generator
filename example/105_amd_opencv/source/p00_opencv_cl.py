#|default_exp p00_opencv_cl
# sudo pacman -S python-opencv rocm-opencl-runtime python-mss
import time
import numpy as np
import cv2 as cv
import mss
start_time=time.time()
debug=True
_code_git_version="041318f9d243c0857aace2a5f4467ebc74cc5e2a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="23:23:45 of Saturday, 2023-04-01 (GMT+1)"
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
clahe=cv.createCLAHE(clipLimit=(2.0    ), tileGridSize=(8,8,))
with mss.mss() as sct:
    while (True):
        img=np.array(sct.grab(dict(top=40, left=0, width=800, height=640)))
        imgr=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        cv.imshow("screen", clahe.apply(imgr))
        fps=((1)/(((time.time())-(loop_time))))
        loop_time=time.time()
        print("{} nil fps={}".format(((time.time())-(start_time)), fps))
        if ( ((ord("q"))==(cv.waitKey(1))) ):
            cv.destroyAllWindows()
            break