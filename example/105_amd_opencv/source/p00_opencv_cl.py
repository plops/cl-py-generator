#|default_exp p00_opencv_cl
# sudo pacman -S python-opencv rocm-opencl-runtime python-mss
import time
import numpy as np
import cv2 as cv
import mss
start_time=time.time()
debug=True
_code_git_version="5fc2f582a0a52c3985725e8d78c1c875f2f9b892"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="11:17:35 of Sunday, 2023-04-02 (GMT+1)"
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
clahe=cv.createCLAHE(clipLimit=(7.0    ), tileGridSize=(12,12,))
with mss.mss() as sct:
    loop_start=time.time()
    count=0
    while (True):
        count += 1
        img=np.array(sct.grab(dict(top=160, left=0, width=1000, height=740)))
        lab=cv.cvtColor(img, cv.COLOR_RGB2LAB)
        lab_planes=cv.split(lab)
        lclahe=clahe.apply(lab_planes[0])
        lab=cv.merge([lclahe, lab_planes[1], lab_planes[2]])
        imgr=cv.cvtColor(lab, cv.COLOR_LAB2RGB)
        cv.imshow("screen", imgr)
        delta=((time.time())-(loop_time))
        target_period=((1)/((60.    )))
        if ( ((delta)<(target_period)) ):
            time.sleep(((((target_period)-(delta)))-((1.00e-4))))
        fps=((1)/(delta))
        fps_wait=((1)/(((time.time())-(loop_time))))
        loop_time=time.time()
        if ( ((0)==(((count)%(100)))) ):
            print("{} nil fps={} fps_wait={}".format(((time.time())-(start_time)), fps, fps_wait))
        if ( ((ord("q"))==(cv.waitKey(1))) ):
            cv.destroyAllWindows()
            break