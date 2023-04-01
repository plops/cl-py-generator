#|default_exp p00_opencv_cl
import time
import cv2 as cv
start_time=time.time()
debug=True
_code_git_version="85416f36fecb3f1e82d1c1e5e4212e7be0780e0b"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="13:42:34 of Saturday, 2023-04-01 (GMT+1)"
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))