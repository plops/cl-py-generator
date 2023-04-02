#|default_exp p01_opencv_mediapipe
# sudo pacman -S python-opencv rocm-opencl-runtime python-mss
import time
import numpy as np
import cv2 as cv
import mss
import mediapipe as mp
import mediapipe.tasks
import mediapipe.tasks.python
start_time=time.time()
debug=True
_code_git_version="5ac1d876333aa6242db8475ba611f702db23bc1c"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="11:25:28 of Sunday, 2023-04-02 (GMT+1)"
model_path="/home/martin/Downloads/efficientdet_lite0_uint8.tflite"
BaseOptions=mp.tasks.BaseOptions
DetectionResult=mp.tasks.components.containers.DetectionResult
ObjectDetector=mp.tasks.vision.ObjectDetector
ObjectDetectorOptions=mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode=mp.tasks.vision.RunningMode
gResult=None
def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print("{} result ".format(((time.time())-(start_time))))
    gResult=result
options=ObjectDetectorOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.LIVE_STREAM, max_results=5, result_callback=print_result)
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
clahe=cv.createCLAHE(clipLimit=(7.0    ), tileGridSize=(12,12,))
with ObjectDetector.create_from_options(options) as detector:
    with mss.mss() as sct:
        loop_start=time.time()
        while (True):
            img=np.array(sct.grab(dict(top=160, left=0, width=1000, height=740)))
            mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            timestamp_ms=int(((1000)*(((time.time())-(loop_start)))))
            detector.detect_async(mp_image, timestamp_ms)
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
            if ( ((0)==(((timestamp_ms)%(2000)))) ):
                print("{} nil fps={} fps_wait={}".format(((time.time())-(start_time)), fps, fps_wait))
            if ( ((ord("q"))==(cv.waitKey(1))) ):
                cv.destroyAllWindows()
                break