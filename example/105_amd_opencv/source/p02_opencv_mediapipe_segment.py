#|default_exp p02_opencv_mediapipe_segment
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
_code_git_version="fe5991e8e673fa16f0e56b68c53696b9568f1977"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/105_amd_opencv/source/"
_code_generation_time="19:38:22 of Sunday, 2023-04-02 (GMT+1)"
model_path="/home/martin/Downloads/deeplabv3.tflite"
BaseOptions=mp.tasks.BaseOptions
ImageSegmenter=mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions=mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode=mp.tasks.vision.RunningMode
gResult=None
oldResult=None
def print_result(result: list[mp.Image], output_image: mp.Image, timestamp_ms: int):
    print("{} result len(result)={}".format(((time.time())-(start_time)), len(result)))
    global gResult
    gResult=result
# output can be category .. single uint8 for every pixel
# or confidence_mask .. several float images with range [0,1]
options=ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.LIVE_STREAM, output_type=ImageSegmenterOptions.OutputType.CATEGORY_MASK, result_callback=print_result)
def view(category, imgr):
    category_edges=cv.Canny(image=category, threshold1=10, threshold2=1)
    imgr[((category_edges)!=(0))]=[0, 0, 255]
    cv.imshow("screen", imgr)
print("{} nil cv.ocl.haveOpenCL()={}".format(((time.time())-(start_time)), cv.ocl.haveOpenCL()))
loop_time=time.time()
clahe=cv.createCLAHE(clipLimit=(15.    ), tileGridSize=(32,18,))
with ImageSegmenter.create_from_options(options) as segmenter:
    with mss.mss() as sct:
        loop_start=time.time()
        while (True):
            img=np.array(sct.grab(dict(top=160, left=0, width=((1920)//(2)), height=((1080)//(2)))))
            mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            timestamp_ms=int(((1000)*(((time.time())-(loop_start)))))
            segmenter.segment_async(mp_image, timestamp_ms)
            lab=cv.cvtColor(img, cv.COLOR_RGB2LAB)
            lab_planes=cv.split(lab)
            lclahe=clahe.apply(lab_planes[0])
            lab=cv.merge([lclahe, lab_planes[1], lab_planes[2]])
            imgr=cv.cvtColor(lab, cv.COLOR_LAB2RGB)
            if ( (gResult is None) ):
                if ( (oldResult is None) ):
                    cv.imshow("screen", imgr)
                else:
                    view(oldResult[0].numpy_view(), imgr)
            else:
                view(gResult[0].numpy_view(), imgr)
                oldResult=gResult
                gResult=None
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