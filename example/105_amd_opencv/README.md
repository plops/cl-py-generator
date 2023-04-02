
| file  | comment                                                      |
|-------|--------------------------------------------------------------|
| gen00 | capture screenshot, perform clahe and display (runs at 60Hz) |
| gen01 | clahe as in gen00 but also mediapipe                         |
|       |                                                              |


- pip installed mediapipe 0.9.2.1
- i am not sure where the correct website is
- https://mediapipe.dev/
- https://google.github.io/mediapipe/
- https://developers.google.com/mediapipe seems to be new starting
  from 2023-04-03 but there are no examples for holistic human model
  detector, yet
- i will try the object detection with efficientnet
  https://developers.google.com/mediapipe/solutions/vision/object_detector#get_started
- python guide:
  https://developers.google.com/mediapipe/solutions/vision/object_detector/python
- python code:
  https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb
- model download: https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models
```
wget https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite
```
- i capture my laptop's screen with 60Hz, mediapipe distinguishes
  image, video and live stream input data type. i think i want to use
  live stream. this adds a monotonically increasing integer
  timestamp_ms as an identitifier to the API and results are returned
  with a callback (or skipped if the last frame hasn't been processed
  yet)


